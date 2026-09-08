import base64
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HPC = Path(__file__).resolve().parents[1]
def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HPC / filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
s = module("submit_release_test", "submit_release.py")
w = module("release_worker_test", "release_worker.py")
SHA = "a5e7f843856d17b7647f8af6571319b45719f669"
ARGS = ["--commit", SHA, "--mode", "inspect", "--dataset-root",
        "/scratch/licenseddata/imagenet/imagenet-1k", "--cpus", "1",
        "--memory", "2G", "--time", "00:05:00", "--partition", "general", "--qos", "debug"]

class ReleaseTests(unittest.TestCase):
    def test_dry_run_never_contacts_remote(self):
        with patch.object(s.subprocess, "run") as run:
            self.assertEqual(s.main(ARGS), 0)
            run.assert_not_called()

    def test_generated_script_valid_and_refuses_without_allocation(self):
        plan, worker = s.build_plan(s.arguments(ARGS))
        script = s.render(plan, worker)
        self.assertNotIn("#SBATCH --gres", script)
        self.assertIn(SHA + "/code", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        env = os.environ.copy()
        env.pop("SLURM_JOB_ID", None)
        result = subprocess.run(["bash"], input=script, text=True, capture_output=True, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SLURM_JOB_ID", result.stderr)

    def test_submit_only_sends_sbatch_after_master_check(self):
        with patch.object(s.subprocess, "run", side_effect=[
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0, "12345\n", ""),
        ]) as run:
            s.main(ARGS + ["--submit"])
            self.assertEqual(run.call_count, 2)
            command = run.call_args_list[1].args[0]
            self.assertEqual(command[-1], "sbatch --parsable --chdir=/scratch/user/uqcche38")
            for flag in ("ProxyCommand=/bin/false", "BatchMode=yes", "ClearAllForwardings=yes"):
                self.assertIn(flag, command)

    def test_invalid_identity_resource_or_path_rejected(self):
        for option, value in [("--commit", "main"), ("--partition", "general\nx"),
                              ("--memory", "2G;id"), ("--dataset-root", "/scratch/../home")]:
            args = ARGS.copy()
            args[args.index(option) + 1] = value
            with self.subTest(option=option), self.assertRaises(ValueError):
                s.build_plan(s.arguments(args))
        with self.assertRaises(ValueError):
            s.build_plan(s.arguments(ARGS + ["--gpu", "l40s:1"]))

    def test_reference_template_cannot_accidentally_run(self):
        config = json.loads((HPC / "configs/golden_retriever_reference.template.json").read_text())
        with self.assertRaises(ValueError):
            s.validate_reference(config, 4)
        config.update(source_dir="/scratch/prepared", dataset_manifest="/scratch/manifest.json",
                      torch_num_threads=4, batch_size=4)
        config["segmentation"]["checkpoint"]="/scratch/sam_vit_h.pth"
        self.assertEqual(s.validate_reference(config,4)["seed"],43)
        config["seed"]=44
        with self.assertRaises(ValueError):
            s.validate_reference(config,4)

    def test_dataset_manifest_exact_counts_hashes_and_disjointness(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = {"synset": "n02099601", "seed": 43, "source_dir": str(root)}
            for split,folder,count in [
                ("training",root/"golden_retriever",400),
                ("validation",root/"val_imgs/golden_retriever_val",50)]:
                folder.mkdir(parents=True)
                manifest[split]=[]
                for i in range(count):
                    p=folder/f"{i:03}.jpeg"
                    p.write_bytes(f"{split}-{i}".encode())
                    manifest[split].append({"prepared_name":p.name,"sha256":w.sha256(p)})
            mp=root/"manifest.json"
            mp.write_text(json.dumps(manifest))
            config={"dataset_manifest":str(mp),"source_dir":str(root)}
            self.assertEqual(w.verify_inputs(config)["duplicate_contents"],0)
            p=root/"val_imgs/golden_retriever_val/000.jpeg"
            p.write_bytes((root/"golden_retriever/000.jpeg").read_bytes())
            manifest["validation"][0]["sha256"]=w.sha256(p)
            mp.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"Duplicate"):
                w.verify_inputs(config)

    def test_inspection_listing_is_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            for i in range(5): (Path(td)/str(i)).touch()
            result=w.listing(td,2)
            self.assertTrue(result["truncated"])
            self.assertEqual(len(result["entries"]),2)


class DataPreparationTests(unittest.TestCase):
    def test_seed_fixed_validation_and_overlap_stop(self):
        d=module("prepare_data_test","prepare_reference_data.py")
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root=base/"licensed"
            for split,count in (("train",8),("val",2)):
                folder=root/split/"n02099601";folder.mkdir(parents=True)
                for i in range(count): (folder/f"{i}.JPEG").write_bytes(f"{split}-{i}".encode())
            result=d.prepare(root,base/"first",43,3,2,lambda p: None)
            d.prepare(root,base/"second",43,3,2,lambda p: None)
            a=json.loads((base/"first/dataset_manifest.json").read_text())
            b=json.loads((base/"second/dataset_manifest.json").read_text())
            self.assertEqual(a["training"],b["training"])
            self.assertEqual(a["validation"],b["validation"])
            self.assertEqual(len((base/"first/training_images.txt").read_text().splitlines()),3)
            victim=root/"val/n02099601/0.JPEG"
            victim.write_bytes(Path(a["training"][0]["source"]).read_bytes())
            with self.assertRaisesRegex(ValueError,"Duplicate"):
                d.prepare(root,base/"overlap",43,3,2,lambda p: None)
            self.assertFalse((base/"overlap").exists())
            with self.assertRaises(ValueError):
                d.prepare(root,root/"forbidden",43,3,2,lambda p: None)

    def test_bad_selected_image_is_not_silently_replaced(self):
        d=module("prepare_data_bad_test","prepare_reference_data.py")
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root=base/"licensed"
            for split in ("train","val"):
                folder=root/split/"n02099601";folder.mkdir(parents=True)
                (folder/"0.JPEG").write_bytes(split.encode())
            def bad(path): raise ValueError("invalid image")
            with self.assertRaisesRegex(ValueError,"invalid image"):
                d.prepare(root,base/"out",43,1,1,bad)
            self.assertFalse((base/"out").exists())

class ReferenceCollectorTests(unittest.TestCase):
    def test_reference_requires_pinned_commit_and_layout(self):
        c=module("reference_collector_test","collect_results.py")
        with self.assertRaises(SystemExit),patch("sys.stderr"):
            c.arguments(["123","--profile","reference"])
        args=c.arguments(["123","--profile","reference","--run-layout","--expected-commit",SHA])
        self.assertEqual(args.output_subdir,"runs/123")
        with self.assertRaises(SystemExit),patch("sys.stderr"):
            c.arguments(["123","--run-layout","--expected-commit","main"])

class ReferenceCollectionIntegrationTests(unittest.TestCase):
    def test_reference_identity_and_launcher_binding(self):
        import hashlib
        c=module("reference_collector_integration","collect_results.py")
        identity={"run_id":"123","slurm_job_id":"123"}
        class Transport:
            bad_research=False
            bad_launcher=False
            def check(self): pass
            def query(self,command):
                return "JobID|JobName|State\n123|humcd-reference|COMPLETED\n" if command.startswith("sacct") else ""
            def download(self,remote,local):
                name=remote.rsplit("/",1)[-1]
                summary={"status":"PASS",**identity,
                         "prototype_quality_proxy":{"validation":{"learned_concept_assignment_rate":0.2}}}
                content={n:(json.dumps(summary) if n=="summary.json" else "fixture").encode()
                         for n in c.PROFILES["reference"]["files"]}
                if name=="launch_manifest.json":
                    data=json.dumps({**identity,"actual_commit":"0"*40 if self.bad_launcher else SHA,
                                     "plan":{"commit":SHA}}).encode()
                elif name=="run_manifest.json":
                    data=json.dumps({**identity,"git_commit":"0"*40 if self.bad_research else SHA,
                        "git_dirty":False,"resolved_config_sha256":hashlib.sha256(b"{}").hexdigest()}).encode()
                elif name=="resolved_config.json": data=b"{}"
                elif name=="artifacts.json":
                    data=json.dumps({**identity,"files":{n:{"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
                                                          for n,b in content.items()}}).encode()
                elif name=="progress.json": data=json.dumps({**identity,"status":"PASS"}).encode()
                elif name=="summary.json":data=content[name]
                else:data=b"fixture"
                local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(data);return True
        with tempfile.TemporaryDirectory() as td:
            args=c.arguments(["123","--profile","reference","--run-layout","--expected-commit",SHA,
                              "--output-root",td])
            t=Transport()
            _,good=c.collect(args,t)
            self.assertEqual(good["collection_status"],"complete")
            self.assertEqual(good["launcher_provenance"]["actual_commit"],SHA)
            t.bad_research=True
            _,bad=c.collect(args,t)
            self.assertEqual(bad["collection_status"],"partial")
            t.bad_research=False;t.bad_launcher=True
            _,bad=c.collect(args,t)
            self.assertEqual(bad["collection_status"],"partial")

class SelectionAuditTests(unittest.TestCase):
    def test_audit_preserves_all_selected_ids_despite_format_errors(self):
        d=module("selection_audit_test","prepare_reference_data.py")
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root=base/"licensed";audit=base/"audit";audit.mkdir()
            for split in ("train","val"):
                folder=root/split/"n02099601";folder.mkdir(parents=True)
                for i in range(3): (folder/f"{i}.JPEG").write_bytes(f"{split}-{i}".encode())
            def check(path):
                if path.name=="0.JPEG": raise ValueError("grayscale")
            result=d.prepare(root,base/"prepared",43,3,3,check,audit,True)
            data=json.loads((audit/"selection_audit.json").read_text())
            self.assertEqual(len(data["training"]),3)
            self.assertEqual(len(data["validation"]),3)
            self.assertEqual(len(result["problems"]),2)
            self.assertFalse((base/"prepared").exists())
            self.assertEqual(len((audit/"training_images.txt").read_text().splitlines()),3)

class ProbeGateTests(unittest.TestCase):
    def test_failed_probe_and_changed_inputs_cannot_start_reference(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"launch.json"
            cfg={"dataset_manifest_sha256":"a"*64,"sam_checkpoint_sha256":"b"*64,
                 "resnet_checkpoint_sha256":"c"*64,"batch_size":8}
            probe={"status":"PROBE_COMPLETED","probe_report":{"status":"PASS"},
                   "slurm_job_id":"123","plan":{"config":cfg}}
            path.write_text(json.dumps(probe))
            plan={"probe_launch":str(path),"probe_launch_sha256":w.sha256(path)}
            self.assertEqual(w.verify_probe(plan,cfg),"123")
            with self.assertRaises(ValueError):
                w.verify_probe(plan,{**cfg,"dataset_manifest_sha256":"0"*64})
            with self.assertRaises(ValueError):
                w.verify_probe(plan,{**cfg,"batch_size":16})
            probe["status"]="FAILED"
            path.write_text(json.dumps(probe))
            plan["probe_launch_sha256"]=w.sha256(path)
            with self.assertRaises(ValueError):
                w.verify_probe(plan,cfg)
