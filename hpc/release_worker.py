"""Compute-node bootstrap sent by submit_release.py, outside the pinned research tree."""
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def listing(path, limit=30):
    """Bounded, nonrecursive metadata inspection; no images are decoded."""
    path = Path(path)
    if not path.is_dir():
        return {"path": str(path), "exists": path.exists(), "is_directory": False}
    entries = []
    with os.scandir(path) as items:
        for item in items:
            if len(entries) == limit:
                return {"path": str(path), "entries": entries, "truncated": True}
            st = item.stat(follow_symlinks=False)
            entries.append({"name": item.name, "bytes": st.st_size,
                            "directory": item.is_dir(follow_symlinks=False),
                            "symlink": item.is_symlink()})
    return {"path": str(path), "entries": entries, "truncated": False}


def verify_inputs(config):
    """Consume an explicit prepared manifest; never infer ImageNet label mapping."""
    manifest_path = Path(config["dataset_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if config.get("dataset_manifest_sha256") and sha256(manifest_path) != config["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest identity mismatch")
    if manifest["synset"] != "n02099601" or manifest["seed"] != 43:
        raise ValueError("Expected golden_retriever and the user-approved seed 43")
    if manifest["source_dir"] != config["source_dir"]:
        raise ValueError("Prepared data root does not match config")
    if manifest.get("ready_for_reference") is False or manifest.get("validation_issues"):
        raise ValueError("Prepared selection has unresolved validation format issues; no implicit replacement/conversion")
    all_paths, all_hashes = set(), set()
    root = Path(config["source_dir"])
    for split, count, folder in (
        ("training", 400, root / "golden_retriever"),
        ("validation", 50, root / "val_imgs/golden_retriever_val"),
    ):
        entries = manifest[split]
        if len(entries) != count:
            raise ValueError("Wrong manifest image count: " + split)
        actual_names = sorted(p.name for p in folder.iterdir())
        if actual_names != sorted(e["prepared_name"] for e in entries):
            raise ValueError("Prepared folder must contain exactly the manifest images")
        for entry in entries:
            if Path(entry["prepared_name"]).name != entry["prepared_name"]:
                raise ValueError("Invalid prepared filename")
            path = folder / entry["prepared_name"]
            canonical = str(path.resolve(strict=True))
            digest = sha256(path)
            if digest != entry.get("input_sha256", entry["sha256"]):
                raise ValueError("Image content changed: " + str(path))
            if "input_path" in entry:
                if str(path) != entry["input_path"] or sha256(entry["source"]) != entry["sha256"]:
                    raise ValueError("Actual/original input mapping changed")
            if canonical in all_paths or digest in all_hashes:
                raise ValueError("Duplicate content/path within or between splits")
            all_paths.add(canonical)
            all_hashes.add(digest)
    return {"dataset_manifest_sha256": sha256(manifest_path),
            "training_count": 400, "validation_count": 50,
            "duplicate_paths": 0, "duplicate_contents": 0}


def verify_probe(plan, config):
    if not plan.get("after_probe") and sha256(plan["probe_launch"]) != plan["probe_launch_sha256"]:
        raise ValueError("Probe launch identity changed")
    probe = json.loads(Path(plan["probe_launch"]).read_text())
    if plan.get("after_probe"):
        if str(probe.get("slurm_job_id")) != plan["after_probe"] or probe.get("actual_commit") != plan["probe_commit"]:
            raise ValueError("Deferred probe job/commit identity mismatch")
        if probe.get("mode") != "resource-probe":
            raise ValueError("Deferred prerequisite is not the requested probe")
        for key, expected in plan["expected_probe_helpers"].items():
            if probe.get("plan", {}).get(key) != expected:
                raise ValueError("Deferred probe helper identity mismatch")
    if probe.get("status") != "PROBE_COMPLETED" or probe.get("probe_report", {}).get("status") != "PASS":
        raise ValueError("A completed successful resource/correctness probe is required")
    for key in ("dataset_manifest_sha256", "sam_checkpoint_sha256", "resnet_checkpoint_sha256"):
        if not config.get(key) or config[key] != probe["plan"]["config"].get(key):
            raise ValueError("Reference inputs differ from verified probe: " + key)
    if config["batch_size"] > probe["plan"]["config"]["batch_size"]:
        raise ValueError("Reference batch size exceeds the tested probe batch")
    return probe["slurm_job_id"]


def main():
    plan = json.loads(base64.b64decode(sys.argv[1], validate=True))
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or not __import__("re").fullmatch(r"bun[0-9]{3}", socket.gethostname().split(".")[0]):
        raise RuntimeError("Requires a Bunya Slurm compute allocation")
    release = Path(plan["release"])
    actual = subprocess.check_output(["git", "-C", str(release), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(release), "status", "--porcelain"], text=True).strip()
    if actual != plan["commit"] or dirty:
        raise RuntimeError("Prepared release is wrong or dirty")
    root = Path(plan["runtime_root"])
    launch = root / "launches" / job
    launch.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "job_id": job, "run_id": job,
              "slurm_job_id": job, "actual_commit": actual, "mode": plan["mode"],
              "host": socket.gethostname(), "plan": plan,
              "status": "STARTED", "python": sys.version}
    path = launch / "launch_manifest.json"
    write_json(path, record)
    try:
        versions = {}
        for name in ("torch", "torchvision", "timm", "numpy", "scipy", "scikit-learn",
                     "scikit-image", "scikit-dimension", "segment-anything"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        spec = importlib.util.find_spec("timm")
        installed = Path(spec.origin).parent / "models" if spec else None
        masking = {}
        for name in ("resnet.py", "sal_layers.py"):
            target = installed / name if installed else None
            expected = sha256(release / "input_masking" / name)
            found = sha256(target) if target and target.is_file() else None
            masking[name] = {"expected_sha256": expected, "installed_sha256": found,
                             "match": expected == found}
        record.update(versions=versions, masking=masking)
        if plan["mode"] == "inspect":
            source = Path(plan["dataset_root"])
            record["dataset"] = {name: listing(source / name) for name in
                                 ("", "train", "val", "validation", "devkit",
                                  "train/n02099601", "val/n02099601")}
            record["models"] = listing(root / "models")
            weights = root / "models/torch/hub/checkpoints/resnet50_a1_0-14fe96d1.pth"
            record["resnet_checkpoint"] = {"path": str(weights), "exists": weights.is_file(),
                                          "sha256": sha256(weights) if weights.is_file() else None}
            record["status"] = "INSPECTED"
            record["scientific_readiness"] = "UNASSESSED"
            print(json.dumps(record, indent=2))
        elif plan["mode"] in ("compatibility", "prepare-model"):
            code = base64.b64decode(plan["preparation_worker_base64"], validate=True)
            if hashlib.sha256(code).hexdigest() != plan["preparation_worker_sha256"]:
                raise ValueError("Preparation worker identity mismatch")
            scope = {"__name__": "humcd_reference_preparation"}
            exec(compile(code, "transmitted_reference_preparation.py", "exec"), scope)
            if plan["mode"] == "compatibility":
                record["prepared_data"] = scope["compatibility"](
                    plan["dataset_manifest"], plan["dataset_manifest_sha256"], plan["prepared_root"])
                record["status"] = "INPUTS_READY"
            else:
                record["model_preparation"] = scope["prepare_model"](root / "models", launch)
                record["status"] = "MODEL_DOWNLOADED"
            print(json.dumps({k:v for k,v in record.items() if k in ("prepared_data", "model_preparation", "status")}, indent=2))
        elif plan["mode"] in ("audit-training", "publish-training"):
            code = base64.b64decode(plan["data_worker_base64"], validate=True)
            if hashlib.sha256(code).hexdigest() != plan["data_worker_sha256"]:
                raise ValueError("Training worker identity mismatch")
            scope = {"__name__": "humcd_training_audit"}
            exec(compile(code, "transmitted_audit_training_candidates.py", "exec"), scope)
            if plan["mode"] == "audit-training":
                record["training_audit"] = scope["audit"](
                    plan["dataset_root"], plan["selection_path"], plan["selection_sha256"], launch, plan["seed"])
                record["status"] = "TRAINING_AUDITED"
            else:
                record["prepared_data"] = scope["publish"](
                    plan["selection_path"], plan["selection_sha256"], plan["prepared_root"])
                record["status"] = "DATA_PREPARED" if record["prepared_data"]["ready_for_reference"] else "TRAINING_PREPARED_VALIDATION_BLOCKED"
            print(json.dumps({k:v for k,v in record.items() if k in ("training_audit", "prepared_data", "status")}, indent=2))
        elif plan["mode"] in ("prepare-data", "inspect-selection"):
            code = base64.b64decode(plan["data_worker_base64"], validate=True)
            if hashlib.sha256(code).hexdigest() != plan["data_worker_sha256"]:
                raise ValueError("Data worker identity mismatch")
            scope = {"__name__": "humcd_data_preparation"}
            exec(compile(code, "transmitted_prepare_reference_data.py", "exec"), scope)
            record["prepared_data"] = scope["prepare"](
                plan["dataset_root"], plan["prepared_root"], plan["seed"],
                audit_dir=launch, audit_only=plan["mode"] == "inspect-selection")
            record["status"] = "SELECTION_AUDITED" if plan["mode"] == "inspect-selection" else "DATA_PREPARED"
            print(json.dumps(record["prepared_data"], indent=2))
        else:
            if versions["timm"] != "0.6.13" or not all(v["match"] for v in masking.values()):
                raise RuntimeError("Masked timm implementation does not match the pinned release")
            config = plan["config"]
            if config["torch_num_threads"] != int(os.environ["SLURM_CPUS_PER_TASK"]):
                raise ValueError("Config threads differ from allocation")
            if plan["mode"] == "reference":
                record["successful_probe_job_id"] = verify_probe(plan, config)
                record["successful_probe_launch_sha256"] = sha256(plan["probe_launch"])
            record["inputs_verified"] = verify_inputs(config)
            checkpoint = Path(config["segmentation"]["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            record["sam_checkpoint_sha256"] = sha256(checkpoint)
            weights = Path(config["resnet_checkpoint"])
            if not weights.is_file():
                raise FileNotFoundError(weights)
            if weights.name != "resnet50_a1_0-14fe96d1.pth":
                raise ValueError("Unexpected classifier checkpoint")
            expected_cache = Path(os.environ["TORCH_HOME"]) / "hub/checkpoints" / weights.name
            if weights.resolve() != expected_cache.resolve():
                raise ValueError("Classifier checkpoint differs from TORCH_HOME cached path")
            record["resnet_checkpoint_sha256"] = sha256(weights)
            for key, found in (("sam_checkpoint_sha256", record["sam_checkpoint_sha256"]),
                               ("resnet_checkpoint_sha256", record["resnet_checkpoint_sha256"])):
                if config.get(key) and config[key] != found:
                    raise ValueError("Checkpoint identity changed: " + key)
            resolved = launch / "config.json"
            write_json(resolved, config)
            record["config_sha256"] = sha256(resolved)
            write_json(path, record)
            if plan["mode"] == "resource-probe":
                sys.path.insert(0, str(release))
                helpers = {}
                for key in ("probe", "numerics"):
                    code = base64.b64decode(plan[key + "_base64"], validate=True)
                    if hashlib.sha256(code).hexdigest() != plan[key + "_sha256"]:
                        raise ValueError("Probe helper identity mismatch")
                    scope = {"__name__": "humcd_" + key}
                    exec(compile(code, "transmitted_" + key + ".py", "exec"), scope)
                    helpers[key] = scope
                probe_output = launch / "probe"
                probe_output.mkdir()
                record["probe_report"] = helpers["probe"]["run"](
                    config, release, probe_output, helpers["numerics"]["numerical_checks"])
                record["status"] = "PROBE_COMPLETED"
            else:
                subprocess.run([sys.executable, str(release / "run_smoke.py"),
                                "--config", str(resolved), "--run-id", job],
                               cwd=release, check=True)
                record["status"] = "EXECUTION_COMPLETED"
    except BaseException as exc:
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(path, record)


if __name__ == "__main__":
    main()
