#!/usr/bin/env python3
"""Render or submit a pinned-release job. Default: render only; no SSH or submission."""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess

ROOT = "/scratch/user/uqcche38/hu-mcd"
RELEASES = "/scratch/user/uqcche38/hu-mcd-git/releases"
OPTIONS = ["-o", "ControlMaster=no", "-o", "BatchMode=yes",
           "-o", "ProxyCommand=/bin/false", "-o", "ClearAllForwardings=yes"]


def absolute_path(value):
    if not isinstance(value, str) or not re.fullmatch(r"/[A-Za-z0-9_./-]+", value) or ".." in PurePosixPath(value).parts:
        raise ValueError("An explicit absolute path without shell metacharacters is required")
    return value


def validate_reference(config, cpus):
    expected = {"class_name": "golden_retriever", "model_name": "resnet50",
                "layer_name": "global_pool", "device": "cuda:0",
                "train_images": 400, "validation_images": 50, "seed": 43,
                "max_shortest_side": 300, "torch_num_threads": cpus}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Reference config requires {key}={value!r}")
    segmentation = config["segmentation"]
    for key, value in {"algorithm": "sam", "model_type": "vit_h",
                       "points_per_side": 32, "min_mask_region_area": 256}.items():
        if segmentation.get(key) != value:
            raise ValueError("Reference segmentation mismatch: " + key)
    if config["clustering"] != {"algorithm": "sparse_subspace_clustering",
                               "n_clusters": None, "min_cluster_size": 50,
                               "outlier_percentile": 1.0, "subspace_dimensionality": None}:
        raise ValueError("Reference clustering settings differ from released behavior")
    if type(config.get("batch_size")) is not int or config["batch_size"] < 1:
        raise ValueError("An explicit positive batch_size is required")
    for key in ("source_dir", "dataset_manifest", "resnet_checkpoint"):
        absolute_path(config[key])
    absolute_path(segmentation["checkpoint"])
    if config.get("output_dir") != ROOT + "/outputs/reference" or config.get("cache_root") != ROOT + "/cache/reference":
        raise ValueError("Reference output/cache roots do not match the collector layout")
    return config


def build_plan(args):
    if not re.fullmatch(r"[a-f0-9]{40}", args.commit):
        raise ValueError("A full lowercase commit SHA is required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.partition or "") or not re.fullmatch(r"[A-Za-z0-9_-]+", args.qos or ""):
        raise ValueError("Explicit partition and qos names are required")
    if not args.cpus or args.cpus < 1 or not re.fullmatch(r"[1-9][0-9]*[MG]", args.memory or ""):
        raise ValueError("Explicit positive CPU and memory requests are required")
    if not re.fullmatch(r"(?:[0-9]+-)?[0-9]{2}:[0-5][0-9]:[0-5][0-9]", args.time or ""):
        raise ValueError("Explicit Slurm time is required")
    if args.mode in ("inspect", "prepare-data", "inspect-selection", "audit-training", "publish-training", "compatibility", "prepare-model") and args.gpu:
        raise ValueError("Inspection must not allocate a GPU")
    if args.mode in ("reference", "resource-probe") and not re.fullmatch(r"[A-Za-z0-9_]+:1", args.gpu or ""):
        raise ValueError("Reference requires an explicit single-GPU type")
    worker = Path(__file__).with_name("release_worker.py").read_bytes()
    plan = {"schema_version": 1, "commit": args.commit, "mode": args.mode,
            "release": RELEASES + "/" + args.commit + "/code", "runtime_root": ROOT,
            "worker_sha256": hashlib.sha256(worker).hexdigest(),
            "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "resources": {"cpus": args.cpus, "memory": args.memory, "time": args.time,
                          "partition": args.partition, "qos": args.qos, "gpu": args.gpu}}
    if args.mode in ("compatibility", "prepare-model"):
        helper = Path(__file__).with_name("reference_preparation.py").read_bytes()
        plan["preparation_worker_base64"] = base64.b64encode(helper).decode()
        plan["preparation_worker_sha256"] = hashlib.sha256(helper).hexdigest()
        if args.mode == "compatibility":
            plan["dataset_manifest"] = absolute_path(args.dataset_manifest)
            if not re.fullmatch(r"[a-f0-9]{64}", args.dataset_manifest_sha256 or ""):
                raise ValueError("Frozen dataset manifest SHA256 required")
            plan["dataset_manifest_sha256"] = args.dataset_manifest_sha256
            plan["prepared_root"] = absolute_path(args.prepared_root)
    elif args.mode in ("inspect", "prepare-data", "inspect-selection", "audit-training", "publish-training"):
        if args.mode != "publish-training":
            plan["dataset_root"] = absolute_path(args.dataset_root)
        if args.mode in ("audit-training", "publish-training"):
            if args.seed != 43:
                raise ValueError("Use the user-approved seed 43")
            plan["seed"] = args.seed
            artifact_path = args.frozen_selection if args.mode == "audit-training" else args.selection_audit
            artifact_sha = args.frozen_selection_sha256 if args.mode == "audit-training" else args.selection_audit_sha256
            plan["selection_path"] = absolute_path(artifact_path)
            if not re.fullmatch(r"[a-f0-9]{64}", artifact_sha or ""):
                raise ValueError("An explicit immutable selection SHA256 is required")
            plan["selection_sha256"] = artifact_sha
            if args.mode == "publish-training":
                plan["prepared_root"] = absolute_path(args.prepared_root)
            data_worker = Path(__file__).with_name("audit_training_candidates.py").read_bytes()
            plan["data_worker_sha256"] = hashlib.sha256(data_worker).hexdigest()
            plan["data_worker_base64"] = base64.b64encode(data_worker).decode()
        if args.mode in ("prepare-data", "inspect-selection"):
            if args.seed != 43:
                raise ValueError("Use the user-approved seed 43")
            plan["seed"] = args.seed
            plan["prepared_root"] = absolute_path(args.prepared_root)
            data_worker = Path(__file__).with_name("prepare_reference_data.py").read_bytes()
            plan["data_worker_sha256"] = hashlib.sha256(data_worker).hexdigest()
            plan["data_worker_base64"] = base64.b64encode(data_worker).decode()
    else:
        if not args.config:
            raise ValueError("A fully resolved --config is required")
        plan["config"] = validate_reference(json.loads(args.config.read_text()), args.cpus)
    if args.mode == "reference":
        if not plan["config"].get("save_scientific_records"):
            raise ValueError("Reference run must save scientific records")
        if args.after_probe:
            if not re.fullmatch(r"[1-9][0-9]*", args.after_probe) or args.probe_launch or args.probe_launch_sha256:
                raise ValueError("Use one numeric after-probe job ID without a completed-probe override")
            if not re.fullmatch(r"[a-f0-9]{40}", args.probe_commit or ""):
                raise ValueError("Deferred probe requires its exact research commit")
            plan["after_probe"] = args.after_probe
            plan["probe_commit"] = args.probe_commit
            plan["probe_launch"] = ROOT + "/launches/" + args.after_probe + "/launch_manifest.json"
            plan["expected_probe_helpers"] = {
                "probe_sha256": hashlib.sha256(Path(__file__).with_name("reference_probe.py").read_bytes()).hexdigest(),
                "numerics_sha256": hashlib.sha256((Path(__file__).parent.parent / "utils/scientific_records.py").read_bytes()).hexdigest()}
        else:
            plan["probe_launch"] = absolute_path(args.probe_launch)
            if not re.fullmatch(r"[a-f0-9]{64}", args.probe_launch_sha256 or ""):
                raise ValueError("Successful probe launch SHA256 is required before a reference run")
            plan["probe_launch_sha256"] = args.probe_launch_sha256
    if args.mode == "resource-probe":
        for key, path in (("probe", Path(__file__).with_name("reference_probe.py")),
                          ("numerics", Path(__file__).parent.parent / "utils/scientific_records.py")):
            code = path.read_bytes()
            plan[key + "_base64"] = base64.b64encode(code).decode()
            plan[key + "_sha256"] = hashlib.sha256(code).hexdigest()
    return plan, worker


def render(plan, worker):
    resources = plan["resources"]
    names = {"inspect": ("humcd-release-inspect", "release-inspect"),
             "prepare-data": ("humcd-data-prepare", "data-prepare"),
             "inspect-selection": ("humcd-selection-audit", "selection-audit"),
             "audit-training": ("humcd-training-audit", "training-audit"),
             "publish-training": ("humcd-training-publish", "training-publish"),
             "compatibility": ("humcd-input-compat", "input-compat"),
             "prepare-model": ("humcd-model-prepare", "model-prepare"),
             "resource-probe": ("humcd-reference-probe", "reference-probe"),
             "reference": ("humcd-reference", "reference")}
    job_name, log = names[plan["mode"]]
    encoded = base64.b64encode(json.dumps(plan, sort_keys=True).encode()).decode()
    q = shlex.quote
    lines = ["#!/usr/bin/env bash",
             "#SBATCH --job-name=" + job_name, "#SBATCH --account=a_ai_collab",
             "#SBATCH --partition=" + resources["partition"], "#SBATCH --qos=" + resources["qos"],
             "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
             "#SBATCH --cpus-per-task=" + str(resources["cpus"]),
             "#SBATCH --mem=" + resources["memory"], "#SBATCH --time=" + resources["time"],
             "#SBATCH --output=" + ROOT + "/logs/" + log + "-%j.out",
             "#SBATCH --error=" + ROOT + "/logs/" + log + "-%j.err"]
    if plan.get("after_probe"):
        lines.append("#SBATCH --dependency=afterok:" + plan["after_probe"])
        lines.append("#SBATCH --kill-on-invalid-dep=yes")
    if resources["gpu"]:
        lines.append("#SBATCH --gres=gpu:" + resources["gpu"])
    lines.extend([
        "set -euo pipefail",
        '[[ -n "$SLURM_JOB_ID" ]] || { echo "Slurm allocation required" >&2; exit 2; }',
        '[[ "$(hostname -s)" =~ ^bun[0-9]{3}$ ]] || { echo "Compute node required" >&2; exit 2; }',
        "release=" + q(plan["release"]),
        '[[ "$(git -C "$release" rev-parse HEAD)" == ' + q(plan["commit"]) + ' ]]',
        '[[ -z "$(git -C "$release" status --porcelain)" ]]',
        'source "$release/hpc/lib.sh"', "require_compute_node", "load_humcd_environment",
        "export PYTHONDONTWRITEBYTECODE=1", "export MPLBACKEND=Agg",
        "export TORCH_HOME=" + q(ROOT + "/models/torch"),
        "export HF_HUB_OFFLINE=1", "export HF_HUB_DISABLE_TELEMETRY=1",
        'export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"',
        'export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"',
        'export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"',
        'export LOKY_MAX_CPU_COUNT="$SLURM_CPUS_PER_TASK"',
        "python -c " + q(worker.decode()) + " " + q(encoded)])
    return "\n".join(lines) + "\n"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--commit", required=True)
    p.add_argument("--mode", choices=("inspect", "prepare-data", "inspect-selection", "audit-training", "publish-training", "compatibility", "prepare-model", "resource-probe", "reference"), required=True)
    p.add_argument("--dataset-root")
    p.add_argument("--prepared-root")
    p.add_argument("--seed", type=int)
    p.add_argument("--frozen-selection")
    p.add_argument("--frozen-selection-sha256")
    p.add_argument("--selection-audit")
    p.add_argument("--selection-audit-sha256")
    p.add_argument("--dataset-manifest")
    p.add_argument("--dataset-manifest-sha256")
    p.add_argument("--after-probe", help="Queue one reference run after this existing probe succeeds")
    p.add_argument("--probe-commit", help="Exact research commit used by the existing deferred probe")
    p.add_argument("--probe-launch")
    p.add_argument("--probe-launch-sha256")
    p.add_argument("--config", type=Path)
    p.add_argument("--cpus", type=int, required=True)
    p.add_argument("--memory", required=True)
    p.add_argument("--time", required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--qos", required=True)
    p.add_argument("--gpu")
    p.add_argument("--save-script", type=Path)
    p.add_argument("--submit", action="store_true", help="Actually submit; default only prints the plan")
    return p.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    plan, worker = build_plan(args)
    script = render(plan, worker)
    if args.save_script:
        with args.save_script.open("x") as f:
            f.write(script)
    print(json.dumps({k:v for k,v in plan.items() if not k.endswith("_base64")}, indent=2), flush=True)
    print("Submission script SHA256:", hashlib.sha256(script.encode()).hexdigest(), flush=True)
    if args.submit:
        subprocess.run(["ssh", *OPTIONS, "-O", "check", "bunya"], check=True, timeout=15)
        # This is the only login-node command. All work is inside the batch script.
        result = subprocess.run(["ssh", *OPTIONS, "bunya",
                                 "sbatch --parsable --chdir=/scratch/user/uqcche38"],
                                input=script, text=True, capture_output=True, timeout=30)
        print(result.stdout, end="")
        if result.returncode:
            raise RuntimeError("Submission failed or outcome uncertain; inspect squeue before retrying: " + result.stderr)
        if not re.fullmatch(r"[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?\s*", result.stdout):
            raise RuntimeError("Unrecognised response; inspect squeue before retrying")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
