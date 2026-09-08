#!/usr/bin/env python3
"""Collect existing HU-MCD results through an authenticated SSH master.

Remote work is limited to Slurm queries and named SFTP downloads.
All analysis, hashes and snapshot management happen locally.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import resource
import shutil
import subprocess
import sys
import time
import uuid

PROFILES = {
    "reference": {"name": "humcd-reference", "log": "reference", "output": "reference",
                  "files": ["summary.json", "metrics_report.md",
                            "training_prototypes/overview.png", "validation_prototypes/overview.png"]},
    "smoke": {"name": "humcd-p3-smoke", "log": "phase3-smoke", "output": "phase3_smoke",
              "files": ["summary.json", "metrics_report.md",
                        "training_prototypes/overview.png", "validation_prototypes/overview.png"]},
    "gpu-probe": {"name": "humcd-gpu-probe", "log": "gpu-probe", "output": "gpu_probe",
                  "files": ["gpu_probe.json"]},
}
SCIENTIFIC_FILES = ("discovery.npz", "discovery.json", "training.npz", "training_segments.json",
                    "training_checks.json", "validation.npz", "validation_segments.json", "validation_checks.json")
FIELDS = "JobID,JobName%100,User,State,Start,End,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode,NodeList"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
SSH_OPTIONS = ["-o", "ControlMaster=no", "-o", "BatchMode=yes",
               "-o", "ProxyCommand=/bin/false", "-o", "ClearAllForwardings=yes"]


class CollectorError(Exception):
    pass


def run(argv, input_text=None, max_file_bytes=None):
    def limit_file_size():
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))
    try:
        return subprocess.run(argv, input=input_text, text=True, capture_output=True,
                              timeout=90, check=False,
                              preexec_fn=limit_file_size if max_file_bytes else None)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CollectorError(str(exc)) from exc


class Transport:
    def __init__(self, host, max_file_bytes=16 * 1024**2):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", host):
            raise CollectorError("SSH host must be a simple configured alias or hostname")
        self.host = host
        self.max_file_bytes = max_file_bytes

    def check(self):
        result = run(["ssh", *SSH_OPTIONS, "-O", "check", self.host])
        if result.returncode:
            raise CollectorError("No usable authenticated SSH master from this process. "
                                 "Log in manually in the same WSL distribution, then retry. "
                                 + result.stderr.strip())

    def query(self, command):
        result = run(["ssh", *SSH_OPTIONS, self.host, command])
        if result.returncode:
            raise CollectorError("Slurm query failed: " + result.stderr.strip())
        return result.stdout

    def download(self, remote, local):
        def quote(value):
            return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = run(["sftp", *SSH_OPTIONS, "-b", "-", self.host],
                         "get " + quote(remote) + " " + quote(local) + "\n",
                         max_file_bytes=self.max_file_bytes)
        except CollectorError:
            local.unlink(missing_ok=True)
            raise
        if result.returncode:
            local.unlink(missing_ok=True)
            detail = (result.stderr + result.stdout).strip()
            if result.returncode != 255 and ("No such file" in detail or
                    f'File "{remote}" not found.' in result.stderr):
                return False
            raise CollectorError("SFTP failed for " + remote + ": " + detail)
        return True


def parse_accounting(raw, job_id, profile):
    rows = list(csv.DictReader(io.StringIO(raw), delimiter="|"))
    job = next((row for row in rows if row.get("JobID") == job_id), None)
    if job is None:
        raise CollectorError("No accounting record for job " + job_id)
    if job.get("JobName") != profile["name"]:
        raise CollectorError("Job name does not match selected profile: " + str(job.get("JobName")))
    return job, rows


def inspect_summary(summary, profile):
    if not isinstance(summary, dict) or summary.get("status") != "PASS":
        raise ValueError("Summary must be a JSON object with status PASS")
    if profile in ("smoke", "reference"):
        rate = summary["prototype_quality_proxy"]["validation"]["learned_concept_assignment_rate"]
        if rate == 0:
            return ["Zero validation assignment to learned concepts: execution PASS does not "
                    "establish useful generalisation or paper reproduction."]
    return []


def write_report(folder, manifest):
    job = manifest["job"]
    lines = ["# HU-MCD collected results", "", "Collection: " + manifest["collection_status"],
             "", "| Slurm field | Value |", "|---|---|"]
    for key in ("JobID", "JobName", "State", "Elapsed", "ReqMem", "AllocCPUS", "ExitCode", "NodeList"):
        lines.append(f"| {key} | {job.get(key, '')} |")
    lines += ["", "## Provenance", "", manifest["artifact_attribution"], "",
              "Hashes describe collected bytes. Run-layout artifacts are checked against producer hashes when available; files are not an atomic cross-file snapshot.",
              "", "## Observations", ""]
    lines += ["- " + msg for msg in manifest["warnings"] + manifest["errors"]]
    if manifest.get("progress"):
        progress = manifest["progress"]
        lines += ["", "## Run progress", "",
                  f"Status: {progress.get('status')}; last completed stage: {progress.get('last_completed_stage')}",
                  f"Updated (UTC): {progress.get('updated_at_utc')}"]
    lines += ["", "## Collected files", ""]
    for item in manifest["files"]:
        if item["status"] in ("downloaded", "reused"):
            lines.append(f"- [{item['local_path']}]({item['local_path']}) ({item['status']}, {item['bytes']} bytes)")
        else:
            lines.append("- Missing: " + item["remote_path"])
    (folder / "report.md").write_text("\n".join(lines) + "\n")


def collect(args, transport, cache=None):
    cache = {} if cache is None else cache
    profile = PROFILES[args.profile]
    transport.check()
    accounting = transport.query(
        f"sacct --jobs={args.job_id} --parsable2 --units=K --format={FIELDS}")
    job, rows = parse_accounting(accounting, args.job_id, profile)
    # Completed IDs may be purged from squeue; query the user's queue and filter locally.
    queue_raw = transport.query("squeue --me --noheader --format='%i|%T|%R'")
    queue = "".join(line + "\n" for line in queue_raw.splitlines()
                    if line.split("|", 1)[0].strip() == args.job_id)
    now = datetime.now(timezone.utc)
    name = now.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    parent = args.output_root.expanduser().resolve() / args.job_id
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / (".partial-" + name)
    folder.mkdir()
    (folder / "sacct.psv").write_text(accounting)
    (folder / "squeue.txt").write_text(queue)
    manifest = {
        "schema_version": 1, "collected_at_utc": now.isoformat(), "ssh_host": args.host,
        "job": job, "accounting_records": rows, "profile": args.profile,
        "artifact_attribution": "Logs are selected by job ID. Artifacts come from a shared "
            "output directory that may have been overwritten by another run; association "
            "with this job is unverified. Neither remote code commit nor original resolved "
            "configuration is available from this collector.",
        "files": [], "warnings": [], "errors": [], "transport_failed": False,
    }
    root = args.remote_root.rstrip("/")
    output = root + "/outputs/" + args.output_subdir
    selected = [(f"{root}/logs/{profile['log']}-{args.job_id}.{ext}",
                 f"logs/{profile['log']}-{args.job_id}.{ext}", True) for ext in ("out", "err")]
    if args.profile == "reference":
        selected.append((f"{root}/launches/{args.job_id}/launch_manifest.json",
                         "results/launch_manifest.json", True))
    if args.run_layout:
        selected += [(output + "/" + f, "results/" + f, True)
                     for f in ("run_manifest.json", "resolved_config.json", "artifacts.json")]
    selected += [(output + "/" + f, "results/" + f, True) for f in profile["files"]]
    if args.include_scientific:
        selected += [(output + "/scientific/" + f, "results/scientific/" + f, True) for f in SCIENTIFIC_FILES]
    selected.append((output + "/progress.json", "results/progress.json", args.run_layout))
    published = {}
    verified_run = False
    try:
        for remote, relative, required in selected:
            local = folder / relative
            entry = {"remote_path": remote, "local_path": relative, "required": required}
            expected = published.get(relative.removeprefix("results/"), {})
            previous = cache.get(remote)
            reused = False
            if verified_run and expected and previous and previous.is_file():
                old = previous.read_bytes()
                if len(old) == expected.get("bytes") and hashlib.sha256(old).hexdigest() == expected.get("sha256"):
                    local.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(previous, local)
                    reused = True
            if reused or transport.download(remote, local):
                data = local.read_bytes()
                entry.update(status="reused" if reused else "downloaded", bytes=len(data),
                             sha256=hashlib.sha256(data).hexdigest())
                if args.run_layout and (relative.removeprefix("results/") in profile["files"] or relative.startswith("results/scientific/")) and not expected:
                    manifest["errors"].append("No producer checksum yet: " + relative)
                if expected and (len(data) != expected.get("bytes") or entry["sha256"] != expected.get("sha256")):
                    manifest["errors"].append("Published checksum mismatch: " + relative)
                if relative.endswith(".json"):
                    try:
                        parsed = json.loads(data)
                        if relative == "results/launch_manifest.json":
                            if (str(parsed.get("slurm_job_id")) != args.job_id
                                    or parsed.get("actual_commit") != args.expected_commit
                                    or parsed.get("plan", {}).get("commit") != args.expected_commit):
                                raise ValueError("Launcher job/commit identity mismatch")
                            manifest["launcher_provenance"] = parsed
                        if relative == "results/run_manifest.json":
                            if str(parsed.get("slurm_job_id")) != args.job_id or parsed.get("run_id") != args.job_id:
                                raise ValueError("Run/job identity does not match requested Slurm job")
                            if args.expected_commit and (parsed.get("git_commit") != args.expected_commit
                                                         or parsed.get("git_dirty") is not False):
                                raise ValueError("Research commit mismatch or dirty prepared release")
                            verified_run = True
                            manifest["run_provenance"] = parsed
                            manifest["artifact_attribution"] = (
                                "Job-specific output directory and producer manifest match the requested job. "
                                "See run_provenance for commit, working-tree state and source hashes. "
                                "This is producer-declared provenance, not independent attestation.")
                        elif relative == "results/resolved_config.json":
                            if not isinstance(parsed, dict) or entry["sha256"] != manifest.get("run_provenance", {}).get("resolved_config_sha256"):
                                raise ValueError("Resolved config does not match the run manifest checksum")
                        elif relative == "results/artifacts.json":
                            if str(parsed.get("slurm_job_id")) != args.job_id or parsed.get("run_id") != args.job_id:
                                raise ValueError("Artifact index belongs to another run")
                            files = parsed["files"]
                            if not isinstance(files, dict) or any(not isinstance(v, dict) for v in files.values()):
                                raise ValueError("Malformed artifact index")
                            for item in files.values():
                                if not isinstance(item.get("bytes"), int) or item["bytes"] < 0 or not re.fullmatch(r"[a-f0-9]{64}", str(item.get("sha256"))):
                                    raise ValueError("Malformed artifact checksum/size")
                            published = files
                        elif relative == "results/progress.json" and isinstance(parsed, dict):
                            if args.run_layout and (str(parsed.get("slurm_job_id")) != args.job_id or parsed.get("run_id") != args.job_id):
                                raise ValueError("Progress belongs to another run")
                            manifest["progress"] = parsed
                        if relative.endswith(("summary.json", "gpu_probe.json")):
                            manifest["warnings"].extend(inspect_summary(parsed, args.profile))
                            if args.run_layout and (str(parsed.get("slurm_job_id")) != args.job_id or parsed.get("run_id") != args.job_id):
                                raise ValueError("Summary belongs to another run")
                    except (ValueError, KeyError, TypeError, AttributeError) as exc:
                        manifest["errors"].append(f"Invalid/incomplete {relative}: {exc}")
            else:
                entry["status"] = "missing"
                if required:
                    manifest["warnings"].append("Missing artifact: " + remote)
            manifest["files"].append(entry)
    except CollectorError as exc:
        manifest["errors"].append(str(exc))
        manifest["transport_failed"] = True
    missing = any(f["required"] and f["status"] == "missing" for f in manifest["files"])
    manifest["collection_status"] = "partial" if missing or manifest["errors"] else "complete"
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(folder, manifest)
    final = parent / name
    folder.rename(final)
    for entry in manifest["files"]:
        if entry["status"] in ("downloaded", "reused"):
            cache[entry["remote_path"]] = final / entry["local_path"]
    return final, manifest


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="Existing numeric Slurm job ID (no arrays)")
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--host", default="bunya")
    parser.add_argument("--expected-commit", help="Require this exact clean research commit")
    parser.add_argument("--remote-root", default="/scratch/user/uqcche38/hu-mcd")
    parser.add_argument("--output-subdir", help="Directory under remote outputs; default depends on profile")
    parser.add_argument("--run-layout", action="store_true", help="Collect outputs/runs/JOB_ID with required provenance")
    parser.add_argument("--max-file-mib", type=int, default=16, help="Hard per-file SFTP write limit (1-256 MiB)")
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "artifacts/bunya")
    parser.add_argument("--include-scientific", action="store_true", help="Collect fitted bases, features, labels and reconstruction evidence")
    parser.add_argument("--watch", action="store_true", help="Bounded foreground polling")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between polls, minimum 60")
    parser.add_argument("--max-polls", type=int, default=60)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[1-9][0-9]*", args.job_id):
        parser.error("job_id must be a positive integer")
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", args.remote_root) or ".." in PurePosixPath(args.remote_root).parts:
        parser.error("remote-root must be an absolute path without whitespace, wildcards or '..'")
    if args.expected_commit and not re.fullmatch(r"[a-f0-9]{40}", args.expected_commit):
        parser.error("expected-commit must be a full lowercase SHA")
    if args.include_scientific and not args.run_layout:
        parser.error("include-scientific requires run-layout")
    if args.expected_commit and not args.run_layout:
        parser.error("expected-commit requires run-layout")
    if args.profile == "reference" and (not args.run_layout or not args.expected_commit):
        parser.error("reference requires run-layout and expected-commit")
    if args.run_layout and (args.output_subdir or args.profile not in ("smoke", "reference")):
        parser.error("run-layout requires smoke/reference and no output-subdir override")
    if not 1 <= args.max_file_mib <= 256:
        parser.error("max-file-mib must be between 1 and 256")
    args.output_subdir = "runs/" + args.job_id if args.run_layout else args.output_subdir or PROFILES[args.profile]["output"]
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_./-]*", args.output_subdir) or ".." in PurePosixPath(args.output_subdir).parts:
        parser.error("output-subdir must be a relative path without wildcards or '..'")
    if any(ch in str(args.output_root) for ch in ("\n", "\r", "\x00")):
        parser.error("output-root must not contain control characters")
    if args.interval < 60 or args.max_polls < 1:
        parser.error("interval must be >=60 seconds; max-polls must be >=1")
    return args


def main(argv=None):
    args = arguments(argv)
    try:
        transport = Transport(args.host, args.max_file_mib * 1024**2)
        cache = {}
        for poll in range(args.max_polls if args.watch else 1):
            path, manifest = collect(args, transport, cache)
            state = manifest["job"]["State"].split()[0].rstrip("+")
            print(f"Job {args.job_id}: {state}; collection {manifest['collection_status']}\n"
                  f"{path / 'report.md'}", flush=True)
            if not args.watch or state in TERMINAL:
                return 0 if manifest["collection_status"] == "complete" else 2
            if manifest["transport_failed"]:
                return 2
            if poll + 1 < args.max_polls:
                time.sleep(args.interval)
        print("Polling limit reached; job may still be running. Run again to resume.", file=sys.stderr)
        return 2
    except (CollectorError, OSError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Collection stopped locally; remote jobs were not changed.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
