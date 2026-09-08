# Local Bunya results collector

Run from Ubuntu WSL after your normal Bunya SSH login and MFA in the same
Linux account. The collector reuses that authenticated SSH master and refuses
a fresh connection. It never submits, cancels or changes a job.

```bash
cd /home/chen/projects/hu-mcd
python3 hpc/collect_results.py 28005092
python3 hpc/collect_results.py 28002639 --profile gpu-probe
```

For a future smoke job using the updated `phase3_smoke.sbatch`:

```bash
python3 hpc/collect_results.py JOB_ID --run-layout --watch --interval 60 --max-polls 60
```

Replace JOB_ID with the actual numeric ID. The updated runner and sbatch file
must first be deployed together with `utils/run_tracking.py`. Existing remote
jobs and their outputs were not changed by this local implementation. The
separate Git preparation workflow in `GIT_WORKFLOW.md` is not yet wired into
the hardcoded smoke-job repository path; preparing a Git release alone does
not make the smoke script execute it.

## What is saved

Each poll creates a timestamped snapshot under `artifacts/bunya/JOB_ID/` with
`report.md`, `manifest.json`, raw Slurm accounting, job-specific logs and named
small result files. Existing snapshots remain intact. Use `--output-root` to
choose another local destination. Requires Linux Python 3.9+ and OpenSSH;
the collector has no extra Python dependencies.

Legacy jobs use shared output directories (`phase3_smoke` or `gpu_probe`). Their
logs identify the job, but their shared artifacts cannot be reliably attributed
to that job. The report explicitly preserves that limitation. Use
`--output-subdir` for another legacy directory under remote `outputs/`.

With `--run-layout`, files come from `outputs/runs/JOB_ID`. The collector requires
run identity, the resolved configuration, a producer artifact index and progress.
It checks the config hash, result hashes and matching job/run IDs. Producer
records are evidence from the job, not independent attestation. Source hashes
and Git dirty state show when a commit alone is insufficient to reproduce code.
Files are fetched individually, not as an atomic cross-file snapshot.

## Bounded live collection

- Poll interval is at least 60 seconds; the default is at most 60 polls.
- Incomplete or invalid live JSON produces a partial snapshot and is retried on
  the next poll while the job runs. Terminal states stop polling; rerun manually
  if final artifacts appear later.
- SSH/query failures stop collection. SFTP failures also stop rather than
  attempting fresh authentication. User login and MFA stay manual.
- Each SFTP child has a hard 16 MiB file-write limit. `--max-file-mib` permits
  1–256 MiB. An oversized file causes a partial collection and stops polling;
  the error does not mean the remote job failed. It is not fetched piecemeal.
- During one watch invocation, finalized result files are reused from the prior
  local snapshot only when their size and SHA-256 match the producer index.
  Reused files are copied into the new snapshot; mutating one snapshot cannot
  mutate another. Logs and live metadata are fetched again.
- Ctrl-C, PC sleep or an expired SSH master do not cancel a remote job.

The runner writes atomic JSON at startup, completed stages, handled exceptions
and finalization. Progress reports the last completed stage; it is not a timed
heartbeat during long stages. Hard kills, node failure or failures before the
tracker starts may leave stale or absent progress; Slurm state and logs remain
necessary. Training/validation overview hashes are published once each sheet is
finished, allowing reuse while later stages run.

A Slurm run requires `--run-id "$SLURM_JOB_ID"`, supplied by the updated sbatch
file. Outputs and cache go into separate `runs/JOB_ID` directories. Reusing an
existing run directory fails instead of overwriting it. Local runs can also
supply a unique ID. Provenance records Git HEAD/dirty state, selected source
hashes, resolved config, and hashes of the training/validation files actually
loaded. The existing summary retains the SAM checkpoint hash. This is not a full
software-environment or model-weight archive.

## Remote scope and storage

Collector remote operations are only `sacct`, `squeue` and named SFTP `get`s.
It does not run Python, remote checksums, directory scans, installations or
experiments on the login node. Producer hashing/progress runs inside the smoke
job's compute allocation. SSH and SFTP apply `ControlMaster=no`, `BatchMode=yes`,
`ProxyCommand=/bin/false` and `ClearAllForwardings=yes`; the existing ControlPath
comes from your WSL SSH config.

Only named JSON, Markdown, overview PNGs and logs are fetched. Datasets,
checkpoints, full prototype galleries and caches are excluded. Local snapshots
accumulate; there is no automatic retention deletion or durable HPC archive.
Back up selected scientific outputs separately from scratch. Downloaded dataset
derivatives remain excluded from Git under `artifacts/bunya/`.

Exit codes: 0 = complete collection and basic artifact validation; 1 =
connection/query/input failure; 2 = partial collection or poll limit; 130 =
interrupted. Exit 0 does not establish scientific success or a successful Slurm
state. The zero-validation-assignment warning is retained.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s hpc/tests -p 'test_*.py' -v
bash -n hpc/phase3_smoke.sbatch
```

Tests cover incomplete-to-complete live polling, connection loss, actual child
file limits, checksum-based reuse with tampered local snapshots, mismatched
run/job IDs, immutable run paths, configuration/input hashes and failure
progress. The updated collector also retrieved existing completed Bunya job
28005092. No new GPU job was submitted; the new producer has local tests but
still needs an end-to-end test during the next scheduled smoke run.
