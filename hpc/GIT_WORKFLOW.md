# Git-based Bunya code preparation

Implemented and live-tested on 8 September 2026 (Brisbane).

## Usage

Edit and test in WSL, then deliberately commit and push the intended changes to
origin/reproduction. Log in to Bunya yourself using the usual SSH/MFA workflow.
From the repository root run:

```bash
bash hpc/submit_git_prepare.sh "$(git rev-parse HEAD)"
```

This submits one CPU, 1 GiB, at most five minutes under a_ai_collab/general/debug.
The local helper sends the preparation script over the existing SSH connection;
experiment source is fetched from GitHub inside the compute job. No fresh SSH
connection or MFA automation is attempted. A returned job ID means submitted,
not successful: inspect sacct and download the job logs with SFTP.
If a connection fails during submission, inspect squeue before retrying to avoid
a duplicate job. Uncommitted changes are not deployed. The requested commit
must be reachable from the fetched reproduction branch.

Prepared code is at:
`/scratch/user/uqcche38/hu-mcd-git/releases/FULL_SHA/code`
with a sibling `provenance.txt`. Preparation uses a temporary directory, a lock,
and an atomic rename. Existing releases are never replaced; a repeat submission
for the same commit fails with an explicit message. Treat prepared releases as
read-only by convention; filesystem permissions do not enforce immutability.
Dependencies, datasets, model weights, submodules and LFS downloads are not
prepared by this step. The current implementation fetches branch history anew
for each release; a shared Git cache can be added if that becomes costly.

## Verified test

- Job: 28159694; node: bun148; COMPLETED; exit 0:0; elapsed four seconds.
- Repository: https://github.com/cchen-1/hu-mcd.git
- Commit: f22adf20f616f5556d07d8f6d099ffea10468421
- Worker checked exact HEAD, clean working tree and branch ancestry.
- Logs: /scratch/user/uqcche38/humcd-git-prepare-28159694.out and .err.
- Local syntax checks, malformed SHA rejection and execution outside a compute
  allocation rejection passed. No GPU experiment was launched.

## Remaining integration

The existing phase3_smoke.sbatch still uses the old project path. Do not assume
it runs this release. Next, adapt an experiment launcher to accept a prepared
release and use a unique run directory for config, logs, outputs and provenance.
Reuse existing model/data/environment paths explicitly. Record commit, resolved
config and Slurm job ID for every run. The collector now supports --run-layout for outputs/runs/JOB_ID.
Its smoke profile still expects the smoke job name and log naming; adapt those
together if the next launcher uses a different layout. Git prepares code; the
collector downloads results. Neither replaces the other.

Existing local modifications and the old Bunya project were preserved. The tested release above predates the subsequent tracking/collector changes.
See ../docs/MAIN_CHAT_HANDOVER.md and the accompanying commit for the packaged
local tools. Publishing those tools does not deploy a new Bunya release.
Prepared releases live on scratch and should be considered disposable, not backups.
