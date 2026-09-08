#!/usr/bin/env bash
# This helper only queries Slurm and jobstats metadata. It does not launch the
# HU-MCD workload and is intended to be run on a Bunya login node.

set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash hpc/check_job.sh JOB_ID" >&2
  exit 2
fi

readonly job_id="$1"

echo "=== Slurm accounting ==="
sacct \
  --jobs "$job_id" \
  --units=G \
  --format=JobID,JobName,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode

echo
echo "=== Requested resources ==="
if control_record="$(scontrol show job "$job_id" 2>&1)"; then
  printf '%s\n' "$control_record" | grep -E \
    'JobId=|JobState=|RunTime=|TimeLimit=|ReqTRES=|AllocTRES=|NodeList=' \
    || true
else
  echo "Active controller record is no longer available for job $job_id."
  echo "This is normal for older completed jobs; sacct remains authoritative."
fi

echo
echo "=== jobstats ==="
module purge
if module load jobstats/2024.08; then
  if ! jobstats "$job_id"; then
    echo "jobstats data is unavailable; use the sacct metrics above."
  fi
else
  echo "jobstats/2024.08 could not be loaded; use the sacct metrics above."
fi
