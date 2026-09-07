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
scontrol show job "$job_id" | grep -E \
  'JobId=|JobState=|RunTime=|TimeLimit=|ReqTRES=|AllocTRES=|NodeList='

echo
echo "=== jobstats ==="
module purge
module load jobstats/2024.08
jobstats "$job_id"
