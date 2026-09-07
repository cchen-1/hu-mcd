#!/usr/bin/env bash

set -euo pipefail

readonly HUMCD_ACCOUNT="a_ai_collab"
readonly HUMCD_REMOTE_ROOT="/scratch/user/uqcche38/hu-mcd"
readonly HUMCD_ENV_PREFIX="/home/uqcche38/.conda/envs/humcd-bunya"

require_compute_node() {
  local short_host
  short_host="$(hostname -s)"
  if [[ ! "$short_host" =~ ^bun[0-9]{3}$ ]]; then
    echo "ERROR: $short_host is not a Bunya compute node; refusing to run." >&2
    exit 2
  fi
  echo "Compute node check: PASS ($short_host)"
}

load_humcd_environment() {
  module purge
  module load miniforge/26.1.0-0
  if [[ -z "${ROOTMINIFORGE:-}" ]]; then
    echo "ERROR: ROOTMINIFORGE was not set by the Miniforge module." >&2
    exit 3
  fi
  source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
  conda activate "$HUMCD_ENV_PREFIX"
}

print_job_context() {
  echo "Job ID: ${SLURM_JOB_ID:-not-set}"
  echo "Host: $(hostname -f)"
  echo "Started (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
