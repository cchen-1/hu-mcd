#!/usr/bin/env bash
# Submit a CPU-only preparation job through an already authenticated SSH master.
set -euo pipefail
sha="${1:-}"
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo "Usage: bash $0 FULL_COMMIT_SHA" >&2; exit 2; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ssh -O check bunya
# Explicit SHA prevents silently deploying uncommitted work or a newer branch tip.
# If submission returns a connection error, inspect squeue before retrying.
ssh -o ControlMaster=no -o BatchMode=yes -o ProxyCommand=/bin/false bunya \
  "sbatch --parsable --chdir=/scratch/user/uqcche38 --export=ALL,HUMCD_PREPARE_SHA=$sha" < "$script_dir/prepare_git.sbatch"
