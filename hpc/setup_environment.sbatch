#!/usr/bin/env bash
#SBATCH --job-name=humcd-env
#SBATCH --account=a_ai_collab
#SBATCH --partition=general
#SBATCH --qos=debug
#SBATCH --constraint=epyc3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=logs/setup-environment-%j.out
#SBATCH --error=logs/setup-environment-%j.err

set -euo pipefail

readonly repo_root="/scratch/user/uqcche38/hu-mcd"
cd "$repo_root"
source hpc/lib.sh
require_compute_node
print_job_context

module purge
module load miniforge/26.1.0-0
if [[ -z "${ROOTMINIFORGE:-}" ]]; then
  echo "ERROR: ROOTMINIFORGE was not set by the Miniforge module." >&2
  exit 3
fi
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"

export PIP_CACHE_DIR="$repo_root/cache/pip"
mkdir -p "$PIP_CACHE_DIR"
if [[ -d "$HUMCD_ENV_PREFIX" ]]; then
  echo "Updating the dedicated HU-MCD environment."
  mamba env update \
    --prefix "$HUMCD_ENV_PREFIX" \
    --file hpc/environment-bunya.yml \
    --prune
else
  echo "Creating the dedicated HU-MCD environment."
  mamba env create \
    --prefix "$HUMCD_ENV_PREFIX" \
    --file hpc/environment-bunya.yml
fi

conda activate "$HUMCD_ENV_PREFIX"
bash scripts/install_masked_timm.sh "$repo_root"
python -c "import torch, torchvision, timm, segment_anything; print({'python_torch': torch.__version__, 'torchvision': torchvision.__version__, 'timm': timm.__version__, 'cuda_build': torch.version.cuda})"
echo "Environment setup status: PASS"
echo "Finished (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
