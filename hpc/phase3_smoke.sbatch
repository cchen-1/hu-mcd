#!/usr/bin/env bash
#SBATCH --job-name=humcd-p3-smoke
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/phase3-smoke-%j.out
#SBATCH --error=logs/phase3-smoke-%j.err

set -euo pipefail

readonly repo_root="/scratch/user/uqcche38/hu-mcd"
cd "$repo_root"
source hpc/lib.sh
require_compute_node
print_job_context
load_humcd_environment

export TORCH_HOME="$repo_root/models/torch"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

test -d data/phase3_smoke/golden_retriever
test -d data/phase3_smoke/val_imgs/golden_retriever_val
test -f models/sam_vit_b_01ec64.pth

/usr/bin/time -v python run_smoke.py \
  --config hpc/configs/phase3_smoke_bunya.json \
  --run-id "${SLURM_JOB_ID:?Submit this script through Slurm}"
echo "Phase 3 smoke status: PASS"
echo "Finished (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
