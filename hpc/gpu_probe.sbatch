#!/usr/bin/env bash
#SBATCH --job-name=humcd-gpu-probe
#SBATCH --account=a_ai_collab
#SBATCH --partition=gpu_cuda
#SBATCH --qos=debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/gpu-probe-%j.out
#SBATCH --error=logs/gpu-probe-%j.err

set -euo pipefail

readonly repo_root="/scratch/user/uqcche38/hu-mcd"
cd "$repo_root"
source hpc/lib.sh
require_compute_node
print_job_context
load_humcd_environment

export TORCH_HOME="$repo_root/models/torch"
export HUMCD_SAM_CHECKPOINT="$repo_root/models/sam_vit_b_01ec64.pth"
export HUMCD_PROBE_OUTPUT="$repo_root/outputs/gpu_probe"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python hpc/gpu_probe.py
echo "GPU probe status: PASS"
echo "Finished (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
