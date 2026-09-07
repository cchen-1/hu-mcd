#!/usr/bin/env bash
#SBATCH --job-name=humcd-stage
#SBATCH --account=a_ai_collab
#SBATCH --partition=general
#SBATCH --qos=debug
#SBATCH --constraint=epyc3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/user/uqcche38/stage-humcd-%j.out
#SBATCH --error=/scratch/user/uqcche38/stage-humcd-%j.err

set -euo pipefail

short_host="$(hostname -s)"
if [[ ! "$short_host" =~ ^bun[0-9]{3}$ ]]; then
  echo "ERROR: $short_host is not a compute node; refusing to extract." >&2
  exit 2
fi

readonly scratch_root="/scratch/user/uqcche38"
readonly archive="$scratch_root/hu-mcd-phase3-smoke.tar.gz"
readonly checksum="$archive.sha256"

echo "Compute node check: PASS ($short_host)"
echo "Started (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "$scratch_root"
sha256sum --check "$checksum"
tar --extract --gzip --file "$archive"
mkdir -p \
  "$scratch_root/hu-mcd/logs" \
  "$scratch_root/hu-mcd/cache" \
  "$scratch_root/hu-mcd/outputs"

test -f "$scratch_root/hu-mcd/run_smoke.py"
test -f "$scratch_root/hu-mcd/models/sam_vit_b_01ec64.pth"
test -f "$scratch_root/hu-mcd/models/torch/hub/checkpoints/resnet50_a1_0-14fe96d1.pth"
echo "Staging status: PASS"
echo "Finished (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
