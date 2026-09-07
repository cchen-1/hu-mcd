#!/usr/bin/env bash
# Build the Bunya smoke archive on the local WSL machine. This script refuses
# to run on hosts whose names look like Bunya login or compute nodes.

set -euo pipefail

short_host="$(hostname -s)"
if [[ "$short_host" =~ ^bunya[0-9]+$ || "$short_host" =~ ^bun[0-9]{3}$ ]]; then
  echo "ERROR: package_local.sh must run on local WSL, not Bunya." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${HUMCD_SMOKE_DATA_DIR:-/home/chen/datasets/hu-mcd-fidelity-imagewoof320}"
sam_checkpoint="${HUMCD_SAM_VIT_B:-/home/chen/models/sam/sam_vit_b_01ec64.pth}"
resnet_checkpoint="${HUMCD_RESNET50_WEIGHTS:-/home/chen/.cache/torch/hub/checkpoints/resnet50_a1_0-14fe96d1.pth}"
dist_dir="$repo_root/dist"
archive="$dist_dir/hu-mcd-phase3-smoke.tar.gz"

for required in \
  "$repo_root/run_smoke.py" \
  "$data_root/golden_retriever" \
  "$data_root/val_imgs/golden_retriever_val" \
  "$sam_checkpoint" \
  "$resnet_checkpoint"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required input is missing: $required" >&2
    exit 3
  fi
done

for tool in rsync tar sha256sum; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: required local tool is unavailable: $tool" >&2
    exit 4
  fi
done

stage_root="$(mktemp -d -t humcd-bunya-package.XXXXXX)"
cleanup() {
  rm -rf -- "$stage_root"
}
trap cleanup EXIT

mkdir -p \
  "$stage_root/hu-mcd/data/phase3_smoke" \
  "$stage_root/hu-mcd/models/torch/hub/checkpoints" \
  "$stage_root/hu-mcd/logs" \
  "$stage_root/hu-mcd/cache" \
  "$stage_root/hu-mcd/outputs" \
  "$dist_dir"

rsync -a \
  --exclude=.git/ \
  --exclude=__pycache__/ \
  --exclude=dist/ \
  --exclude='*.pyc' \
  "$repo_root/" "$stage_root/hu-mcd/"

# Dereference the local subset symlinks so the archive contains actual images.
rsync -aL "$data_root/" "$stage_root/hu-mcd/data/phase3_smoke/"
cp "$sam_checkpoint" "$stage_root/hu-mcd/models/sam_vit_b_01ec64.pth"
cp "$resnet_checkpoint" \
  "$stage_root/hu-mcd/models/torch/hub/checkpoints/resnet50_a1_0-14fe96d1.pth"

tar --create --gzip --file "$archive" -C "$stage_root" hu-mcd
(
  cd "$dist_dir"
  sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256"
)

echo "Package status: PASS"
echo "Archive: $archive"
du -h "$archive"
cat "$archive.sha256"
