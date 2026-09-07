# HU-MCD reproduction workflow

## Recreate the local CPU environment

```bash
conda env create -f environment-cpu.yml
conda activate humcd-upstream
bash scripts/install_masked_timm.sh
```

The masking installer keeps one backup of timm 0.6.13's original `resnet.py` and
then installs the repository's masking-aware ResNet implementation.

## Phase 1: local engineering smoke test

This phase validates the complete data path on CPU. It is not a paper-result run.

Dataset: Imagenette2-160 from fast.ai, a 94 MB, ten-class ImageNet subset with
train/validation splits. The prepared smoke subset uses symbolic links, so images
are not duplicated.

Downloaded inputs used for this run:

- Imagenette2-160: `https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz`
  (SHA-256 `64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5`)
- SAM ViT-B: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`
  (SHA-256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`)

Imagenette contains images derived from ImageNet. Use it for local research and
engineering checks subject to the applicable ImageNet terms; do not redistribute
the extracted images with this repository.

Local deviations from the paper configuration:

- SAM ViT-B instead of ViT-H
- 8 prompt points per side instead of 32
- 3 training images and 2 validation images for one class
- 2 SSC clusters with a minimum cluster size of 2
- a fixed one-dimensional PCA basis per retained concept
- CPU execution

These settings test loading, SAM masks, masked ResNet activations, sparse subspace
clustering, PCA concept bases, completeness, validation assignment, and prototype
rendering. They must not be reported as reproduction metrics.

Run from the repository root:

```bash
conda activate humcd-upstream
python run_smoke.py --config configs/smoke_local.json
```

Expected artifacts:

- `/home/chen/results/hu-mcd-smoke/output/summary.json`
- `/home/chen/results/hu-mcd-smoke/output/training_prototypes/overview.png`
- `/home/chen/results/hu-mcd-smoke/output/validation_prototypes/overview.png`
- reusable segmentation, activation, and SSC caches under
  `/home/chen/results/hu-mcd-smoke/cache/`

## Dataset preparation

The selected Imagenette class names map directly to the class keys in
`imagenet1k_class_info.json`. To rebuild the small HU-MCD layout:

```bash
python scripts/prepare_imagenette_smoke.py \
  --imagenette-root /home/chen/datasets/imagenette2-160 \
  --output /home/chen/datasets/hu-mcd-smoke \
  --class-name garbage_truck \
  --train-count 3 \
  --val-count 2
```

The script writes `dataset_manifest.json` with the exact source images and uses
absolute symbolic links into the extracted Imagenette directory.

## Phase 2: local Imagewoof fidelity gate

Imagewoof2-320 contains the paper target class `golden_retriever` and preserves
more source detail than the 160 px engineering dataset.

Downloaded archive:

- `https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz`
- SHA-256: `7db6120fdb9ae079e26346f89e7b00d7f184f8137791609b97fd0405d3f92305`

Prepare the 20/10 symbolic-link subset:

```bash
python scripts/prepare_image_subset.py \
  --dataset-name Imagewoof2-320 \
  --dataset-root /home/chen/datasets/imagewoof2-320 \
  --output /home/chen/datasets/hu-mcd-fidelity-imagewoof320 \
  --class-name golden_retriever \
  --synset n02099601 \
  --train-count 20 \
  --val-count 10
```

Run the fidelity configuration:

```bash
python run_smoke.py --config configs/fidelity_imagewoof320.json
```

The runner writes an English `summary.json`, `metrics_report.md`, training and
validation prototype sheets, stage timing, RSS, completeness, concept counts and
dimensions, and objective prototype assignment/coverage proxies. Manual semantic
prototype quality still requires visual inspection.
