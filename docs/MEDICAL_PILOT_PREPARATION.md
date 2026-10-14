# Colour-medical pilot: preparation and decision record

Date: 2026-09-10. No classifier inference, SAM, concept discovery, training,
or evaluation experiment was submitted. The only new Slurm job was the dataset
audit 28227929: COMPLETED, 0:0, 15 seconds, bun050, peak batch RSS 1,381,180 KiB.
SSH reused the authenticated master. Remote processing took place on Slurm.

## Dataset: integrity passed, selected release not yet accepted

The user downloaded the author's OneDrive archive and accompanying MD5 file.
The OneDrive folder contains no accompanying CSV, as confirmed by the user.
Its MD5 is correct for that folder, but it is **not** the locked Zenodo archive.

| Source | Train | Validation | Test | Archive MD5 | Status |
|---|---:|---:|---:|---|---|
| Downloaded older OneDrive corrected split | 8208 | 575 | 1232 | 947968c463c25c5d79b946d4311afffb | Integrity passed on WSL and Bunya; scientific use blocked |
| Locked Zenodo 12739457 corrected split | 8215 | 573 | 1227 | 84920fb70c83b234c295b6f0d4ae2bc0 | Metadata verified; matching image archive unavailable |

Private copies of the older source:

- WSL: `/home/chen/datasets/dermamnist-c/onedrive-legacy-947968c4/`
- Bunya: `/scratch/user/uqcche38/datasets/dermamnist-c/onedrive-legacy-947968c4/`

The locked `zenodo-12739457` directories remain separate. Do not select the
OneDrive archive automatically as a fallback. Windows originals are unchanged.

All 10,015 older-source images were read successfully. Arrays are uint8,
224 × 224 × 3; labels match the **older** author CSV in all three splits.
There are no images with three identical channels. Two exact-pixel duplicate
groups occur within training (four image entries), none across splits.
Recorded lesion IDs do not cross splits. This does not establish independence:
the author's confirmed duplicate-pair list reveals **seven cross-split pairs**:
two training–validation and five training–test pairs. These are not the same
as exact-pixel duplicates and are missed by checking lesion IDs alone.

The matching legacy CSV was recovered from the author's pinned GitHub tree:
`DermaMNIST/DermaMNIST_Analysis/CSV_files/combined_metadata_corrected.csv`,
MD5 `504579018fc1d910c32f74f3de9ffbc0`. The current CSV is the separate
`combined_metadata_corrected-HAM10000_corrected.csv`. Both have the same image
ID set. Seven images move from validation/test into training in the current CSV.
No image was removed or silently replaced during the audits.

Mapping evidence consists of the published archive checksum, pinned author
CSV-order construction notebook and full row-label agreement. We have **not**
independently compared the original JPEG for every array row. Patient IDs are
unavailable; no patient-independent claim is supported. No exhaustive new
near-duplicate search was performed.

Source: [author duplicate confirmation](https://github.com/kakumarabhishek/Corrected-Skin-Image-Datasets/tree/ac655ac5a5b93264c57d69d6d520778d915bfc89/DermaMNIST/HAM10000_DuplicateConfirmation).

## Recovery attempt and remaining choice

`hpc/recover_dermamnist_release.py` reassembled a separate candidate using all
existing pixels, the author's current CSV order and its published split.
It did not modify the original file or introduce a new sampling rule.
The candidate MD5 is `ddd62abd44609bf6dacf047aca22b87b`, **not** the expected
Zenodo MD5. It is retained with status `CANDIDATE_NOT_APPROVED`.

The mismatch does not establish whether compressed representation, source
pixels, or another construction detail differs. Do not call it byte-identical
or pixel-identical to Zenodo without additional evidence.

Preferred next step: obtain the exact released file through the
[Zenodo file link](https://zenodo.org/records/12739457/files/dermamnist_corrected_224.npz?download=1),
or obtain an author-provided mirror with the expected MD5. WSL retry returned
HTTP 504 again. The OneDrive HTTP401 from automated access was worked around by
the user's browser download; it does not imply the file is inaccessible to users.

Alternative requiring an explicit protocol decision: validate a locally derived
dataset with the current author's split and separately document its pixel origin.
This would be a named derived dataset, not a silent substitution for Zenodo.
No such dataset has been approved for an experiment.

## ResNet weights: concrete candidate, not a performance claim

Downloaded:
`pstnmz/FailCatcher-models/dermamnist_resnet18_224_randaug1_fold_0.pt`

- Hugging Face revision: `247d8a476050db968a7e9ec65d9762d867fdbe53`.
- SHA256: `93c1c4f931c03cf1e4e5f4c736a6a12668380e3d760e62df22b71f879896ea72`;
  agrees with the public LFS hash; 44,801,562 bytes.
- Private local directory: `/home/chen/datasets/models/failcatcher/247d8a476050db968a7e9ec65d9762d867fdbe53/`.
- Plain ResNet18, seven-class linear head, 512-dimensional pooled features.
  Strict loading passed for torchvision and the installed HU-MCD masked timm
  implementation. Installed mask source matches this repository. All 122
  state tensors are finite. No prediction or numerical masking check was run.
- Public training code uses MedMNIST `size=224`, ImageNet initialization,
  augmentation, and normalization mean/std 0.5 in all RGB channels. This is a
  better resolution fit than the early official 28-to-224 resized benchmark.
- Source code checked against Git blob IDs at commit
  `8b40fb4f49a7fe90fb7b00132a466e1070e05b64`. Its train/validation pool contains
  original train + validation, followed by calibration splitting and five-fold
  fitting. Default checkpoint selection in the inspected function monitors
  validation loss; a training-file introductory comment mentions validation
  accuracy instead. Use executable behavior and saved run records, not that
  comment, for any future reconstruction of training.
- Conservative pool audit: 8,010 original train+val IDs have zero shared image
  IDs, recorded lesion IDs, or author-confirmed duplicate pairs with the 1,227
  corrected test images. All 573 corrected validation IDs belong to that pool;
  they are **not an independent fresh holdout** for this candidate.
- The checkpoint itself does not embed a training image manifest or a hash
  linking it to its training log. The matching public run name and source are
  evidence, not proof of the historical execution configuration.

The candidate fold was chosen as fold 0 before reading its metrics, not by
selecting the best test score. The matching author's run reports, on the
**original 2,005-image test split**, accuracy 60.25%, balanced accuracy 65.91%,
and AUC 0.9177. Balanced accuracy here is the mean of the seven class recalls.
Always predicting the largest class would give 66.88% ordinary accuracy on
that old test set, but only 14.29% balanced accuracy. Thus this model is not an
accuracy-leading classifier; its class discrimination and majority-class
behavior need separate assessment. These are author-reported numbers, not
measurements on our corrected split or proof that the downloaded file produces
them. No model choice will be made by trying checkpoints on the corrected test.

This is a 2026 preprint-associated release, not a verified top-conference or
top-journal model. Its practical advantages are available weights, training
code, run logs, RGB 224 input, a small ResNet backbone and a compatible linear
head. Newness and venue alone are not selection criteria.

Sources: [model files](https://huggingface.co/pstnmz/FailCatcher-models/tree/247d8a476050db968a7e9ec65d9762d867fdbe53),
[training source](https://github.com/pstnmz/FailCatcher/blob/8b40fb4f49a7fe90fb7b00132a466e1070e05b64/benchmarks/medMNIST/trainings/train_resnet18_medMNIST.py),
[matching run metrics](https://github.com/pstnmz/FailCatcher/blob/8b40fb4f49a7fe90fb7b00132a466e1070e05b64/benchmarks/medMNIST/runs/dermamnist/resnet18_224_20251031-231119_randaug1_numepochs100_bs128/metrics_fold_0_test.json).

Other candidates considered:

| Candidate | Advantage | Reason not to select immediately |
|---|---|---|
| Official MedMNIST ResNet50_224 | Closest backbone size to current HU-MCD; documented Scientific Data benchmark and weight release | Original training uses 28-pixel images enlarged to 224 with nearest-neighbor resizing; weights on Zenodo currently unavailable to our request |
| Correction authors' native-224 ResNet18/50 | Closest dataset protocol | Training logs/code found, but no verified downloadable checkpoint; native-224 command uses a custom 3×3-stem network, not the standard ResNet50 used here |
| Unexplained Hugging Face ResNet50 files | Files exist | Missing model cards, split identities and preprocessing; cannot use file name or claimed accuracy as sufficient provenance |

The official old benchmark reports R50_224 accuracy 73.1%, R18_224 75.4%;
these old-split results do not rank models on corrected native-224 data.
Source: [official experiments](https://github.com/MedMNIST/experiments),
[official results](https://medmnist.com/).

## Proposed simplest pilot — not submitted

Keep a seven-class classifier. For concept discovery, begin with **one existing
diagnostic label**, provisionally melanoma (`mel`). This avoids inventing a
new benign/malignant grouping and changing the model's task.

- Proposed discovery list: 400 images from the approved corrected training
  split, seed 43, frozen before model results are inspected; report distinct
  lesion counts. Do not silently inherit the ImageNet grayscale exclusion rule.
- Proposed held-out concept examples: all 70 corrected test melanoma images.
  Preserve all outcomes, including misclassified images and empty/zero-feature
  regions. Do not force a 50-image subset merely to match the ImageNet example.
- Keep SAM ViT-H and the existing scientific thresholds initially. Switching
  ResNet50 to ResNet18 is a declared model change for a medical pilot, not an
  identical ImageNet reproduction: feature dimension changes from 2048 to 512.
- An optional 10–20 examples per class visual inventory can help inspect input
  variety; it is not an adequate per-class concept-fitting sample. Current
  reference fitting rejects clusters with fewer than 50 regions, so tiny
  per-class runs could confuse sample scarcity with transfer failure.

Before concept discovery: approve the dataset identity and model choice; adapt
the loader to the seven labels, classifier-specific normalization and checkpoint;
check whole-image predictions and classwise recall on the final untouched test
as an evaluation, not a checkpoint search; verify masked/full-image numerical
consistency and reconstruction with this backbone, without another SAM probe.
Make the medical launcher independent of hard-coded ImageNet/ResNet50 paths.
Bind code, inputs, checkpoint, precision and outputs. No current ImageNet run
configuration has been edited.

First deliverables should compare initial cluster examples, fixed random
training and held-out members, background/skin/hair examples, zero features,
unassigned regions and classwise classifier errors. Expert descriptions remain
unreviewed until reviewed. Anatomical/clinical meaning is not proved by good
reconstruction, high allocation rate or a plausible AI-generated concept name.

## Evidence and tooling

`artifacts/medical-pilot-review/` contains source snapshots, archive/array audits,
split comparisons, seven duplicate crossings, Slurm output, cross-host checks,
checkpoint hashes, architecture audit, public run metrics and recovery evidence.
Full image manifests remain private, with a locally collected remote copy.
`hpc/audit_dermamnist_legacy.py` reproduces the completed integrity audit.
`hpc/recover_dermamnist_release.py` records the unsuccessful exact-release
recovery attempt. Earlier failures are preserved. No changes were committed or
pushed for this task. Existing unrelated work is preserved.
