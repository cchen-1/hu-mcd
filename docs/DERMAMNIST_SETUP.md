# DermaMNIST-C storage setup

Latest update (2026-09-10): **The older OneDrive archive passed complete WSL/Slurm integrity checks, but the locked Zenodo image release remains unavailable.**
See [the preparation and decision record](MEDICAL_PILOT_PREPARATION.md).
Slurm audit 28227929 completed in 15s. The OneDrive 8208/575/1232 split
contains seven author-confirmed cross-split duplicate pairs; it is stored
separately and is not approved for experiments. A reconstructed candidate
did not match the released MD5 and was not promoted. No model experiment ran.

The following sections preserve the **initial setup record**, before the
OneDrive download; its statements about image checks refer to that time.

Initial status: **PARTIAL — IMAGE DOWNLOAD BLOCKED BY ZENODO HTTP 504**.
The corrected metadata and source provenance are installed and verified on WSL
and Bunya. The 224-pixel image archive has NOT been downloaded. This is not a
ready dataset or a medical-transfer experiment.

## Locations

- WSL: `/home/chen/datasets/dermamnist-c/zenodo-12739457/`
- Bunya: `/scratch/user/uqcche38/datasets/dermamnist-c/zenodo-12739457/`
- Local job evidence: `artifacts/dermamnist-setup/`

Both locations contain `raw/`, `provenance/`, `manifests/`, and `prepared/`.
`prepared/` is intentionally empty: no conversion or sample selection occurred.
The existing ImageNet inputs, experiments, outputs and unrelated edits were not changed.
No RDM collection is configured for this dataset.

## Source identity

- Corrected release: https://zenodo.org/records/12739457 (v1.1).
- Image filename: `dermamnist_corrected_224.npz`.
- Expected image MD5: `84920fb70c83b234c295b6f0d4ae2bc0`.
- Metadata filename: `DermaMNIST-C.csv`.
- Verified metadata MD5: `1c54bc9f483e97a9d7dbbde076a45900`.
- The metadata was obtained from the authors' GitHub repository; its bytes match
  the published Zenodo checksum. No guessed or replacement split was used.
- Author repository: https://github.com/kakumarabhishek/Corrected-Skin-Image-Datasets
- Pinned author commit: `ac655ac5a5b93264c57d69d6d520778d915bfc89`.
- `provenance/CreateNPZfromCSV.ipynb` documents CSV-order appending to each split,
  label mapping, uint8 RGB storage, and bicubic resizing. `author-source.json`
  records its hash. This is mapping evidence, not an independent original-JPEG
  pixel comparison.

## Verified scope

| Split | Metadata image IDs | Distinct lesion IDs |
|---|---:|---:|
| train | 8215 | 5758 |
| val | 573 | 563 |
| test | 1227 | 1149 |

Image IDs are unique. No lesion ID crosses splits. WSL/Bunya metadata hashes,
script hashes, class counts and generated metadata indexes agree exactly.
`metadata_index.csv` contains **expected** NPZ row positions based on the authors'
construction code. These positions have not yet been checked against the archive.
There are no patient IDs in this metadata: patient independence is unverified.
Exact/near pixel duplicates and NPZ labels, shape, dtype and RGB channels remain
unverified. Counts above are metadata counts, not processed-image counts.

## Slurm evidence

- 28227015: COMPLETED / 0:0, 2s — directories and environment, bun048.
- 28227067: FAILED / 1:0, 64s — both Zenodo record URLs returned HTTP 504;
  no image archive was created. Metadata checksum passed before the download.
- 28227191: COMPLETED / 0:0, 2s — metadata-only audit, bun048.

All remote inspection and file processing ran through Slurm. SSH and SCP used
only the existing authenticated master; new-login fallback was disabled.
Local attempts at the main URL, the earlier same-checksum record, IPv4 HEAD
requests and the direct file API encountered timeouts/504. The author's
Hugging Face dataset page contained no image archive, so it was not a mirror.
No further retry job or automation is scheduled.

## Resume after the service recovers

From local WSL:

```bash
bash /home/chen/projects/hu-mcd/hpc/download_dermamnist.sh \
  /home/chen/datasets/dermamnist-c/zenodo-12739457
/home/chen/miniforge3/envs/humcd-upstream/bin/python \
  /home/chen/projects/hu-mcd/hpc/prepare_dermamnist.py \
  /home/chen/datasets/dermamnist-c/zenodo-12739457
```

The downloader preserves partial downloads, logs failures and publishes the final
filename only after the release checksum passes. The full auditor checks all
array shapes/types and row labels, records per-image pixel hashes and exact
pixel duplicates, and rejects cross-split overlap. It preserves every image and
the supplied splits. No classifier or SAM is loaded.

After a successful local download/audit, transfer the archive and provenance to
Bunya; run the same full audit in a new CPU Slurm job, collect its report and
compare hashes. Do not run the download or audit scripts on a Bunya login node.
The metadata-only sbatch in `artifacts/dermamnist-setup/` is retained as executed
evidence; do not mistake it for a full image audit.

Scientific choices still open: target class, medical classifier and its training
history, concept-discovery image count, and evaluation protocol. No 400/50 subset
has been assumed. Some corrected validation classes have fewer than 50 images.

Validation of the new tooling: shell syntax; successful synthetic archive audit;
rejection of a deliberately mismatched row label; metadata-only readiness state;
real metadata audits on both WSL and Slurm. Source scripts are local changes and
were not committed or pushed during this setup.
