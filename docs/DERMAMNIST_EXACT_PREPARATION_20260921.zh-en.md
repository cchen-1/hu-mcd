# Exact DermaMNIST-C preparation / 精确发布版数据准备

2026-09-21. User authorization is **data preparation only**. It supersedes the
earlier download hold, but authorizes no medical classifier, sampling-list
publication, concept discovery or evaluation experiment.

Target: Zenodo record12739457, `dermamnist_corrected_224.npz`,
MD5 `84920fb70c83b234c295b6f0d4ae2bc0`,1131258316bytes.
Companion `DermaMNIST-C.csv`: MD5 `1c54bc9f483e97a9d7dbbde076a45900`,747941bytes.
Reuse already acquired metadata after exact verification. Both the original
metadata and its source record snapshot remain part of the archive.

## Actual pre-submission state / 提交前事实

- Authenticated SSH master is active. No matching new DermaMNIST preparation
  job was found in current queue or local submission receipts.
- Local dataset trees and Downloads contain the OneDrive corrected identity
  and a separate derived candidate, but no complete file matching the exact
  published size/MD5. Existing files remain unchanged. Windows filename now
  observed as `one_drive_dermamnist_corrected_224.npz`.
- Prior Derm7pt job28731216 FAILED1:0 after4seconds. Its collected, verified
  log reports PermissionError at the supplied Q9903 ZIP path before copy or
  extraction. This does not establish whether the denial is at the collection,
  a descendant directory, or the ZIP itself. No retry was submitted.

## Frozen data-only workflow / 固定执行流程

One Slurm preparation job:1CPU,8GiB,2h,0GPU; maximum download time110min minus
preparation elapsed time; no retries, resume or alternative mirror after
failure. Planned new storage remains below3GiB across the two compressed raw
copies plus metadata/audit records. CPU code deployment is a separate5min
infrastructure step; each worker checks its exact commit and clean checkout.

1. Inside Slurm, record RDM path access and process identity. Inventory named
   scratch data/preparation roots and Q9903 datasets, capped at depth8 and
   20000files. Hash complete-size candidates; mismatched/partial files are
   retained. An inaccessible directory is **unknown**, not proof of absence.
2. Acquire a per-release RDM lock and recheck candidates to prevent simultaneous
   duplicate downloads. Verify archive-parent write access before downloading.
   If the inventory or permissions are unresolved, stop with evidence.
3. Reuse exact files if found. Otherwise fetch each missing official file once
   into a new job-specific scratch directory. Preserve partial failures and
   HTTP/curl evidence. Never overwrite the OneDrive or derived identities.
4. Reuse `hpc/prepare_dermamnist.py`: published hashes, six NPZ arrays,
   uint8/224×224×3 images, labels/order against author CSV construction,
   split counts8215/573/1227, lesion overlap and exact pixel duplicates.
   Separately check all author-confirmed duplicate pairs against current IDs.
   Record any within-split duplicates without removing them.
5. Only after all gates PASS, copy raw NPZ/CSV, author provenance, source
   record, audit scripts, image identity manifest and checksums to a new RDM
   staging directory, verify copied hashes, then publish a sealed version.
   Existing completed archives are verified/reused, never overwritten.
6. Generate `dataset_paths.json` with scratch/RDM paths, hashes, audit binding
   and `downstream_experiments_authorized=false`. Local acceptance still
   awaits result collection; no experiment launcher is automatically enabled.

RDM destination / 归档目录（planned until sealed receipt exists）：

```
/QRISdata/Q9903/datasets/dermamnist-c/zenodo-12739457-md5-84920fb70c83b234c295b6f0d4ae2bc0/
```

Scratch working copies / 实验工作副本：

```
/scratch/user/uqcche38/datasets/dermamnist-c/zenodo-12739457/work/<SLURM_JOB_ID>/
```

Collector root / 本地收集入口：
`artifacts/bunya/next-phase-20260921/dermamnist-exact/`.
Raw images are not automatically recopied locally; small audit/provenance and
path records are collected through the existing hashed artifact collector.
RDM is the recovery source; repeated experimental reads use scratch.

## Access evidence / 访问问题的解释边界

[UQ Bunya UserData Guide](https://github.com/UQ-RCC/hpc-docs/blob/main/guides/Bunya-UserData-Guide.md)
documents HPC-enabled collections under `/QRISdata/QNNNN` and notes that users
with separate student/staff credentials may need the HPC identity added as a
collaborator. This is a possible cause of a permissions failure, not a confirmed
diagnosis for this collection. No ACL, membership, login or 2FA change is made
by the worker; no permission bypass or message to support is attempted.

Submission is not successful download or archival. Preserve every failed
attempt; do not retry automatically. After dispatch, stop waiting and await
user completion notification unless an already-completed failure can be
reported from the single submission-state snapshot.
