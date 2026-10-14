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

## Actual first-stage result / 实际前置检查结果

Execution commit `743d29886aed3c4878c5ffb07aef83de2b5a0d7e`; deployment28732646
COMPLETED0:0/3s. Data preparation28732647 FAILED1:0/4s on bun147.
`/QRISdata` is readable, but `/QRISdata/Q9903` itself and its dataset paths
return Permission denied for the job identity. The inspected scratch tree has
a verified matching companion CSV but no exact target NPZ; RDM file presence
remains unknown. No HTTP download was attempted, nothing was archived and no
ready dataset-path configuration was published. This is an access-gate failure,
not a Zenodo transfer or file-integrity failure. The prior Derm7pt failure was
on the same node; a node-specific/identity-propagation explanation is not ruled
out. User asked to check HPC access and uqcche38 read/write membership. No retry.

完整校验后的失败证据已收集到
`artifacts/bunya/next-phase-20260921/dermamnist-exact/collections/28732647/20260921T045921.036585Z/`。
权限原因尚未确认；下载与归档保持暂停，不影响已有A作业，不扩大后续模型实验授权。

## Access recovery evidence / 已验证节点与首次实际获取

The user confirmed collection HPC access and uqcche38 read/write membership.
Read-only Slurm diagnostic28733281 completed0:0/4s on **bun148**, explicitly
excluding prior nodebun147. Actual directory listings of Q9903, datasets and
Derm7pt raw succeeded; advisory read/write/traverse checks were true.
`id -G uqcche38` included549903 (Q9903RW), while the process supplementary group
list omitted it. Node and explicit trailing-slash directory access were changed
together: a unique stale-group or automount cause is not proven. No ACL changed.

First acquisition job **28733630** is now bound to bun148,
execution **83972e99763ef6ecaebba3cbad6c0bf14e62e63a**,1CPU/8GiB/**01:59:00**/0GPU, reserving1minute
within the earlier2h budget for completed short checks. Deployment job
28733627. No earlier completed job attempted an HTTP download.
Existing-file identity and real archive-parent write access are still mandatory
first-stage checks; a new failure stops without retry. Original Derm7pt failure
remains preserved and is not resubmitted under this data-only request.

提交不等于下载或归档已成功。当前已解决的是bun148上的实际读访问；写入、完整下载、
图像验收和归档需以本次作业最终证据为准，不预先标记通过。

## Final outcome of the authorized single attempt / 单次尝试最终结果

Job28733630 FAILED1:0/35s on bun148. Its complete existing-file inventory found
no exact target NPZ in the inspected scratch/Q9903 dataset trees. Companion
CSV matched and was reused. Creation of the RDM archive-parent directory and
exclusive lock succeeded: this establishes actual write access on bun148.
The sole HTTP request returned curl22/HTTP504 after31seconds, received0bytes.
No automatic retry, alternate mirror, derived-data fallback or new experiment.

**Not completed:** target image download, array audit, sealed RDM archive and
ready dataset-path publication. The parent directory/lock are not an archived
dataset. The0byte download record does not establish any corrupted full file;
it records a failed transfer. The upstream504 cause is not uniquely identified.

Exact collection:
`artifacts/bunya/next-phase-20260921/dermamnist-exact/verified-node-collections/28733630/20260921T050725.304151Z/`.
The user is asked whether to download this exact official file in their local
browser or pause. If provided later, validate1131258316bytes and MD5
`84920fb70c83b234c295b6f0d4ae2bc0` before any new Slurm audit/archive step.
Existing OneDrive files, approved experiment inputs and failure records remain
unchanged. This data authorization still does not permit model experiments.

## 2026-09-21 final data acceptance update / 最新数据验收

Earlier HTTP504 and access failures above are historical and preserved. Newly available exact Downloads NPZ was reused offline (allow_download=false) in Slurm28738133, code822860db132f35a540b079b9bef87d7c156ce8d8: COMPLETED0:0/37s. MD5 matches84920fb70c83b234c295b6f0d4ae2bc0; all10015images/labels and official8215/573/1227splits pass. Archived/verified under Q9903 independent version and accepted locally using23artifact hashes plus sealed15-file archive receipt. No new download; OneDrive source preserved. Two reader ResourceWarnings are logged as MED26-009, local context-manager fix only, no rerun.

Accepted data paths and receipt: `artifacts/bunya/next-phase-20260921/medical-decisions/{accepted-dataset-paths.json,data-preparation/acceptance.json}`. Data acceptance does not authorize medical experiments. External overlap is not yet checked; concrete choices: `docs/DERM7PT_PROTOCOL_DECISIONS_20260921.zh-en.md`.
