# Approved execution and dataset identity / 已批准执行与数据身份

2026-09-21. This supersedes the pending-approval wording in the historical
readiness document, without overwriting that proposal or its evidence.

## A — approved, preparation verified / 已批准、执行准备已验证

The user approved all five final-input-mask conditions and the stated caps:
identity, disk erosion r1/r2 and dilation r1/r2; original model, fixed bases,
precision and accepted discovery scoring (`norm_batch=False`) unchanged.
Ten independent jobs, each 1 L40S / 4 CPU / 16 GiB / 20 minutes, at most two
concurrent; total GPU allocation ceiling 3h20. Two scheduling lanes use
`afterany` only to limit concurrency: failure in one class does not invalidate
another class's scientific inputs. Code publication uses `afterok` plus actual
commit/clean-release/worker-hash checks inside each job.

Source hash checks are integrated into each class job's first stage, rather
than making A depend on a separate medical download/audit. Original 500 images,
5687 region identities, source manifests, feature and mask caches are bound by
SHA-256. Exact effective masks and all cached validation scores/assignments are
checked before inference. New `hpc/mask_sensitivity.py` decomposition was tested
locally against all ten accepted validation caches: PASS, no CNN prediction.

每类先做GPU identity检查；失败保留数组／日志并停止该类。6个因空区域而缩小的
batch使用原剩余行做同大小不变区域对照；不跨原batch补位，不放宽阈值。全部空区域、
零特征、原始行ID和历史benchmark零向量C0行为保留；新分析的无效分配为-1，不能把它
计为学习概念。新空／零区域仍留在原有效区域分母。每条件保存packed masks、特征、
1000类logits、得分、分配、间隔和重构误差。每类新输出上限1GiB、总上限10GiB。

Expected 5–10min/class is an estimate, not a measurement. Runtime cap stays
20min/class. No SAM, SSC, concept refit, sample change, retry or automatic
resource expansion. Collection, paired-image bootstrap and interpretation
wait for the user's completion notification after submission.

## B — approved inputs preparation; unresolved scientific choices

用户批准seed43病灶优先：400个不同病灶、每病灶随机1张。测试70张全部保留，按61个病灶
处理相关性，不声称患者独立。模型训练、最终图像身份和预算仍待确认，名单尚未冻结。

The user supplied the original Derm7pt ZIP on RDM and authorized Slurm-only
verification, copying, extraction and audit on scratch. Local ZIP:
159,112,080 bytes; SHA-256
`89a4749e7e43d1c2e73876aaeb867af5b2624e929ee3d1ae7202268588566b54`.
Expanded members total163,238,049 bytes. Bounded staging allocation:
2 CPU / 8 GiB / 20min / no GPU, scratch storage cap1GiB.
Scratch copy and extraction are job-specific. Original RDM ZIP is read-only.
Future input paths are taken from the actual audit result, never presumed.

Local original ZIP metadata contains1011 cases, official train/valid/test
413/203/395. Four melanoma-metastasis cases occur2/1/1 across those splits.
All original diagnostic subtypes and seven categorical signs are counted
separately; no melanoma grouping, binary sign contrast, default25-pixel crop,
image conversion or exclusions are implicitly approved. Every dermoscopic and
clinical image is inventoried; metadata row, case number, file/pixel hashes,
mode, dimensions and duplicates are saved. Cross-dataset independence is not
established by this audit. Patient IDs are not independently verified.

## Clarifying the existing corrected file / 同名corrected文件的分歧

已实际核对用户指出的Downloads文件（并非凭文件名推断）：
`C:\Users\uqcche38\Downloads\dermamnist_corrected_224.npz`。
其SHA-256为`8aec08cc552f11728f6641b9e8f0a078ae9afbde64fe15910c8ba9683eec2809`，
与此前已完整审计的OneDrive corrected归档逐字节相同；并非损坏或“未经corrected”。

| Identity / 身份 | Bytes | MD5 | Train/validation/test |
|---|---:|---|---|
| Existing author OneDrive corrected file |1131558300|947968c463c25c5d79b946d4311afffb|8208/575/1232|
| Plan's exact Zenodo12739457 release |1131258316|84920fb70c83b234c295b6f0d4ae2bc0|8215/573/1227|

This is a split/version distinction, not a blanket claim that a corrected file
is unusable. The accepted legacy integrity audit found7 author-confirmed
duplicate pairs crossing train/validation (2) or train/test (5). These pairs
are not exact-pixel duplicates and are not revealed by lesion-ID separation
alone. Current author metadata moves7 images to training. Replacing only the
CSV while retaining the old split arrays would break row identity.

An earlier separately reconstructed candidate uses the current split, but its
archive MD5 differs from Zenodo. Archive differences alone cannot tell whether
only compression differs or whether pixels/construction also differ. That
candidate is not approved and has not been claimed pixel-identical to Zenodo.

**Current action: no download, no medical model training.** Preserve all files.
Two later choices remain: (1) acquire/verify the exact published file, or
(2) explicitly adopt and audit a named derived dataset from existing pixels
using the current split, with provenance and unresolved pixel-identity limits.
The user has not selected either route. No automatic fallback or data deletion.

Evidence: `artifacts/bunya/next-phase-20260921/Downloads-DermaMNIST-C-identity.json`,
`/home/chen/datasets/dermamnist-c/onedrive-legacy-947968c4/manifests/dataset_audit.json`,
`docs/MEDICAL_PILOT_PREPARATION.md`, and the preserved published metadata snapshot.

## Delivery contract / 本批交付约定

Receipts and actual IDs are written under
`artifacts/bunya/next-phase-20260921/submissions/`. Submission is not acceptance.
Update the existing private bilingual report and English implementation notes;
preserve all earlier accepted results, reviews and protected outputs. After
one submission/accounting snapshot, do not wait or monitor. No result is filled
with zero while pending; user notification triggers collection and analysis.
