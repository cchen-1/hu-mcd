# HAM classifier → Derm7pt concept discovery / 新阶段准备

Recorded 2026-09-24 Australia/Brisbane. Stable decision ID: MED26-HAM-DERM-DISCOVERY-01.

## Latest approval / 最新批准 — 2026-09-24

User: “数据你下载一份到download放到一个单独的文件夹内然后scp到bunya scratch里的合理位置 因为之后要频繁读写的 另外101例全部拟合没问题 前置工作做到位之后就可以继续准备bunya上提交相关job了”.

R101 all101 cases are now explicitly approved for concept fitting, with no Derm7pt hold-out in this first run. The cohort question below is historical/resolved, not pending permission. Download to a dedicated Windows Downloads folder and transfer to scratch are authorized. No repeated permission request for these actions. Later transferability/faithfulness evaluation remains deferred; fitted Derm7pt examples are in-sample.

- Official Harvard Dataverse metadata request returnedHTTP403 before any archive transfer. Author-endorsed official ISIC2018Task3Training provides the HAM10000 image set; use its original images, ground-truth ZIP and lesion grouping CSV. No ISIC validation/test set requested. Official URLs and response headers retained. Exact archive equality to Harvard has not been established; source identities and pixels are checked, not assumed.
- Local destination: `C:\Users\uqcche38\Downloads\HAM10000_original_ISIC2018_20260924`.
- Scratch root: `/scratch/user/uqcche38/datasets/ham10000/isic2018-task3-20260924/`; raw archives in `raw/`, immutable per-job extracted inputs in `prepared/`, receipts in `audit/`.
- Slurm directory preparation28883377: COMPLETED0:0,4seconds,1CPU/1GiB/5minute cap; compute-generated READY receipt collected. No filesystem processing on login node.
- Authorized data audit uses2CPU/8GiB/20minutes/0GPU; transfer hash, ZIP CRC, IDs, labels and decoded identity comparison. No silent replacement or relabel. Job submission awaits complete local download and transfer.
- Five local synthetic preparation tests PASS; source plan and hashes will be bound before the audit. Downloader does not retry or overwrite partial failures.

Training/discovery direction is approved. Exact runnable training/discovery plans still require accepted raw bindings and explicit resource/configuration records, not a fresh blanket approval. Current preparation does not claim models trained or concepts fitted.

## User decision / 用户决定

> 好 同意此次方案 先按照心概念发现算法与发布代码基本保持一致 的方法来训练基于HAM的resnet50，再用derm7pt来做concept discovery，把pipeline跑通，我们后面再补齐迁移评价

This authorizes the direction: train a HAM-based ResNet50 and subsequently FIT concepts on Derm7pt with the released discovery algorithm largely unchanged. This supersedes the earlier proposed HAM-discovery → Derm7pt-frozen-basis-transfer direction for this NEW branch only. Old accepted results and authorizations remain historical evidence. Inverse-frequency CE was withdrawn before this approval and is NOT part of this direction. No new tuning, alternate SAM, automatic retry or resource expansion.

用户批准新方向：HAM训练分类器，Derm7pt拟合概念，先跑通流程。此分支不同于已有HAM发现→Derm7pt固定概念转移。拟合用过的Derm7pt图片不能再声称是该概念库的外部留出评价。旧400/70及R101结果全部保留，不覆盖；400个HAM发现病灶不是分类器全部训练集，也不自动成为新分支的发现集。

## Actual preparation / 本轮实际准备

- Existing authenticated SSH master verified: `Master running (pid=88416)`. Initial sandbox-local socket check returned Operation not permitted; permitted retry of the same check succeeded. No new login, remote shell computation or job submission.
- Latest handover includes an independent nine-class baseline batch (deployment28881848); do not resubmit or collect that batch without its completion notice. No model/job completion status inferred from this local preparation.
- Existing accepted source manifest contains10,015uniqueimageIDs, split8215/573/1227. This is evidence of source identity mapping availability, NOT acceptance of the still-unlocated raw HAM JPEG collection.
- Added `hpc/ham_input_preparation.py`: calls the existing ImageClass/ConceptDatasetClass directly. No changes to released scientific modules or old medical launchers. Short-side cap300 retains aspect ratio before SAM; classifier size from model_cfg. Unexpected image modes fail visibly rather than convert or substitute.
- Added two local synthetic tests in `tests/test_ham_input_preparation.py`; both PASS. Four geometries (600×450,450×600,180×240,224×224) confirm intended resize/tensor arithmetic and unchanged source bytes; grayscale rejection checked. No real medical inputs, SAM, classifier inference or weights used. This is software preparation, NOT training completion or full launcher acceptance.
- Warnings: inherited `np.bool8` deprecation; Matplotlib default cache directory unwritable, successful temporary fallback under /tmp. No observed tensor effect. Record via HAM26-PREP-001; use explicit writable MPLCONFIGDIR in a future launcher.

## Unresolved execution dependencies / 待补齐

1. Raw HAM JPEG + metadata location. Do not reconstruct original detail from the224NPZ, scan remote storage on login nodes, or download another archive by assumption. Any remote audit/preparation must use Slurm.
2. Derm7pt fitting scope: asked whether existing melanomaR101 (all101 for fit, no held-out claim) or a separately agreed discovery/hold-out split. R101 was previously approved for evaluation, not automatically an approved fitting sample list. No assumptions about all1011records, other disease classes or sample replacement.
3. Bind original-JPEG mapping and classifier training preprocessing before freezing executable config. Reuse plainCE, standard seven-outputResNet50 and prior basic recipe as preparation defaults; exact config/hash/commit/resources must be explicit at submission. No weightedCE or optimization sweep.
4. With only101discoveryimages, leave original min_cluster_size50 and all other filters unchanged. Fewer/zero retained concepts are possible; do not lower thresholds to make the run look successful. Report fitting outcome separately from engineering integrity.
5. Discovery fit budget depends on actual cohort/regions; not frozen. Prior HAM400/70discovery cap2h is historical, not an automatic new Derm7pt allocation. No job submitted this turn.

## Retained scientific intent / 保持的算法

SAMViT-H;32points/side;min_mask_region_area256;released region filtering;short-sidecap300;cropping_mode0;input masking−1/erosionthreshold.25;global_pool2048;normalizedSSC;released cluster-number heuristic;minimumcluster50;max_samplesNone;outlierquantile1.0;released subspace/assignment/scoring. Medical class identity and image bindings are explicit adaptations. New SAM geometry invalidates oldE224mask-cache identity; no silent reuse.

Follow-up evaluation remains deferred as requested. Derm7pt labels/signs are not inputs to SSC or tuning targets. Report discovery-training examples as in-sample; do not present concept fitting itself as evidence of external concept transferability or clinical validity.
