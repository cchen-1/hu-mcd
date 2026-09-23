# HAM classifier → Derm7pt concept discovery / 新阶段准备

Recorded 2026-09-24 Australia/Brisbane. Stable decision ID: MED26-HAM-DERM-DISCOVERY-01.

## 2026-09-24 当前验收与正式训练 / Current acceptance and classifier execution

**28883445 ACCEPTED_INPUTS_MODEL_NOT_RUN**: COMPLETED0:0,158s,MaxRSS2307524K. Local verification covers110indexed artifacts (all except1.5GBNPZ),101RGBlosslesspre-SAMimages and unchanged fixedR101identities,10,015classifiermanifest rows and24exacttensorchecks. Large classifier cache remains on scratch and is SHA256-bound before model loading. Evidence: `artifacts/bunya/ham-derm-discovery-20260924/inputs-acceptance.json`.

SAM geometry:98images450×300,one each452×300/433×300/446×300. None are224square. The classifier-only cache has10,015pixel differences from the former direct224cache because the now-approved path first applies released short-side300loading; this is an expected interpolation difference, not an identity error. Source JPEGs, corrected split identities and labels are unchanged.

Next executable classifier config: `configs/medical/ham-classifier.approved.json`; plain CE,50epochs/batch64/trainingseed43;8215/573/1227 source split;6450updates/410750training image visits; strict validation macroOVRAUC selection then one frozen test evaluation and same-batch8adapter checks. FP32/noAMP,cuDNN TF32true/matmulTF32false retained. Resource cap1L40S/4CPU/16GiB/1h/5GiB;worker ceiling3420s. Prior782s is a reference measurement, not a promise of runtime. No automatic extension/retry. This job trains only the HAM classifier; Derm7pt fitting waits for its checkpoint acceptance, with101fit/0held-out cases. No discovery sampling seed is consumed by classifier training.

The existing trainer is reused without changes. The new protocol has a separately gated1-hour cap; historical classifier configs retain4-hourcaps. New wrapper persists actual config/version,input hashes,progress,warnings,failure trace and artifact hashes.13local software tests PASS; no SAMprobe or model test run was added. The authenticated SSH master remains valid and the queue was empty before submission preparation.

Historical sections below retain their original status timestamps; they do not override this current acceptance.

## 2026-09-24 当前输入准备作业 / Current input preparation

**28883445 SUBMITTED**,2CPU/8GiB/20min/4GiBoutput/0GPU. Source commit **1131bd71f43c2dc202edc938bc6f3c2674c2f097** pushed. Complete source archive, actual worker bytes and plan SHA256 verified inside Slurm before execution. Data audit28883403 locallyACCEPTED; fixedR101all101approved. No expired afterok dependency, retries or model submissions.

The input plan enforces `sam_input_policy={short_side_cap:300,preserve_aspect_ratio:true,extra_square_resize:false,use_classifier_cache:false}`. Derm7ptSAMinputs use the released loader only; classifier-only224cache has an explicit forbidden-as-SAM role record. ExpectedDermsizes98×450×300,1×452×300,1×433×300,1×446×300. All原图保留，禁止SAM前额外方形缩放。源码中的224只用于分类器缓存，不是SAM输入。

Wait for user completion notice before collecting this batch. Required receipts: status/progress, actualplan, environment, warnings, artifacts index, classifier manifest,24no-modeltensorchecks, R101fitmanifest and101losslessinputs. LargeclassifierNPZ remains a pinned scratch cache; do not pretend it was locally collected when only its remote hash was collected. Then bind accepted hashes to the pending model launchers. No SAMprobe, predictions, training or concepts were executed by this CPU preparation.

Local evidence: `artifacts/bunya/ham-derm-discovery-20260924/{inputs-plan.json,inputs-release.json,inputs.submission.json,input-preparation-tests.log,data-acceptance.json}`. Six synthetic tests PASS. Legacy np.bool8 warning persists without observed tensor effect; writableMPLCONFIGDIR resolves prior default-cache fallback for the new job.

## 2026-09-24 geometry clarification and accepted original data / 几何冻结与原图验收

User explicitly confirmed: **不在SAM之前额外统一成224方图**; preserve released short-side300/aspect-ratio loading and released input/discovery flow. This resolves the temporary pause on new submissions. There is no new SAM geometry choice pending.

- Slurm28883403COMPLETED0:0/86seconds,MaxRSS89488K; local receipt accepts10,015raw600×450RGBimages, all source IDs/labels/lesionIDs and8215/573/1227split members, and10,015/10,015direct-bicubic224pixel matches. No original overwritten. Proof: `artifacts/bunya/ham-derm-discovery-20260924/data-acceptance.json`.
- Next CPU preparation materializes two explicitly distinct roles: (1) classifier-only224cache derived through released short-side300 then classifier resizing, never fed toSAM; (2) Derm7ptR101lossless pre-SAM inputs from released ImageClass only. R101expected sizes:98×450×300,1×452×300,1×433×300,1×446×300. No224pre-SAMwarp, crop, padding, alternateSAM, new model or fitting.
- Six local synthetic tests pass, including non-square pre-SAM geometry/lossless pixels and classifier-cache tensor equality with the released dataset. Future remote check first8rows per source split checks input arithmetic only, not predictions or anotherSAMprobe.
- CPU preparation cap2CPU/8GiB/20min/4GiBoutput/0GPU. After submission, await user completion notice to collect the new input-manifest hashes; then finalize bound classifier and discovery launchers. Failure is not automatically retried. Model-stage drafts are not submitted experiments.

## 2026-09-24 当前提交状态 / Current submitted batch

- Official ISIC HAM10000 images, labels and lesion groupings downloaded successfully to `C:\Users\uqcche38\Downloads\HAM10000_original_ISIC2018_20260924`; three source files total2,772,262,421bytes. URLs/headers/SHA256 receipts retained. Original image ZIP SHA256 `4151b6de5e31eb21586732d2dc0df047737501d5f5333cb52b071ad704364cc0`.
- Local ZIP-directory/CSV checks:10,015unique image IDs exactly match accepted corrected identities;0diagnosis differences;0lesion-ID differences. This is not a full decode/pixel audit. Multipart S3ETag is NOT treated as whole-file MD5.
- SCP completed successfully to `/scratch/user/uqcche38/datasets/ham10000/isic2018-task3-20260924/raw/`. No source overwritten or removed.
- Data audit **28883403 SUBMITTED**,2CPU/8GiB/20min/0GPU. Code **7547f3decad201530eeb13e1fb8b8d0d5b510779** pushed to fork reproduction. Worker actual bytes match committed source; worker+plan SHA256 checked within Slurm before Python execution. No stale schedulerdependency on completed directory job; successful directory receipt retained.
- Do not wait/poll for batch completion. User completion notice triggers collection, audit acceptance and subsequent model-stage preparation. No classifier training, SAM or concept fitting submitted. Model-stage draft has dependencies and proposed ceilings, not completed executable launchers: `artifacts/bunya/ham-derm-discovery-20260924/model-stage-draft.json`.
- Evidence: `artifacts/bunya/ham-derm-discovery-20260924/{download_manifest.json,local-header-check.json,local-lesion-id-differences.json,transfer.json,audit-plan.json,audit.submission.json,committed-file-check.json}`.

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
