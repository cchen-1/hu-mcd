# Medical ResNet50: concrete proposal M / 医学分类器具体方案 M

> **2026-09-22 APPROVED:** 用户批准此完整M50/batch64/训练seed43方案及1L40S/4CPU/16GiB/4h/5GiB、共用部署1CPU/1GiB/5min；仅源域分类器，不含医学概念发现或外部推理。历史提案正文保留；当前授权及替代关系见[集中决策表](NEXT_PHASE_CURRENT_DECISIONS_20260922.zh-en.md)。工程验收与研究适用性分开，后续报告类别级/melanoma表现与训练曲线，不因低分自动重训。

2026-09-22 Australia/Brisbane. **Proposal only: not approved, not trained, not a medical result.** This specifies a single source-classifier run; medical HU-MCD discovery and Derm7pt inference are separate, unapproved execution stages.

**2026-09-22澄清：训练seed43与50epochs均未获批准；选样seed43的批准不延伸到训练。50并非已验证的充分或最佳训练长度。稳定决策ID：MED26-002/M-SEED、M-EPOCHS，详见当前集中决定表的依据部分。**

## Roles and evidence / 数据用途与依据

| Role / 用途 | Fixed accepted source / 数量 | Use / 使用边界 |
|---|---|---|
| Classifier training / 七类分类器训练 | DermaMNIST-C train **8215images**, all7classes | Update model parameters on all training images, including multiple images per lesion; no400-image restriction, no rebalancing or deduplication |
| Model selection / 模型选择 | val **573images**; melanoma22images/20lesions | Choose checkpoint once by validation macroOVR AUROC; no test or Derm7pt selection |
| Final classifier test / 最终源域分类测试 | test **1227images**, all7classes | Only after checkpoint identity is frozen; includes the fixed70melanoma images/61lesions |
| Later HU-MCD discovery / 后续概念发现 | melanoma train1021images/533lesions → **400distinct lesions,1image each,seed43** | User approved this direction. It is a subset for explaining the frozen classifier, not its whole training set. Exact list remains unfrozen pending model/budget readiness. No new selection this turn |
| Later source concept evaluation / 源域概念评价 | all70test melanoma images/61lesions | Retain all; lesion-aware uncertainty, not70independent lesions or patients |
| Later external concept–sign analysis / 外部征象关联 | Derm7pt R101/P100/E248 pending | No classifier training, checkpoint selection or HU-MCD fitting on these images/sign labels |

已验收资料：数据作业28738133、验收 `next-phase-20260921/medical-decisions/data-preparation/acceptance.json`；NPZ MD5 `84920fb70c83b234c295b6f0d4ae2bc0`。数据已经准备完毕，不再下载或重做完整性审计；新作业只检查其实际输入/权重绑定及新路径正确性。跨数据集H检查与医学训练可独立准备；未解决重叠会限制或阻塞相应外部评价。

## Why this candidate / 选择依据与差异

Bounded evidence review includes the correction authors' pinned scripts at `ac655ac5a5b93264c57d69d6d520778d915bfc89`, official MedMNIST benchmark history, the previously inspected FailCatcherR18 candidate, and current author/model landing pages. We have **not found a verified existing checkpoint** satisfying standardResNet50 + corrected-native224 source splits + complete training/preprocessing identity. This is not a claim that none exists.

- Correction authors' `run_224.sh:15–16` and `train_and_eval_pytorch_corrected.py:52–61,83–87` select a custom3×3-stem R50 for corrected224, normalization0.5, no ImageNet initialization. Their optimizer is Adam1e-3,100epochs, milestone drops at50%/75%, best validationAUC (`:24–26,115–116,152–158`). They also evaluate test each epoch (`:136–138`); that monitoring is not adopted here. [Pinned author training source](https://github.com/kakumarabhishek/Corrected-Skin-Image-Datasets/blob/ac655ac5a5b93264c57d69d6d520778d915bfc89/DermaMNIST/DermaMNIST_Training/train_and_eval_pytorch_corrected.py)
- The existing [timm ResNet50 a1 ImageNet model](https://huggingface.co/timm/resnet50.a1_in1k) supplies an initializer, not medical weights. Our accepted checkpoint bytes are SHA256 `14fe96d1f9fb311a60490082d2077e6e60427dcfe21839ddf934cce948f72b0f`; existing path `/scratch/user/uqcche38/hu-mcd/models/torch/hub/checkpoints/resnet50_a1_0-14fe96d1.pth`. Reuse this exact file rather than a newly resolved default download.
- The downloadedR18 candidate is a different backbone with training-provenance/validation-overlap limitations. No automatic fallback to it or to custom-stemR50.

**M is a declared transfer-learning design, not verbatim replication of the correction paper's classifier.** Standard7×7stride2 stem, maxpool,2048-d global-average-pool, linear7-class head. Use existing timm0.6.13-compatible state layout. Source-recipe evidence supports Adam/milestone/validationAUC; the following lr,epochs,initialization,augmentation and normalization are operational proposal choices, not author-prescribed or proven optimal. The proposed first-round goal is to explain one frozen medical classifier, not test cross-training-seed stability. One seed supports only this conditional case study; it cannot characterize training variability.

## Frozen if approved / 待批准完整配置

| Item | M proposal / 提案 |
|---|---|
| Initialization/head | Load pinned a1 checkpoint strictly as1000-class backbone first, then replace only head with Linear(2048,7),bias=True; initialize underseed43 with recorded framework defaults; fine-tune all layers and BatchNorm |
| Label order |0akiec,1bcc,2bkl,3df,4mel,5nv,6vasc; preserve input row IDs/lesion IDs |
| Source image preprocessing | Native224RGB NPZ; convertuint8/255; ImageNet mean(.485,.456,.406),std(.229,.224,.225); no extra source resize, center/random crop, hair removal or color adjustment |
| Training augmentation | Independent horizontal/vertical flips,p=.5 each; train only; noMixUp/CutMix/color jitter. Operational orientation assumption, not clinical invariance proof |
| Sampling/loss | Shuffle all8215train images each epoch, drop_last=False, no weighted sampler, unweighted7-class CE, no label smoothing. Multiple views retain image weighting; report class/lesion distribution |
| Optimizer | Adam,lr1e-4,betas(.9,.999),eps1e-8,weight_decay0 |
| Schedule |50epochs maximum, batch64; lr1e-4 duringepochs1–25,1e-5 during26–38,1e-6 during39–50; scheduler applied after completed epochs25/38; no early-stopping search |
| Checkpoint selection | Highest validation macro one-vs-rest AUROC across all7classes; strict improvement, ties keep earliest; nonfinite/undefined criterion stops instead of substituting another metric |
| Seed/dataloader | Python/NumPy/Torchseed43; explicit per-worker seed and shuffle generator;2data-loader workers, CPUtorchthreads≤allocated request; record RNG state and environment |
| Precision | Float32, noAMP; match existing declared runtime: cuDNN TF32=True, matmul TF32=False, float32_matmul_precision=highest, cuDNNbenchmark=False, cuDNNdeterministic=False, deterministic_algorithms=False. Therefore seeded but **not promised bitwise deterministic**. A's original precision remains unchanged |
| Outputs | Actual resolved config, codeSHA, data/initializer hashes, all input IDs and roles, per-epoch losses/metrics/LR, last+best checkpoint with optimizer/RNG state, validation selection trace, best final validation/test logits and predictions with identities, warnings/stage records |
| Metrics | Overall accuracy, balancedaccuracy(mean7recalls), per-classrecall, confusionmatrix, macroOVR AUROC with eachclass's actual positives/negatives; image-level primary, no patient-independence claim. No arbitrary clinical pass threshold |

训练类别分布极不平衡：nv5309/8215；validation中df仅5图、vasc7图。选择macroAUROC不意味着小类估计稳定。保留并披露这一限制，不为了结果切换loss、重采样或选择指标。

## New-path checks and budget / 必要检查与资源

These checks serve a new medical7-class model path; they do not repeat the accepted dataset audit or launch SAM probes.

1. Before submission, local structural/config checks: head/label map, split-role separation, no test access in selection, schedule boundaries, stable serialization. The dedicated worker and launcher integration are now implemented and locally software-tested; proposed execution remains blocked until explicit approval and release binding. See `configs/medical/medical-classifier.proposed.json` and the preparation receipt. No GPU/model experiment was used for preparation.
2. At the start/end of the single authorized run if later approved: verify actual input/initializer hashes and environment; check finite tensors/loss, row correspondence, head dimensions. After fitting, strict state transfer to ordinary and HU input-masking-compatibleR50, compare batch8 unmasked/all-one-mask features/logits and head+bias reconstruction on fixed first8validation rows. Never use training-mode BatchNorm for evaluation. Compare same batches; retain failed arrays, do not loosen tolerances. Proposed existing numerical scale rtol/atol1e-4 for features/logits,1e-5 for linear-head algebra; these scales are proposed gates, not demonstrated equivalence of the new implementation. Failure pauses the affected downstream path and is not a reason to refit automatically.
3. Freeze best checkpoint SHA before reading final1227test outcomes. Retain ordinary errors; nonfinite outputs, identity mismatch or unexplained constant predictions require diagnosis. Numerical validity does not certify clinical suitability or understandable concepts.

| Stage | Upper resource budget / 上限 | Basis / 依据 |
|---|---|---|
| Local code/config preparation |0BunyaGPUhours; no model/data experiment | Authorized preparation only |
| Code deployment if M approved |1requestedCPU,1GiB,5min,0GPU | Existing SlurmGit release launcher; no scientific calculation on login node |
| One classifier run incl fixed mapping/reconstruction checks and final test |**1L40S,4requestedCPU,16GiB,4h,5GiBnewoutputs** |50×129=6450optimizersteps;410750training image visits;50×573=28650validation forwards, plus final test/checks. No medical backward-pass throughput measurement yet;4h is a ceiling, not measured runtime. Planning estimate1–3h has low confidence |
| Medical SAM/SSC/concept discovery and external inference |**Not budgeted/approved for execution** | Need classifier acceptance and medical region counts; no inference of SSC cost solely from image count |

Workload comparison with tail batches: **50/b64 =410750 training visits,6450 updates,28650 validation forwards;100/b128 =821500 visits,6500 updates,57300 validation forwards**. Both tail batches contain23. Near-equal updates do not imply equal gradients, BatchNorm statistics, compute or convergence; the author recipe also differs in architecture/initialization/LR. Fifty is not a verified sufficient or optimal length.

After the full schedule and identity/finite/adapter checks, freeze the selected checkpoint before test. Completion remains pending acceptance. Ordinary errors, low scores or still-improving curves are reported without extending the run. Nonfinite/identity/incomplete-workload failures block acceptance. Final constant argmax predictions require diagnosis before downstream acceptance; they do not alone prove a code bug. No clinical-quality threshold is invented after observing results. The bilingual acceptance matrix is in `NEXT_PHASE_CURRENT_DECISIONS_20260922.zh-en.md`.

No retries, automatic resume/extension, batch/precision change or HPO. If within-run progress predicts exceeding the cap, save failure/partial state and stop; an incomplete training schedule is not the planned accepted run. Keep onlylast/bestfulltrainingstates rather than50fullcheckpoints to respect5GiB; preserve everyepoch's metrics/selection evidence. Input/cache reuse is not itself new output.

Alternative: defer M and supply a traceable matching standardR50 checkpoint (0trainingGPUhours; a separately budgeted load/mapping check still needed). Exact authorcustomR50 training is a distinct protocol with different architecture/initialization, not a silent substitute. R18 remains an explicit alternative only.

**Approval boundary:** M would authorize only this source-classifier protocol/budget, subject to completed engineering checks and pinned release. It would not authorize400-imageSAM/SSC, medical insertion/deletion, Derm7pt model application, tuning, further classes or model alternatives. H/R/S decisions can proceed independently.
