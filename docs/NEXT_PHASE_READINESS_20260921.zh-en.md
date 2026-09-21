# Next-phase readiness / 下一阶段实际准备与待批协议

2026-09-21 · private / 私有 · **NOT an experiment approval / 不代表实验已批准**

## Actual state / 实际状态

Authenticated Bunya master pid88416 passed; actual `squeue` was empty and `sacct -S 2026-09-17 -X` returned no new jobs. This is a scheduler snapshot, not a remote cache-retention audit. No new A/B model run is evidenced. The accepted ten HU discoveries/evaluations and Golden MCD/ACE reference results remain reusable. Earlier `STATUS.json` pending/acceptor fields were archived and corrected from final receipts, not from plans. The historical discovery CSV is preserved; a new joined current table gives accepted evaluation status.

现有认证连接有效；实际队列为空，9月17日以来未查到新作业。此前验收结果直接复用；远端scratch缓存是否仍完整需Slurm核对，不由本地副本推断。历史状态字段已归档，当前状态依据验收回执修正。没有新增模型实验。

## A: concrete candidate for approval / A待批准组合

All10 classes, fixed500 validation images,5687 raw regions,413 exact-zero baseline features,5274 valid baseline regions; no image lacks a valid region. Accepted model configs all specify3×224×224; original dataset resize/binarization is used rather than assuming the default config's interpolation string describes actual code. Actual `ConceptDatasetClass` uses its torchvision Resize path, with source conditional erosion at raw mask area>.25 and masking_mode=-1, before final binary masking.

逐图／区域ID、原mask与最终mask哈希、原始特征行、零特征和有效行映射已核对；原缓存文件与输入图片哈希匹配已有验收记录。10类网格均224×224，batch8、原权重／精度保持。5条件共28435条账本记录，不以CNN实际调用数代替总区域数。

- Candidate conditions: identity; erosion/dilation with Euclidean discrete disks r1/r2 at final224 grid, zero exterior, truncation at boundary, independent from original effective mask. No re-erosion, hole filling or overlap normalization. **Pending approval.**
- Candidate scores: accepted discovery path `norm_batch=False`: `s_k=||P_k^oblique f||₂ / ||f||₂`, with learned bases and complement fixed. Not independent orthogonal projection. This matches accepted `run_smoke.py:397–402`; benchmark's imagewise max-norm scaling is a distinct path. Raw component norms may also be saved as diagnostics but do not replace the primary definition.
- Primary candidate: per-image valid baseline retention, counting new empty/zero as nonretained and keeping them in the baseline denominator. Original zero rows retain their IDs and historical benchmark field, not valid concept labels. Class means then equally weighted10-class macro average. Learned/OC baseline groups reported separately. Conditional valid/valid agreement is secondary, with explicit denominator.
- Candidate uncertainty: paired images within each class,2000bootstrap replicates, analysisseed20260921. Baseline area/margin quartiles computed within class from valid original regions, exact cuts saved before inference; ties may leave empty bins (reported, not re-binned). All conditions for each image resampled together. No fitted-seed or patient-independence uncertainty claim.
- Geometry-only preview, **not a model result**: identity/r1/dilations have no empty masks; r2 has6 (airliner1, hummingbird1, container_ship1, beach_wagon3). Four of these have original zero features, two were valid. Maximum nonempty condition rows28429. New nonempty zero-feature counts are unknown until an approved run.
- Original CNN batch slots are recorded. Empty masks are not fed to an unsupported CNN path, and later IDs are not moved across original batch boundaries. Six r2 batches each lose1slot. For their42 surviving rows, an additional unchanged-mask control at the same reduced batch size is required to measure kernel/batch effects before attributing differences to geometry. This adds at most42forward rows (total28471), no new samples or SAM. If that control fails, stop the affected class, do not loosen thresholds or silently insert filler regions.
- Identity gate: masks bitwise exact; feature comparison max-relative1e-4 and elementwise rtol/atol1e-4 using the previous same-batch numerical-check scale; scores rtol/atol1e-5 from the accepted numerical replay; assignments identical on originally valid rows; original zeros matched exactly. Applying these scales to the new cached-feature replay is a declared gate, not a prior empirical guarantee. Any failure requires diagnosis, not automatic tolerance relaxation. The exact CPU decomposition implementation must reproduce the saved oblique score path before a GPU run consumes it.

主协议、得分定义、空区域分母、bootstrap及资源仍待用户选择。CPU原掩码重建和简单形态学测试通过，不等于GPU identity控制通过；后者必须在正式作业第一阶段完成。6个受空mask影响batch需额外同大小控制，防止此前已知batch相关数值差异混入研究结论。

### A resource proposal / A资源提案

| Item / 项目 | Proposed / 提议 | Evidence / 依据 |
|---|---|---|
| Jobs |10 independent class jobs; max2 concurrent / 每类独立，最多并发2 | Per-class cache/identity failures remain isolated |
| Each |1×L40S,4CPU,16GiB,20min | No new SAM/SSC/PCA; same masked ResNet50/basis |
| Total cap |3h20 GPU-hours; no retry/automatic expansion | Upper allocation bound, not predicted runtime |
| Estimated elapsed | Roughly5–10min/class / 粗略每类5–10分钟 | Original validation feature stages total23.99s; original assignment+prototype stages total453.98s across10classes. Five-condition workload is an extrapolation; stage includes display, and new path/CPUthreads/IO differ. Not based on31/33s baseline flipping jobs |
| New storage | Expected<2GiB; cap10GiB | Packed masks≈170MiB, float32 features≈222MiB, optional all1000logits≈109MiB; scores/mappings/configs and copied bases add overhead. Existing caches are read-only and not counted as new output |
| CPU preflight | Included in separate pending preparation plan |550 named hash checks from accepted per-class cache/config/basis manifests |

Resources and scientific protocol are pending, not submitted. Actual execution must bind a complete commit and frozen plan, retain stage progress and collect independently after the user reports completion.

## B: verified metadata, access gaps and model choice / B事实、材料缺口和模型方案

The official [Zenodo12739457 API](https://zenodo.org/api/records/12739457) still identifies the224 archive as1,131,258,316bytes, MD5 `84920fb70c83b234c295b6f0d4ae2bc0`; metadata MD5 `1c54bc9f483e97a9d7dbbde076a45900`. A single55s local attempt received11,901,436bytes before the time cap. Partial bytes are preserved and are **not an accepted archive**. API availability has improved relative to earlier504responses; complete download remains unknown.

源域元数据计数如下（**不是图像验收结果**）：

| Split / 划分 | All images / 全部图片 | Melanoma images / 黑色素瘤图片 | Melanoma lesions / 病灶 |
|---|---:|---:|---:|
| train |8215|1021|533|
| val |573|22|20|
| test |1227|70|61|

Training melanoma:374lesions have multiple images, maximum6images/lesion. Candidate image-uniform400 (Python Random43 shuffle sorted image IDs) contains315distinct lesions. Candidate lesion-first400 would cover400lesions with one independently selected image each; its exact within-lesion RNG rule must be frozen after user choice. Neither list is approved/frozen. Source test uses all70images with lesion grouping for uncertainty; no claim of70independent lesions. No patient IDs.

Published current metadata has0cross-split pairs among18author-confirmed duplicate pairs, after stripping their documented `.jpg` filename extension. This does not substitute for exact-archive array auditing or a cross-dataset identity check. Legacy OneDrive still has7known cross-split pairs and remains blocked; the derived candidate remains unapproved. No image or split was modified.

官方 [Derm7pt入口](https://derm.cs.sfu.ca/Download.html)要求填写访问申请后使用发放的登录信息。已请求用户提供获取后的私有路径，不索取聊天明文密码，不代交表单。当前检查的本地数据目录没有完整发布；外部目标类与七项征象的实际正负数、跨源重复、可用病例数均**未知**。不以论文1011总病例或413/203/395划分冒充本项目已验收计数。

Pinned official loader commit `ce436877573c0a53cfa0d224bda20d9479b795aa`: `dataset.py:109,358–383` defaults to border crop25, and `:47–49` groups melanoma subtypes including metastasis. Neither default is silently adopted. External diagnosis grouping, exclusion/handling of metastasis, official-test versus exploratory scope, per-case concept response and multicategory sign contrasts await actual metadata counts. No automatic label binarization, cross-diagnosis expansion or pixel-localization claim.

### B model evidence and a bounded training draft / B模型证据及训练草案

Limited search scope: correction authors' pinned repo `ac655ac5a5b93264c57d69d6d520778d915bfc89`, official MedMNIST experiments, prior inspected FailCatcher/HuggingFace candidates, and targeted public-source search. No newly verified checkpoint meets standard ResNet50 + corrected-native224 training/split provenance. This means **not found in this search**, not “none exists”.

| Candidate | Evidence | Current decision |
|---|---|---|
| Correction-author R50 native224 | `run_224.sh` selects custom3×3-stem network without `--resize`; `train_and_eval_pytorch_corrected.py:23–26,83–87,115–116,153–158` uses Adam1e-3,100epochs default, drops at50/75%, best validationAUC, no ImageNet pretraining | Useful recipe evidence, not the requested standard pretrained R50; no verified checkpoint found |
| Official MedMNIST R50_224 | Prior official code/record review: original28 resized224, old splits | Not native224 corrected-data training; not silently selected |
| Downloaded FailCatcher R18 | Hash/strict load already accepted, provenance limitations documented | Explicit alternative only; not the restored R50 mainline |
| Existing ImageNet R50 a1 initializer | Standard7×7 stem; SHA256 `14fe96d1f9fb311a60490082d2077e6e60427dcfe21839ddf934cce948f72b0f` | Candidate initializer only; no medical training result |

**Training draft, not submitted or approved:** standard timm ResNet50 from existing a1 initializer, replace head with7outputs in fixed `akiec,bcc,bkl,df,mel,nv,vasc` order; all layers fine-tuned; native224RGB, ImageNet mean/std; no lesion crop, hair removal, color adjustment or strong augmentation. Candidate training augmentation: horizontal/vertical flips eachp=.5; no test augmentation. Adam lr1e-4, betas(.9,.999),eps1e-8,weight_decay0; unweighted CE;50epochs, batch64, drops×.1 at epochs25and38, seed43. Select highest validation macro one-vs-rest AUROC, tie earliest; save every epoch's class metrics/logs, last and best checkpoints with RNG/optimizer state. Fixed test only after selection; no HPO/reselection on test or Derm7pt. Report ordinary/balanced accuracy, per-class recall, confusion and macroOVR AUROC with denominators.

该草案是会议要求的标准预训练R50迁移方案，**并非作者训练脚本原样复现**：架构、初始化、归一化、增广、学习率及epoch均不同。候选50epochs为410750训练图访问、6450个optimizersteps（每epoch129batches，保留尾批）；暂无本机医学训练吞吐实测。候选上限1×L40S、4CPU、16GiB、4小时，磁盘5GiB；这是停机上限而非预言耗时，不训练直到精确数据验收和用户另行批准此完整协议。不因超时而自动续跑或扩容。

Medical SAM discovery/SSC and Derm7pt application budgets remain unfrozen: external counts unavailable and medical region counts unknown. No GPU job is submitted for them, and no repeated small SAM probe is planned. Source discovery can proceed independently of external data once its own data/model/protocol/budget are accepted. Final external independence claims still require cross-dataset review.

### Optional CPU preparation batch awaiting approval / 待批准CPU准备批次

1CPU,8GiB,2h,0GPU,newdisk≤3GiB. First check550named A cache/config/basis hashes, recording failures by class. Independently fetch exact Zenodo file once into a new job-specific private folder (or reuse an existing exact file only after checksum verification), maximum110min, no retry. Verify published MD5, array counts/shapes/labels, pixel/split duplicates and18known duplicate pairs. No sampling, classifier, SAM or deployment of scientific changes. Bunya throughput unknown; current WSL rate implies roughly87min, not an HPC runtime claim. Alternative: user-supplied exact-MD5 file, followed by a short CPU audit.

After an authorized submission: save fullSHA/config/resources/output receipt, one scheduling check, then stop waiting. User will notify completion; only then collect/accept/analyze. Failures are not resubmitted automatically.

## Evidence / 证据

Local evidence root: `artifacts/bunya/next-phase-20260921/`.

- `scheduler-current.txt`, `accounting-since-final.psv`, `refresh.json` — actual scheduler refresh.
- `inventory/A-region-ledger.csv`, `A-bindings.json`, `A-inventory.json` — all region identities, candidate geometry-only counts and accepted-cache bindings.
- `A-workload.csv`, `B-source-metadata-counts.csv`, `B-sampling-feasibility.json`, `B-current-metadata-known-duplicates.json` — denominators and budget evidence.
- `sources/` — pinned source snapshots, current Zenodo metadata and bounded partial-download receipt.
- `geometry-tests.log` —3targeted tests PASS. Initial fixture omitted original-image pointer; repaired before any model use, failed log preserved. Dependency `np.bool8` warning remains recorded; initial matplotlib temporary-cache warning fixed by explicit local cache path.
- `preparation.plan.draft.json` — not approved; no job ID. `hpc/next_phase_prepare.py` is Slurm-only and preserves independent A/B failures. `hpc/final_mask_intervention.py` wraps the source dataset after binary resize; `scripts/prepare_next_phase_inventory.py` reproduces local inventory without inference.
- Same private report entry: `artifacts/bunya/ten-class-review/report/index.html`. New research outputs remain **not run**, not0. English Obsidian notes track scientific limitations; canonical anomalies preserve every new failure/warning.

No confirmed private RDM path yet; local source/accepted-result copies exist, but persistent external archival is not claimed. No public image/report upload.
