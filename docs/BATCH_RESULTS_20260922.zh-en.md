# 两条工作线结果验收 / Two-workstream result acceptance

Recorded22September2026 (Brisbane). Source: fresh final Slurm accounting, hash-verified collected artifacts and cached numerical/mapping replay. Local scripts do not run models or refit concepts. Existing ImageNet discovery/evaluations, medical H/M evidence, source datasets and failure histories remain preserved.

记录日期2026-09-22布里斯班。状态依据本轮最终Slurm记录及已收集文件，非旧STATUS推断。所有远端计算／文件检查仍在Slurm；本轮本地统计仅消费缓存。

## 实际完成范围 / Actual completion

- **A:** six complete five-condition classes; four failed at radius2erosion reduced-batch controls. All ten original-batch identity controls and radius1erosion results accepted. No automatic reruns.
- **医学D1:**400discovery images/distinct lesions plus all70held-out/61lesions accepted for engineering consistency. Three initialclusters → one learned32-dimensional subspace; complement2016dimensions is separate. This is not clinical/semantic validation.
- **External:** explicitly approved E224/R101/S, immutable commit`dcc61934973aa9054a4e689ccf2334652447c575`, pushed to fork/reproduction. Deployment**28796207**, inference**28796209**, submitted once; only post-submission snapshotPENDING/dependency. Not collected or accepted. Wait for the user's completion notice.

A六类完整、四类失败；医学400/70完整验收。外部两作业已提交，不把准备／排队写成完成，不等待或自动收集新批次。

| Class / 类别 | Job | State / Exit | Elapsed | Peak host RSS MiB | Accepted scope / 验收范围 |
|---|---|---|---|---:|---|
|airliner|28731217|FAILED / 1:0|00:01:38|1574.88M|identity0,erosion1|
|golden_retriever|28731219|COMPLETED / 0:0|00:01:25|1574.63M|identity0,erosion1,erosion2,dilation1,dilation2|
|hummingbird|28731220|FAILED / 1:0|00:05:45|1350.66M|identity0,erosion1|
|Siamese_cat|28731221|COMPLETED / 0:0|00:00:49|1483.46M|identity0,erosion1,erosion2,dilation1,dilation2|
|tailed_frog|28731222|COMPLETED / 0:0|00:01:50|1516.18M|identity0,erosion1,erosion2,dilation1,dilation2|
|container_ship|28731223|FAILED / 1:0|00:05:52|1392.79M|identity0,erosion1|
|police_van|28731224|COMPLETED / 0:0|00:02:49|1756.78M|identity0,erosion1,erosion2,dilation1,dilation2|
|beach_wagon|28731228|FAILED / 1:0|00:00:35|1817.01M|identity0,erosion1|
|zebra|28731229|COMPLETED / 0:0|00:01:22|1474.00M|identity0,erosion1,erosion2,dilation1,dilation2|
|ox|28731232|COMPLETED / 0:0|00:00:38|1639.38M|identity0,erosion1,erosion2,dilation1,dilation2|

All A requested1L40S/4CPU/16GiB/20min each, no observed OOM/timeouts. Slurm RSS is host memory, not VRAM. D1 requested1L40S/8CPU/32GiB/2h; actual12m55s, hostRSS7955.82MiB, PyTorch GPUallocated peak5.70GiB/reserved6.22GiB. D0 completed29s, deployment6s; requested1CPU each but accounting allocated2CPUs—scheduler allocation differs from request, not manual expansion (MED26-010). Existing receipts reused.

A无OOM或超时证据。D1的12分55秒是实测；E224的约3分钟SAM外推只是估计，不保证运行时长。

## A：可支持的结论与失败定位 / Supported sensitivity result and failure diagnosis

The primary metric is mean per-image retention of valid baseline assignments, then equal-class aggregation; invalidated regions remain in the denominator. All ten radius1erosion results give **78.2522%**. This is not segmentation accuracy or human concept quality. The radius2/dilation figures below cover only six successfully completed classes and cannot be called ten-class estimates or directly compared with a different class set.

主指标保留基线有效片段分母，不丢弃扰动后的零特征。十类r1保留率78.25%；r2／扩张仅六类，缺失不填0，不因样本漂亮而改变纳入范围。

| Condition | Completed classes | Equal-class image-mean retention |
|---|---:|---:|
|identity0|10/10|100.000000%|
|erosion1|10/10|78.252242%|
|erosion2|6/10|69.079271%|
|dilation1|6/10|80.413509%|
|dilation2|6/10|73.521220%|

**ROB26-005 — observed failure:** all four stopped at `hpc/mask_sensitivity.py:198`, testing unchanged inputs after one empty region reduced an8-row batch to7. Original same-batch features matched exactly. Cached evidence reports maxabsolute feature differences0.000450–0.001063 and maxrelative1.24e−3–1.94e−3. All28observed reduced-batch control assignments remain unchanged, but feature and score tolerances are not passed. Do not override the gate based only on argmax stability.

**Possible explanation:** changing batch shape may change GPU arithmetic under the preserved precision/kernel implementation. No new GPU diagnostic was run to attribute a specific operator or TF32 causal effect. The common trigger is established; the precise numerical mechanism is not. This is distinct from changing normalization: this path uses `norm_batch=False`.

四类均在新增缩批路径触发，不是原batch重建失败。不能仅凭这28例不改分配就放宽阈值，也不声称已证实论文结论错误。下一步如补齐四类，应先准备保留原batch槽位的具体空区域处理及针对性门槛，不新增SAM／概念拟合；补跑需要单独批准，当前未提交。

All28435planned region-condition slots are listed:21823completed,6612explicitly missing. Geometry/IoU/symmetric difference, featurecosine/relative change, original winner versus strongest competitor margin (including negative margins), validity/transition tables and frozen area/margin strata are available. Class intervals use image-level pairedbootstrap2000,seed20260921; uncertainty is conditional on fixed discovery/model. Fixed random2original regions/class plus first valid erosion1switch are visualized; diagnostic selection is labeled and original prototypes remain in the existing report.

## B：医学发现完整性与解释范围 / Medical discovery and interpretation

| Split | Images / lesions | Raw / valid regions | Nonempty zero features | Learned / complement assignment | Full-image MEL predictions |
|---|---|---|---|---|---|
| Training |400/400|917/849|68 (7.42%)|514/335|341/400|
| Held-out |70/61|157/143|14 (8.92%)|82/61|50/70|

Each fixed image is processed and retains a nonzero region. Raw SAMproposals/metadata, composed/effective masks, features/logits, ordered segment IDs, initial labels, q1outlier flags, PCA membership, bases, final assignments and examples are accounted for. Independent cached replay checks every effective mask against the original conditional erosion and every valid assignment. Full-image and segment reconstruction errors are below8.5e−14 relative in replay; the stored same-batch ordinary/masked adapter differences are0.

固定400/70未重选；掩码方向、原始／有效mask关系、零特征、初始／PCA／最终映射及原型排序通过。数值一致不意味着医学适用性或概念可理解性。

The released Kheuristic returns3from retained regions per image. Initialclusters807/31/11; minimum50 retains only807. Its final destinations:514C001,293complement; filtered31+11also map tocomplement. Therefore807initial members ≠514final members. The single basis has32dimensions; completeness**0.662911**, globalimportance**0.439451** (squared target-weight component norm/weight norm squared), rank1/1. Complementimportance0.560549 is separate. No concept search, q99 removal, threshold adjustment or classifier change occurred. `outlier_percentile=1.0` keeps finite rows; it is not1%removal (MED26-015).

**Visual observation (MED26-020, provisional):** inspected only12fixed top3/random3 examples across both splits. Some top examples retain broad pigmented regions; randomheldoutISIC_0026150 andISIC_0029454hide central regions while retaining surrounding skin. This suggests that a single subspace groups differing region patterns. It is not a medical diagnosis, a confirmed semantic label or a claim about every member. All top10/random10,lowmargin3per split and random8per initialcluster remain available, with original/overlay/selected-only views and unreviewed human forms.

部分原型与随机示例并非同样清晰的区域模式，需审阅全部固定样例而非只看top。补空间不默认是失败；C001也不默认等于病灶或某项征象。

**Cross-batch comparison (MED26-022):** classifierheadweight/bias exactly match the accepted Mcheckpoint. All70saved full-image predictions match priorbatch64test output, bothMEL50/70. Maxabsolute logit difference**0.0184037685** betweenbatch64andbatch8 is preserved. This numerical difference is not hidden or called bitwise equivalence; the same-batch adaptation itself haszero differences. GPUbatchshape effects are a plausible unisolated explanation. External execution uses the already approvedbatch8.

## 差错、警告与版本 / Errors, warnings and versions

- **C-009:** inherited tuple assertions in `utils/utils_mcd.py:88,94` are ineffective and emitSyntaxWarning. Independent completebasis geometry/reconstruction gates passed for32+2016here; no observed malformed complement. Shared scientific source remains unchanged.
- **MED26-018/019/020:** zero features, one retained subspace, and provisional visual heterogeneity are disclosed research limitations, not silently repaired to improve results.
- **ROB26-005/006:** failed controls and missing conditions are real affected-scope blockers; six independent complete classes remain usable.
- **EV26-003:** pinnedskimage `np.bool8` deprecation in all11jobs; localMatplotlib cache fallback warning. No observed changed arrays, no silent dependency upgrade.
- **MED26-011:** local1917-file transfer resumed in bounded batches for efficiency; initial collector-schema/importpath and unconfigured localGitauthor issues corrected. An initial SFTPupload failed on a missing parent before anyjobsubmission; fixed transferdirectory, retainedfailure. DraftE224pixel-hash preflight corrected to the acceptedmode/shape/dtypehashformat before anyinference. These are resolved engineering issues, not new failed experiments.

Acommit40418edfa4e4ac81eda02d954fe59de52a0f14f0 andD1commit5623abe8a843deaea7ab838a8c76c41afbad565e have **identical eight shared scientific files**, also unchanged inE224commitdcc6193…. Wrappers differ by intended new execution paths; “same source” does not exempt numerical controls. All raw workeranomalies andstderr preserved and linked; the unified anomaly registry is append-only.

## E224的当前边界 / Current external scope

Approved deployment1CPU/1GiB/5min; inference1L40S/8CPU/32GiB/30min/4GiB, including Sdescriptive statistics. Fixed101author-MELofficialtest cases, includingmetastasis1. Originalimages retained; privatePILbicubic224×224workinputs beforeSAM change aspectratio/texture resolution. Source400sampling and all70heldout unchanged. Fixed classifier and one concept pluscomplement; no fitting/retraining/SAMprobe.

PriorH:214perceptualcandidatepairs across20R101cases, not confirmedduplicates; no automaticexclusion, patientindependence unknown. One-case regularpigmentation andstreaks groups remain tiny; all12contrastdenominators displayed when externalresults arrive. Unioncoverage is an image-level response and association does not establish semantic identity, clinical utility or spatiallocalization. No p-values/bootstrap/direction-flipping. Prior9-case userreview has limited nonprofessional scope and does not resolve these issues.

**Already submitted:**28796207→28796209; no furtherexecution required from the user now. Report pending, not placeholdercurves. **Future optional decision:** whether to authorize a targeted four-class Acompletion after a concrete fixed-slot correction is reviewed; no rerun automatically submitted. No new medical sensitivity/baseline/class expansion proposed merely to improve this result.

## Deliverables / 交付

- Single bilingual entry: `artifacts/bunya/ten-class-review/report/index.html` → batchreview gallery.
- New scientific figuresPNG/PDF, original/overlay/selected-only galleries, masksensitivitychanges, stableIDs and manualreviewforms: `artifacts/bunya/batch-review-20260922/bundle/`.
- CSVsummaries, complete28435slotledger, conceptindex, reviewtemplate, cluster→assignment, geometry/transitions/strata, rawwarnings and acceptance in the samefolder/root.
- Generators: `scripts/accept_next_phase_batch.py`, `scripts/analyze_mask_sensitivity_geometry.py`, `scripts/build_next_phase_batch_report.py`, `scripts/build_mask_sensitivity_examples.py`; existingreport integration is preserved. Private only, no public image/reportupload.
