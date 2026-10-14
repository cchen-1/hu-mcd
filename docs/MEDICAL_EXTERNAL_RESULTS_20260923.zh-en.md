# Derm7pt E224 外部结果 / External results — 2026-09-23

## 当前结论 / Current conclusion

**工程验收通过，不能据此判定医学适用性通过。** 部署28796207和外部28796209均COMPLETED、ExitCode0:0。全部101例按批准的R101/E224/S执行，无额外拟合、重训、预测、排除或失败重提。冻结的epoch10分类器仅将29/101例预测为MEL（28.71%）；源域留出为50/70（71.43%）。两者是不同队列，不能将42.72个百分点差直接解释为因果迁移损失。

**Engineering acceptance passed; medical adequacy is not certified.** Both deployment28796207 and external28796209 completed0:0. All101 cases followed approved R101/E224/S, without refitting, retraining, additional predictions, exclusions or resubmission. The frozen epoch10 classifier predicts MEL for29/101 external cases (28.71%), versus50/70 source held-out (71.43%). These are different cohorts; the42.72 percentage-point contrast is not a causal transfer-loss estimate.

## 执行与数据完整性 / Execution and data integrity

| Item / 项目 | Actual / 实际 |
|---|---|
| Full execution commit / 完整执行提交 | `dcc61934973aa9054a4e689ccf2334652447c575` |
| Deployment / 部署 |28796207,6s; requested1CPU/1GiB/5min; Slurm allocated2CPU, peakRSS72.29MiB |
| External / 外部 |28796209,3m16s;1L40S/8CPU/32GiB/30min; peakRSS6044.30MiB |
| GPU memory / 显存 |allocated6114689536bytes; reserved6677331968bytes; utilization time-series unavailable |
| Collected / 收集 |524 producer files,103302201bytes; all size/SHA256 checks passed; scheduler/log/deployment receipts also retained |
| Input / 输入 |101/101 original records and labels; all101 nonsquareRGB originals independently matched to local accepted ZIP; exact BICUBIC224×224 pixels reproduced |
| Fixed classifier / 分类器 |epoch10; SHA256 `07a6f65dbb23eb7c97510bef09c4b624a1a86007b27a64e324151fd1dfd2a655` |
| Fixed SAM / SAM权重 |ViT-H; SHA256 `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e` |
| Precision / 精度 |batch8,FP32/noAMP,cuDNN TF32on,matmul TF32off,highest,benchmark/deterministicfalse; unchanged fromD1 |
| Basis / 概念基 |all saved arrays exactly equal acceptedD1; C00132dimensions, complement2016; no external clustering |
| Validation / 验收 |all296 raw/effective masks, segment IDs, features, flags, scores and assignments; all101 coverage rows×2; all12+12 statistics |
| Reconstruction / 重构 |maximum relative feature error8.89e−14 for valid regions,5.18e−14 for full images; independent FC maximum absolute error1.91e−6 |

Slurm's deployment allocation of2CPU differs from the submitted1CPU request. The request was not increased; scheduling/allocation granularity is a plausible explanation, not verified here. No resource expansion was requested. / 部署实际分配2CPU与申请1CPU不同；没有扩大申请。调度分配粒度是可能解释，未单独核实。

## 分类器与概念分配 / Classifier and concept assignment

| Scope / 范围 | Images | Raw regions | Zero features | C001 / valid | Complement / valid | MEL predictions |
|---|---:|---:|---:|---:|---:|---:|
| Source discovery / 源域发现 |400distinctlesions |917 |68(7.42%) |514/849(60.54%) |335/849 |341/400 |
| Source held-out / 源域留出 |70images/61lesions |157 |14(8.92%) |82/143(57.34%) |61/143 |50/70 |
| External R101 / 外部 |101cases, patientsunknown |296 |37(12.50%) |94/259(36.29%) |165/259 |29/101 |

External prediction counts in the frozen label order: AKIEC2, BCC2, BKL22, DF0, MEL29, NV46, VASC0. All101 reference diagnoses are in the approved MEL grouping, including one metastasis; this is not a seven-class test set. Disease AUROC, seven-class balanced accuracy, specificity and clinical sensitivity across a representative population are not estimated here.

Full-image logits were computed **before SAM and without segment masks** (`hpc/medical_external.py:118–128`, execution commit above). Thus the observed29/101 full-image MEL result is already present at the classifier-input stage; downstream SAM/discovery assignments do not produce these predictions. This separates that outcome from segmentation quality, while leaving domain/preprocessing/classifier explanations unresolved.

外部预测按固定标签顺序为AKIEC2、BCC2、BKL22、DF0、MEL29、NV46、VASC0。全部101例均属批准的MEL分组，含1例转移性诊断；不能估计七类完整表现、疾病AUROC、特异度或代表性临床人群的敏感度。

全图logits在**SAM之前、不使用片段mask**时计算。因此29/101的全图MEL结果已存在于分类器输入阶段，不是下游SAM或概念分配产生的。仍不能区分输入域、预处理与固定分类器能力的贡献。

Every image retains at least one valid region. All37 zero features have nonempty effective masks. They remain invalid(−1), never arbitraryC001; no deletion to improve results. C001 appears in89/101 images, with mean coverage0.570997 and median0.743662;12 completed images have genuine zeroC001coverage. Complement appears in88images with mean0.242793; it is not a second learned concept. Denominators differ between region assignments and image-level coverage.

每图至少一个有效片段；37个零特征对应非空mask，保持−1而不是任意分给C001。C001出现在89/101例，平均覆盖57.10%、中位74.37%；12例是真实零覆盖，不是缺失。补空间出现在88例，平均覆盖24.28%，不计入学习概念。分配比例的片段分母与覆盖率的图片分母不同。

## 征象关联与视觉审阅 / Sign association and visual review

Research question: **does the area covered by frozenC001 differ across existing image-level sign groups?** The outcome is descriptive image-level association, not proof thatC001 equals a named sign or localizes it. All12 prespecified non-absent-vsABS comparisons were completed forC001;12 additional complement comparisons are separately reported. Independent pairwise AUC calculation and all group means/counts match saved outputs. No p values, bootstrap, direction reversal or outcome-dependent selection.

研究问题：**冻结C001覆盖的图片面积是否随已有征象分组而不同？** 只提供图片级描述性关联，不证明语义等同或空间定位。12项C001及独立12项补空间均完成；独立计算的AUC、均值和分母一致。无p值、bootstrap、翻转方向或根据结果选取比较。

Irregular dots/globules versusABS has AUC0.660218(84vs12), mean difference0.197658. This is one descriptive contrast among12, not a selected significant discovery. Regular pigmentation(1vs38,AUC0.684211) and regular streaks(1vs41,AUC0.268293) each have only one positive case; they are especially unstable. All contrasts, including low/negative-direction values and complement results, remain inCSV/report. Missing counts are zero; other sign categories are disclosed rather than pooled intoABS.

不规则dots/globules对ABS：AUC0.660218（84对12）、均值差0.197658，仅为12项之一的描述结果。规则pigmentation和规则streaks各仅1个阳性病例，尤其不稳定。全部正负方向结果与补空间保留；没有把其他类别合并为ABS。

The private report provides all101 original images at original aspect ratio, all296 effective-region triptychs, saved seven-class logits and raw labels; C001/complement top10(one image each), fixed random10(RandomState4303 independent of training/SAM), lowest-margin3, and first3 zero-feature diagnostics. The cross-domain figure uses the first2 saved top and first2 fixed random members in each domain. Display enlargement adds no detail. Initial source clusters remain in the previous gallery; no external initial-cluster membership is invented. Human reviews startUNREVIEWED and preserve the existing source-review storage.

私有报告提供全部101原图、296个原图/选中overlay/仅选中区域面板、logits与原标签；C001和补空间分别展示top10、独立seed4303随机10、低间隔3；另有前3个零特征诊断。跨域对照固定取各域原型及随机清单前2例，不挑选好看结果。无外部聚类，不伪造初始成员。展示放大不增加细节；原人工审阅记录保留，新外部审阅为未审阅。

Limited nonclinical AI inspection of the fixed comparison: external top case760S0000 andcase793S0002 retain broad areas; randomcase578S0001 andcase781S0001 retain more bounded central dark regions. These four newly inspected external examples cannot establish population-level quality or medical meaning. / 有限AI视觉观察：外部top的case760S0000和case793S0002保留较大区域；random的case578S0001和case781S0001保留更局部的中心深色区域。仅4个新增外部例子的非专业观察，不足以判断总体质量或医学语义。

## 异常、解释边界与后续 / Issues, limits and next steps

- MED26-021: E224 approval now fulfilled and accepted. All101 original aspect ratios were changed by the explicitly approved warp. Preserve original images and mapping. This is same-size transfer, not native-resolution SAM evaluation.
- MED26-018: external37/296 nonempty-mask zero features extend the source observation. Numerical reconstruction passes but does not explain feature disappearance; no per-layer causal check was performed.
- MED26-023: lower external MELprediction29/101 and learned assignment94/259 are observed. Model limitations, domain, geometry and cohort composition are plausible contributors; none is isolated. Fixed-model interpretation remains the goal, not automatic classifier optimization.
- MED26-024:12descriptive sign contrasts include two one-member groups and unknown patient independence. No expert semantic/spatial validation; nominal associations are not medical claims.
- MED26-008: prior H has214threshold-similarity pairs across20R101cases; still unreviewed/unconfirmed. All101retained. Exact-match negatives do not prove independence. Earlier nine-case basic review does not resolve overlap or medical labels.
- MED26-011: initial local acceptance omitted Pythonseed43 when replaying released random region-label composition. This caused an order mismatch, not an established remote defect. After mirroring original seed/order, every saved mask matched. The first failed local-check traceback remains; no model job was retried.
- EV26-003: one `np.bool8` deprecation warning recorded; no observed integrity impact.
- MED26-025: deployment requested1CPU and Slurm allocated2CPU;6s execution. Request unchanged, recorded allocation difference only.

重点限制：已批准的224形变、外部预测较低、非空零特征、稀疏征象分组、未知患者独立性与未确认重叠。没有把这些事实直接认定为论文结论错误，也没有把数值验收通过当作医学认证。本地RNG验收错误已修复并保留历史；其他未验证事项未标为已解决。

**Next useful work:** review fixed randomC001/complement members and prioritized H candidate pairs, using existing private pages. If a new comparison is approved later, isolate input geometry under the same classifier/basis/cohort, with a concrete separately frozen protocol and budget. This turn submits no experiments. The four failed A robustness classes remain partially complete; no tolerance relaxation or retries. Their accepted conditions and missing counts remain in the existing report.

**下一步：**先审阅固定随机成员与H相似候选。若新增比较，优先在相同分类器、概念基与病例范围下隔离输入几何影响，另行冻结配置和预算后审批。本轮不提交新实验；A四类原失败继续保留部分结果和缺失，不放宽容差、不重提。

## Evidence / 证据

- `artifacts/bunya/medical-external-results-20260923/{sacct.psv,collection.json,acceptance.json,per-image.csv,per-region.csv,contrasts.csv,concept-summary.csv}`.
- Immutable producer outputs: sibling `collection/`; deployment readiness/provenance and original logs retained.
- `scripts/{collect_medical_external_results.py,accept_medical_external_results.py,build_medical_external_report.py}`; generation hashes and display selections accompany the report.
- Existing single entry: `artifacts/bunya/ten-class-review/report/index.html`; linked external bundle, not a separate public site. Private images/data are not pushed.
- Approved scope: [E224 protocol](MEDICAL_EXTERNAL_EXECUTION_20260922.zh-en.md); prior accepted discovery/robustness: [22Sep batch](BATCH_RESULTS_20260922.zh-en.md). Historic proposals remain history, not fresh authorization.
