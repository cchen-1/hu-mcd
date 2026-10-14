# Paper coverage and result interpretation / 论文覆盖与结果解读

Date: 2026-09-24 (Brisbane). Report-only work; no Bunya computation, model predictions, experiments, resampling or scientific parameter changes.

## Scope / 覆盖

- Accepted / 已验收: ten-class HU-MCD discovery; ten-class MCD and ACE — HU-MCD released-code reference discovery; ten classes × four methods (including Random) × two directions = 80 evaluation cells. Golden results are reused, not rerun. ACE retains138 candidates under its nominal test:113 above and25 below control. These are not human-quality labels.
- Partial comparison / 部分可比: supplement Table2 and paper Fig4. Five of ten HU-MCD initial counts match published counts; other differences are one. Maximum absolute completeness difference from published rounded values is0.02255. Author exact input identities and raw curve values are not established; no exact-reproduction claim.
- Missing / 未完成: paper Table1 human study, including task prediction, recognition, description consistency/distinctness and related unique-description analysis. Current informal/AI review does not substitute for the41-participant study. Benchmark MCD has fewer than5 concepts in six classes; Golden ACE has3. The paper's MCD human-study minimum5 requires a separately specified protocol, not a silent change to existing results.
- Missing / 未完成: supplement Fig7 Input Masking versus original-scale grey inpainting versus cropped/resized inpainting, both evaluation directions. Segmentation erosion/dilation is a different extension, not this ablation.
- Existing additional work / 额外工作: reconstruction/identity checks, cluster-to-assignment diagnostics, zero-feature analysis, partial segmentation robustness, medical transfer. These do not fill the human-study/inpainting gaps. Independent medical work follows its own latest handover and job records.

## Interpretation / 解读

At the first actual concept step, mean pixel fractions are HU-MCD23.66%, MCD23.00%, ACE22.88%; deletion accuracies36.0%,56.6%,60.32%; insertion62.6%,49.0%,39.28%. HU/MCD contribute500 images and ACE499. The early pooled pattern is directionally consistent with stronger predictive influence for HU-MCD under the executed protocol. Similar average area is not matched image-wise area, and later contributors/class weights change. No AUC, interpolation, missing endpoints or significance test was added.

类别例外不能隐藏：Hummingbird MCD首步84%对HU-MCD66%，同时暴露约38%对13%像素；Police Van和Zebra也不能写成HU-MCD处处领先。统计比较必须同时读像素、准确率和贡献人数。completeness、概念数和分配率都不是可理解性指标；ACE两侧名义检验可保留低于控制的候选，影响可延续到评价。

The report contains34 bilingual chart-discussion blocks: observations, interpretation, class-specific exceptions, uncertainties and next questions. The ten-class aggregate is primary; Golden now shares the same per-class comparison layout. Historical batch pages remain traceable.

## Visual comparison / 概念对照

All ten classes have ACE | MCD | HU-MCD pages, complete concept indexes (332 learned concepts), saved top-four ranks, separate prototype/random sections, exact example IDs, original/overlay/selected-region views, and3600×2200 PNG exports. No new selections:234 existing top/random validation views are reused. Complements and original train/validation/initial/final detail pages remain separate. Missing fourth concepts are explicitly absent.

Mask display opacity and scaling were normalized without changing source images or exact masks; image and mask hashes were checked. MCD top masks use the original>.4 display rule, whereas final random membership uses argmax; ACE validation nearest-member examples are a display extension. These differences remain disclosed. Rankings across methods are not semantic correspondences.

有限定性观察只涵盖固定Airliner和Golden面板，不覆盖所有成员。Golden HU-C10和MCD-C02都可暂称犬头/脸部；Airliner部分MCD原型与随机例覆盖差异较大，但掩码展示规则不同，不能直接定位为聚类或分配错误。背景概念不自动等于失败。全部正式人工审阅仍为未审阅。

Recommended next presentation: retain the systematic rank-based matrix, then add a separately marked provisional semantic-theme matrix and same-input comparison where cached coverage permits. These latter two views are proposals, not completed controls. Prioritize Airliner (supplement Fig5 cross-method example), Beach Wagon and Tailed Frog (paper Fig3). Author-selected examples are not automatic top-K; Golden is useful for our diagnostics but not an original Fig3 class.

## Evidence and preservation / 证据与保留

Canonical report: `artifacts/bunya/ten-class-review/report/index.html`.
Generator: `scripts/research_hub/interpretation.py`, integrated with the existing baseline report generation path.
Receipts/scripts/data: `artifacts/report-interpretation-20260924/`.
Registry: REPRO26-001 through REPRO26-005 in `docs/ANOMALIES.jsonl`; English Obsidian Implementation Notes updated, original content preserved.
Sources: archived published paper and supplement, accepted baseline analysis and exact frozen display selections. Official PDF fetching returned403 this turn; archived hashes and source-page renders are retained. No claim that author data are unavailable everywhere.
Original HU IDs, review storage schema, scientific arrays/images, failure evidence and unrelated work remain preserved. Publication requires offline browser/link checks and rejects concurrent live-report changes.
