# Research report restructuring — 2026-09-23

This is a presentation and local-evidence update, not a new experiment. The live entry remains `artifacts/bunya/ten-class-review/report/index.html`; the independent previous report is preserved under `artifacts/report-redesign-20260923/source-report/`.

## Actual evidence / 实际证据

- Ten HU-MCD discoveries and ten-class HU-MCD + class-specific Random bidirectional evaluations are accepted. MCD and ACE baselines are accepted for Golden Retriever only.
- Segmentation sensitivity: six classes have five accepted conditions; four have identity and 1-pixel erosion only. Later failed/missing conditions are not zero or accepted.
- Medical: classifier, source discovery (one retained concept), and frozen external transfer to 101 cases are complete with disclosed research limitations. Earlier preparation-only statements are historical.
- 本轮未新增模型计算、预测、样本选择、阈值变化或重提作业。当前状态来自本地验收记录，不是再次查询远端 Slurm。

## Terminology / 术语

SSC means **Sparse Subspace Clustering** (稀疏子空间聚类). The lowercase `ssc` flag in `benchmark_methods.py` denotes the insertion direction and must not be confused with the clustering algorithm. The report now has a glossary for methods, masks, subspaces, evaluation metrics, precision and medical labels. Completeness and global importance are not semantic quality scores.

## AI inspection / AI 观察

99 missing entries were completed after viewing 25 fixed-example sheets. Combined with 58 existing entries, all 157 HU-MCD entries (147 learned concepts and 10 complements) have provisional AI observations. Each new entry uses the first two saved initial random members, two final-training random members, two validation random members and top1 per split: up to eight displayed segment examples, potentially overlapping identities. No resampling or inference. Their IDs and region/overlay hashes are in `ai-inspection-completion.json`. Human status is unchanged.

Examples warranting review: Siamese C06 has eye prototypes but broad initial/training random content; police-van C11 provides a relatively coherent wheel contrast; container-ship C10 has water-background consistency; container-ship C12 differs in sampled initial/final breadth; ox C09 mixes text/watermark-like regions with other objects. These are sample observations, not population purity estimates or established assignment errors.

## Human validation / 人工校验

1. Concept descriptions, mixtures, prototype/random support and train/validation consistency: semantics have no numerical ground truth. Record exact examples and scope.
2. Medical similarity candidates: 214 pairs involving 20 external cases need identity review; matching hashes and perceptual similarity have different meanings.
3. Medical concept/sign interpretation and localization: domain expertise and spatial evidence are needed; image-level association is insufficient.
4. Grayscale scene-selection risk: prespecify categories and samples before claiming bias.

Current forms are exploratory; do not compute a quality rate before defining denominator, scope, criteria and disagreement handling. They do not reproduce the paper human study.

## Paper-comparison gaps / 论文实验缺口

- Main §4 / Supplement Table 2: ten-class discovery counts and completeness available; paper completeness is rounded to two decimals. Exact author lists/seeds were not found in the inspected materials, so inputs are not claimed identical.
- Main Fig. 3: complete local concept views exist, but author selections are not automatic top-K and same IDs do not imply semantic matching.
- Main §4.2 / Fig. 4: nine-class MCD/ACE stages are now submitted (63 receipts, commit `1e094cd7fa9f3e6990e6aad1294e85b96b82905e`) but uncollected/unaccepted. No full four-method 500-image comparison yet.
- Main §4.1 / Table 1: no formal human study; task-prediction accuracy, recognizability, description similarities and inferential comparisons remain absent. AI descriptions cannot replace responses from the paper's 41 participants.
- Supplement Figs. 6–7: grey-background and tight-crop-resize input ablations remain absent. Erosion/dilation sensitivity is a separate question.
- Medical transfer and boundary sensitivity are extensions, not missing requirements of the original paper.

## Rebuild / 生成

1. `scripts/build_research_hub.py` with the existing `humcd-upstream` Python writes a staging report. It verifies acceptance cells and new observation hashes, copies immutable evidence, generates English-only figures and bilingual pages.
2. `scripts/research_hub/adapt_legacy.py STAGE` with Python/lxml adapts saved supporting pages; `legacy_translations.py` contains explicit translations. It reads originals from the preserved source snapshot, not already transformed DOM.
3. `scripts/research_hub/validate_report.py STAGE` checks every local link and anchor.
4. Local offline Chrome checks language switching, review schema compatibility, filtering, layouts and image loading. Publish only after validation, preserving the previous report.

Scientific image assets are reused unchanged. New interpretation and glossary pages are not replacements for original logs, manifests, configurations or arrays. Raw JSON/CSV/source excerpts retain original field names and language. The old schema-1 review payload, dataset key and concept IDs are preserved.

## Concurrent status refresh / 并行状态更新

During final preservation checks, four HPC files changed through the independently authorized nine-class baseline expansion. The report task did not edit them. Actual `batch.json` and63submission receipts are now incorporated as submitted, unaccepted work. This batch no longer needs new execution approval; collection follows the user completion notice. Scientific accepted-result bindings remain unchanged. New files added to the live report during this work are preserved in the staged bundle and the pre-publication backup.
