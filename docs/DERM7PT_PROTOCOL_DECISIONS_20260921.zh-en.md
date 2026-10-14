# Derm7pt external evaluation: decisions pending / 外部评价待决策

2026-09-21. **PROPOSED, NOT APPROVED.** Metadata preparation only; no cohort frozen, medical model run, automatic exclusion or label overwrite. 本文不构成医学实验授权。

## Accepted preparation / 已完成准备

- User accepts MED26-006: the nine same-case references and 984 empty `case_id` fields are verified observations, not established original-data errors. Current wording is **用 derm 图替代 clinic 图 / replacement of clinic images with derm images**. The nine manual reviews remain user-owned. No repeated identity-field check without new evidence; no sample/label/path/identifier change.
- New local exact NPZ in Downloads matches Zenodo12739457 MD5 `84920fb70c83b234c295b6f0d4ae2bc0`, SHA256 `f8d80f4fac5a0a1f53256e0c337456a8ab8020828a94a9eb8a1eb8ef5d748486`. Reused through Slurm28738133, code `822860db132f35a540b079b9bef87d7c156ce8d8`, COMPLETED0:0/37s; CPU only, no new HTTP request. All23 collected artifact hashes and the independently transferred sealed15-file RDM receipt agree. Source splits8215/573/1227; train melanoma1021images/533lesions, test70images/61lesions. Existing OneDrive archive preserved.
- Accepted paths: `artifacts/bunya/next-phase-20260921/medical-decisions/accepted-dataset-paths.json`. RDM version: `/QRISdata/Q9903/datasets/dermamnist-c/zenodo-12739457-md5-84920fb70c83b234c295b6f0d4ae2bc0`; experiment copy: `/scratch/user/uqcche38/datasets/dermamnist-c/zenodo-12739457/work/28738133`.
- Two unclosed-CSV-reader ResourceWarnings occurred after full reads (worker lines217–218). Recorded as engineering warnings; scoped context-manager fix does not change rows or scientific checks. No rerun. Prior HTTP504 and launcher/access failures remain in history.
- A sensitivity jobs remain at the latest pending snapshot; no duplicate submission or model-job changes. No automatic waiting for A.

## Existing overlap evidence / 已有重叠证据的边界

| Evidence / 证据 | What is established / 已证实 | What it does not establish / 未证实 |
|---|---|---|
| Exact DermaMNIST-C audit28738133; author18confirmed pairs | 0 known pairs crossing its current splits;0cross-split identical-pixel groups;2identical-pixel groups wholly in train. No exclusions. / 当前内部划分检查通过，训练内部重复保留 | No comparison to Derm7pt; no new near-duplicate search; no patient-disjointness / 不证明外部独立 |
| Accepted Derm7pt audit28734478 |1011distinct derm inputs,9within-case cross-role aliases;0internal cross-split exact-pixel groups | Cross-role aliases are not repeated derm samples; not a cross-dataset check |
| Correction authors' paper and pinned duplicate list | HAM10000/DermaMNIST internal correction, not a published Derm7pt crosswalk |18internal pairs cannot be reused as a cross-dataset exclusion list |
| SelfClean paper and author repository | Reports9near-duplicate issues among2022Derm7pt role images,1irrelevant sample,2label errors under their expert protocol | Within-dataset findings, not DermaMNIST–Derm7pt overlap. Their9have NOT been identity-matched to our9; do not equate counts or apply their removals |
| Repository/local collection search | No accepted DermaMNIST-C↔Derm7pt pair-level overlap manifest found in inspected project evidence | Limited search, not proof that authors never supplied one or that datasets are disjoint |

Sources: [correction paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787307/), [pinned author duplicate confirmation](https://github.com/kakumarabhishek/Corrected-Skin-Image-Datasets/tree/ac655ac5a5b93264c57d69d6d520778d915bfc89/DermaMNIST/HAM10000_DuplicateConfirmation), [SelfClean paper, §4.2/Table1](https://proceedings.mlr.press/v225/groger23a.html), [SelfClean released evidence](https://github.com/Digital-Dermatology/SelfClean-Revised-Benchmarks).

Search scope: accepted local medical preparation/review/collection manifests and overlap-related filenames; correction paper, author analysis README, duplicate-confirmation README/list and NPZ construction notebook at `ac655ac5a5b93264c57d69d6d520778d915bfc89`; Derm7pt original paper/loader already pinned in the source review; targeted public searches for DermaMNIST/HAM10000–Derm7pt overlap and primary SelfClean references. Source copies/checksums are in `medical-decisions/sources`. No external revised cohort has been adopted.

Source filenames use different namespaces; absence of a literal ID intersection would not prove independence. NPZ authors resized original images bicubically to224×224 (`CreateNPZfromCSV.ipynb`; extracted `create-npz.py.txt:36–42`). Derm7pt originals have different geometry. Raw pixel-hash comparison without harmonization can miss the same underlying photograph. A perceptual match is only a candidate; a negative search also cannot establish patient independence.

## Decision D1: external scope / 外部范围

Counts below are metadata counts, not chosen or changed sample lists. None depends on model output or manual image review.

| Option / 选项 | Cases / derm图 | Alias cases / 同角色引用病例 | Consequence / 含义 |
|---|---:|---|---|
| **R: official test, author MEL grouping (recommended / 推荐)** | **101** |827(one melanoma metastasis) | Preserve author's diagnosis grouping and official held-out split; include in situ and primary melanomas plus1metastasis. Retain raw subtype. This is exploratory concept association on an official holdout, not a validated confirmatory clinical study. |
| P: official test, exclude melanoma metastasis |100 |none | Explicit narrower primary-melanoma scope, differs from author MEL; cannot assume source HAM melanoma has an identical subtype definition. Exclusion would require this decision, not the alias observation. |
| E: all splits, same narrower scope as P |248 |800 | Larger exploratory cohort, no claim of untouched official-test evaluation; uses88train+60valid+100test. Do not expand just to improve associations. |

Recommendation R avoids an additional unsupported diagnosis exclusion. No present evidence says case827 must be removed from derm-only use. Its manual review can qualify interpretation later; the current observation is not a reason to delete it. If P is chosen for the research question, its100cases are fixed by diagnosis and split, not image appearance. Train+valid148primary cases can remain unused; they are not automatically authorized for model development or target adaptation.

原始完整计数表 / Complete source-derived counts: `candidate-scopes.csv`, `raw-sign-counts.csv`, `proposed-contrasts.csv`. Generator: `scripts/prepare_derm7pt_protocol_decisions.py`; receipt records input/script hashes. Dataset `cases.csv` is the already accepted Slurm export; no new `case_id` audit.

## Decision D2: labels, response and statistics / 标签、响应及统计

**Recommended combined protocol S (awaiting approval).** Keep raw labels and add a grouped analysis column following the author loader (`ce436877573c0a53cfa0d224bda20d9479b795aa`, `derm7pt/dataset.py`, vascular/pigmentation/regression grouping at489–508). All grouping entries are in `proposed-label-groups.json`. This is categorical grouping, not an invented numeric order.

| Sign / 征象 | Proposed groups and actual R101 counts / 分组及例数 |
|---|---|
| Pigment network / 色素网 |absent31,typical12,atypical58 |
| Blue-whitish veil / 蓝白幕 |absent53,present48 |
| Vascular structures / 血管结构 |absent67,regular18,irregular16 |
| Pigmentation / 色素沉着 |absent38,regular1,irregular62 |
| Streaks / 条纹 |absent41,regular1,irregular59 |
| Dots/globules / 点及球状结构 |absent12,regular5,irregular84 |
| Regression structures / 消退结构 |absent53,present48 |

- Main response: case-level area fraction of the **union** of effective model-input masks whose finite, nonzero features are finally assigned to that concept, divided by total model-input area. Each concept has its own union; inter-concept overlap is allowed, totals need not sum to1. OC reported separately. This measures coverage under segmentation/assignment, not pure semantic intensity, classifier importance or anatomical localization. No classifier-dependent response is silently substituted.
- Successful extraction with no assigned valid member =0 coverage, with empty/zero-feature flags retained. Failed extraction or missing input/output =missing, never0. Retain all cases in the cohort ledger and report actual valid denominators. Diagnosis misclassification is not an exclusion. All-zero benchmark argmax must not create artificial concept membership.
- Report the complete concept×label matrix: **12 fixed comparisons per learned concept**, each non-absent group against absent; other-category cases stay in the cohort but are outside that pair's denominator. Include all concepts, zero coverage and negative associations. AUC direction fixed: higher coverage predicts the named non-absent group; never replace AUC with max(AUC,1−AUC). Also show group counts and distributions/medians. No unordered0/1/2 correlation, best-pair selection, overall interpretability score or after-the-fact merging.
- Recommended first pass: **descriptive effects only, no p-values or significance claims**. Optional uncertainty (separate S+ option): ordinary case-level bootstrap2000, seed20260921, jointly resample all concept/label values for a case;95%percentile intervals. Omit intervals for contrasts with either group<10 (proposed reporting guard, not a validated scientific cutoff). Keep their counts, raw values and point estimates when both groups exist. Record degenerate draws and valid-replicate counts. These are marginal exploratory intervals conditional on the fit, not simultaneous claims, training-seed uncertainty or patient-level guarantees.
- Source internal70images are a different population: any uncertainty there must resample61lesions with their images together; no patient-independence claim. Do not pool their labels or denominators with external R101.

Alternative S-raw: retain all raw categories for descriptive distributions only, defer grouped pairwise AUC. This makes fewer statistical choices but fragments already sparse groups. Alternative clinical7binary contrasts would change the target and denominators; not silently substituted. If formal multiple-testing inference is wanted, specify a separate testing family and dependence assumptions before results, rather than turning nominal intervals into discoveries.

**Recommendation: R+S** now (official101, author categorical groups, coverage and full descriptive matrix, no inferential claims). S+ is optional and is NOT implied. This freezes only a future evaluation specification; source-model, preprocessing, discovery and GPU budgets still require a separate approved plan.

## Decision D3: necessary overlap checks and resource budget / 必要检查与预算

No new audit job submitted yet. Use accepted scratch NPZ and Derm7pt, never dense RDM reading. No model, training, SAM, embeddings, relabeling or automatic removal.

| Option | Scope / 范围 | Hard cap / 上限 | Evidence gained / 边界 |
|---|---|---|---|
| **H: recommended data-only audit** |10015source rows ×1011derm images across all source splits and diagnoses; preserve role/split/ID hashes. Compare current decoded pixels and bicubic224harmonized pixels. Generate64-bit pHash and64-bit dHash candidates;8rotations/reflections of Derm7pt only, each representation explicit; candidate if either Hamming distance≤6. Also save nearest candidate per derm image under each hash, ties deterministic. Export pair IDs/distances and review indexes. |**4CPU,8GiB,30min,0GPU**; ≤2GiB derived output. CPU upper bound2core-hours; elapsed estimate5–15min, not measured for this path. No retry/extension. |10,125,165cross-dataset pairs;81,001,320orientation comparisons per hash, processed in blocks. Exact matching is representation-specific; perceptual candidates require review. No-match ≠ independence. |
| H0: smaller exact-only check |Same complete populations; decoded/harmonized pixel hashes and metadata mapping only |1CPU,8GiB,10min,0GPU;≤0.5GiB output |Cannot address recompression/crop/near duplicates; external-overlap uncertainty remains materially broader. |
| Hold |Retain existing evidence boundary and defer |0 |No cross-dataset independence claim. |

H is a one-off candidate search, not a validated sensitivity guarantee. Distance6 is an operational retrieval choice, not a same-lesion classifier; crops, brightness changes and tiny lesion regions can evade it. No D4 change reaches model input. Store every qualifying pair (or explicitly stop incomplete at the output cap); a readable first review page may show at most200candidate pairs in deterministic(distance,sourceID,dermID) order, with the full manifest and undisplayed count. Nearest-only supplements are labelled separately. No embedding model or automatic result-driven threshold adjustment.

Reuse the existing source internal-known-pair and Derm7pt exact audits rather than repeating them. Check only the new comparison path (identity mapping, transform/hash determinism and tiny synthetic same-image examples). These are software checks, not evidence that the real search is exhaustive. Preserve all candidates even when labels disagree. Confirmed cross-dataset conflicts pause affected external evaluation; report IDs and options, never silently drop images. Public SelfClean evidence remains a separate within-dataset warning unless record identity is verified.

**Approval of H/H0 means data audit only. Approval of R/S means protocol only. Neither authorizes a medical model experiment.** Model identity/standardResNet50 weights or training recipe, preprocessing, necessary mapping checks, discovery/runtime and storage estimates remain concrete dependencies. The already approved400lesion seed43 sampling direction and fixed70internal test images are unchanged; no list was generated/frozen this turn.
