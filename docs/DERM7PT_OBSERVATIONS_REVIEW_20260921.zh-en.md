# Derm7pt 原始观察复核 / Source-observation review — 2026-09-21

**观察成立，尚不认定为原始数据错误；人工图像审阅未完成。 / Observations confirmed, not classified as original-data errors; manual image review pending.**

- 原数据检查 / Original audit: Slurm **28734478**, commit `83972e99763ef6ecaebba3cbad6c0bf14e62e63a`. Original parser: `hpc/derm7pt_audit.py:41,55–66,89–96`; launcher checks `hpc/workstream_runtime.py:72–74`.
- 本轮独立核查 / Independent audit and renderer commit: `eb3c77506ed279c6adc6dd5891f5f1ad54dc5855`. Scripts: `scripts/audit_derm7pt_observations.py`, `scripts/build_derm7pt_observation_review.py`; exact executed SHA256 in verification.json. Two readers: guarded physical UTF8 lines/literal commas; pandas C parser, strings, no NA inference. All **19,209 original cells** match each other and the prior Slurm export.
- 本轮仅本地核查，没有新远端检查或模型 / Local evidence only; no new remote job/model. Local ZIP `/mnt/c/Users/uqcche38/Downloads/release_v0.zip`; same archive identity as verified RDM/scratch ZIP.
- ZIP: release_v0, **159112080 bytes**, SHA256 `89a4749e7e43d1c2e73876aaeb867af5b2624e929ee3d1ae7202268588566b54`.
- CSV ZIP member `release_v0/meta/meta.csv`; SHA256 `471199fb86bcc97cd1521bcc4255bb3f9c563e2ed0ad3ff836392d81a3ed9d3b`.
- Actual CSV read by Slurm: `/scratch/user/uqcche38/datasets/derm7pt/release_v0/staged/28734478/extracted/release_v0/meta/meta.csv`.
- Original RDM recovery ZIP: `/QRISdata/Q9903/datasets/derm7pt/release_v0/raw/release_v0.zip`; unchanged.

## 1. 同路径观察 / Same-path observation

9个病例中，原始derm/clinic字符串、拼接后的scratch路径与ZIP成员均相同；两种角色读取同一文件，字节和解码像素一致。**不是两个不同文件碰巧相同，更不是两个derm样本。** No distinct-path equal-byte groups occur in the prior complete inventory. No inode inspection is claimed; identity is established by identical path/member plus matching source/cached hashes.

**更正之前笼统的“原因未知” / Correction to earlier blanket unknown-cause wording:** All nine ORIGINAL `notes` cells explicitly document replacement of clinic images with derm images. Four state clinical images were missing; author paper Section III, PDF p4 reports four such cases. Five name replaced paths without stating the reason. The additional five motives remain unknown; this does not establish source-data error or paper invalidity.

All nine case_num: **565,800,824,827,833,834,839,844,851**. Raw physical lines: **566,801,825,828,834,835,840,845,852**, including header line1. Full original identifiers/fields, parsed paths and per-role byte/pixel hashes are in `nine-pairs.csv`; exact raw lines in `raw/nine-original-lines.txt`.

## 2. case_id 空值与身份 / Empty case_id and identity

`case_id` is column18 of19 and exists in every row. Missing rule: exact zero-length raw string. **984 empty,27nonempty (27distinct);0whitespace-only and0NaN/null/none/NA/N/A sentinels.** Original raw blanks, not pandas-created NaN or fallback. All1011rows and original fields are available in `all-records-index.csv`; complete984and27subsets have separate files. Examples show the first3missing and first3nonmissing rows in source order, not a random representativeness claim.

Author paper Section II, PDF p2 defines a case as a unique lesion. That defines the publication's unit, not the semantics of the CSV `case_id` column and not patient identity. `case_num` uniquely locates1011case records and filenames identify files; neither is a verified patient linkage key. `sex`/`location` are descriptive metadata, not patient identifiers. No formal `case_id` definition or sparsity explanation was found within the inspected scope. Do not infer a patient per row, group all empty IDs as one patient, or infer independence from image appearance.

Search scope: archive README and all source notes; author pinned commit `ce436877573c0a53cfa0d224bda20d9479b795aa` README, dataset.py and web_images notebook; paper; public website;14issue/PR entries including8actualissues and related issue11reply. The issue11reply is non-maintainer commentary, not author confirmation. Loader `dataset.py:109–146,336–356,388–400` uses declared modality columns and positional split indexing; no case_id linkage or missing-clinic fallback there. Evidence receipt includes full URLs, files and hashes. Not finding a field definition in this scope is not proof authors never provided one.

## 3. derm-only 影响 / Derm-only impact

**直接验证 / Verified:**1011derm records→1011different paths,1011file hashes,1011pixel hashes. Both observations do not cause an observed input duplication or label/path join error when each row contributes derm exactly once. Original mapping correctness does not certify clinical label truth or modality suitability.

**范围未冻结 / Scope not frozen:** see `scope-impact.csv`. All-diagnosis official test includes395cases and2aliases (827,834). Melanoma excluding metastasis official test includes100cases and0aliases; including metastasis101cases and1alias(827). Full-split melanoma excluding metastasis248cases includes1alias(800), including metastasis252includes3(800,824,827). These are conditional counts, not approval to select/change inputs. Source DermaMNIST fixed70test images are a different dataset.

Image-level descriptive analyses and human concept review remain possible. Model/evaluation execution still requires its own protocol authorization. Case-level inferential intervals/tests need an explicit unverified independence assumption; patient-disjoint claims, patient sample sizes and patient-level generalization uncertainty are not established. Cross-dataset/near-duplicate checks remain separate. Clinical-vs-derm independent-view comparisons must disclose the9shared views.

## 4. 私有审阅交付 / Private review deliverables

Canonical report: `artifacts/bunya/ten-class-review/report/index.html` → `workstreams/derm7pt-review/index.html`.
Source bundle: `artifacts/bunya/next-phase-20260921/derm7pt-review/bundle/`.

- Nine complete side-by-side cards using native original image bytes; clickable full-resolution images and2640×1408PNG exports. No AI clinical/image-type labels.
- Fillable human review: modality per role, assessability, obvious anomalies, evidence/example ID, suggested action, reviewer/date/status. All initiallyUNREVIEWED; local save plus JSONexport/import and CSVfallback. No automatic input changes.
- Full1011row HTMLindex and CSV;984empty/27nonempty subsets; rawCSV and selected rawlines; per-role hashes, paths and exact counting rule.
- Reproducible audit/renderer scripts, companion notebook (not executed), official source snapshots/PDF, verification/manifest. All remain private.

No deletion, path/label changes, identifier filling, deduplication, resampling or model run. No data correction requires approval from these two observations alone. Manual suitability review and earlier external-cohort/protocol choices remain open; earlier accepted datasets and failure history are preserved. MED26-006 records the updated interpretation.

## 5. 展示验收 / Presentation QA

Renderer identity fix commit `0f69919f83e2e3c310349fca195debe708c240c2`; original audit execution commit remains `eb3c77506ed279c6adc6dd5891f5f1ad54dc5855`. First report integration rejected the independent medical page because the prior gate assumed every page belonged to the ImageNet concept atlas; prior published report stayed intact. MED26-007 records the corrected manifest-guarded separate dataset identity and passed integration.

Static QA:199HTMLpages,0broken links,0external runtime assets;157concept pages,27acceptedcollections and41protected original files unchanged. Offline Chromium:18imagepanels/9cards/1011indexrows, canonical entry and review persistence/export/import PASS. All human review fields remain UNREVIEWED; QA test values cleared. No new scheduler query or Slurm job in this source-review turn; A scheduling state is the previous snapshot, not refreshed here.
