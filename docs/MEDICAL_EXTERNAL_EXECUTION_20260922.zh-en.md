# MED26-E224：固定医学概念的Derm7pt外部应用 / Frozen medical concept transfer

## Explicit approval / 明确批准

The user requested external evaluation after acceptance of the completed batch, then explicitly selected **“批准 E224 及所列上限（推荐）”**. This supersedes the historical prohibition on external inference only for this scope. R101 and descriptive S remain as previously approved. No classifier optimization, refitting, medical concept rediscovery, expanded cohort, bootstrap, significance tests, clinical claims or automatic retries.

用户在本批验收后要求开始外部评价，并明确批准E224及预算；该范围内替代旧“外部推理未授权”。R101/S保持原批准规则，不扩大至重训、重新发现、额外病例或显著性检验。

## Accepted dependencies / 已验收依赖

- D1 `28792059`, commit `5623abe8a843deaea7ab838a8c76c41afbad565e`: 400 discovery images/distinct lesions; all70 source held-out images/61lesions. One retained concept `HUMCD-MEL-C001`, initial cluster0/807members,32dimensions. Other initial clusters31/11 filtered by unchanged minimum50. Complement is separate,2016dimensions.
- Local acceptance: `artifacts/bunya/batch-review-20260922/acceptance-both.json`; all raw/effective masks, features, IDs, zero handling, initial/PCA/final membership, prototype rules and reconstruction checked. The held-out set has exactly50/70 MEL predictions; no sample exclusion. This is engineering acceptance, not clinical adequacy or semantic validation.
- Frozen epoch10 classifier SHA256 `07a6f65dbb23eb7c97510bef09c4b624a1a86007b27a64e324151fd1dfd2a655`; SAM ViT-H SHA256 `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`.
- Accepted Derm7pt scratch data and metadata from28734478; original ZIP preserved on RDM. No intensive RDM reads, downloads or re-audit.

医学D1的完整性通过；一个概念不等于一个医学语义。训练68/917、留出14/157非空区域产生零特征，保留原数组且不作有效分配。已有DF零召回、后期过拟合、身份信息不足等限制继续有效。

## Frozen scope and execution / 固定范围与执行

1. **R101**: official test, all six author MEL diagnoses including one metastasis; original CSV-row order; all101 original records/derm paths retained. No filtering by predictions, appearance or overlap candidates. Case IDs are not patient IDs.
2. **E224**: private working PNG per original RGB image, Pillow BICUBIC warp to224×224 **before SAM**. Preserve original bytes/pixel hashes, width/height, original labels and actual-input mapping. This deliberately changes aspect ratio and fine structures; it is a controlled same-size transfer, not native-resolution segmentation evaluation. Unexpected non-RGB input is an error, not a silent conversion. Local data-only preflight verified all101 existing archive inputs against the accepted audit hash format (which includes mode/shape/dtype).
3. Fixed epoch10 seven-class ResNet50, MELindex4, global_pool2048, batch8, source mean/std, FP32/noAMP, cuDNN TF32on, matmul TF32off, highest precision, benchmark/deterministic false. No pretrained fallback. Frozen D1basis/FC parameters checked together.
4. Same SAM ViT-H points32/minregion256, released composition/minarea1%, maxshortestside300 upper bound (224work images are not enlarged), cropping0, masking−1, conditional erosion at rawarea>.25. Save raw SAM proposals/metadata, composed masks, effective masks, feature/logit arrays and IDs. This is new external inference, **not a probe or refit**.
5. Exact released `batch_concept_activations(...,norm_batch=False)` on nonzero region features. Keep all raw rows, invalid exact-zero assignments explicitly−1, never arbitraryC0. All images retain their place; completed images with no assigned members have actual zero coverage, missing processing is not zero.
6. **S descriptive**: per-image union of effective assigned masks/full224² area; original seven labels plus author grouping retained. All12 fixed non-absent-v-ABS contrasts for C001, complement separately. Preserve positive direction, AUC, group means/difference, actual group/missing/other-label denominators. No p-values, bootstrap, sign flip, semantic equivalence or localization claims. One-member groups remain disclosed.
7. Save full-image seven-class logits/predictions; R101 is an all-MEL cohort, so report MEL prediction fraction, not seven-class balanced accuracy/AUROC or clinical sensitivity without scope qualifications.

外部输入与原图分别保存，mask=1是选中区域。概念覆盖率与征象的图片级关联不能证明语义等同、医学定位或患者独立性。原9例人工审阅只涉及显示与初步外观，不验证跨数据集重叠。H中R101有20例/214相似候选，尚未确认重复；不自动删图，保留身份风险。

## Resources, gates and stop conditions / 资源、门槛与停止条件

| Stage | Approved ceiling | Evidence / reuse |
|---|---|---|
| Immutable deployment |1CPU/1GiB/5min/0GPU | Slurm checks full commit, worker hash, compact inputs, unchanged scientific files, environment; no inference |
| E224 inference + descriptive S |1L40S/8CPU/32GiB/30min/4GiB new output | D1 measured12m55s for470images, SAM684.4s;101image linear SAM estimate≈147s, not a guarantee.30min is a ceiling, no extension |

GPU work bound1680s leaves scheduler shutdown/manifest time. Results live in a new job-specific directory. Dataset, model, basis, protocol, acceptance receipt and full commit hashes are mandatory. Deployment may precede E224 through `afterok`, with a runtime readiness gate; no reliance on expired D0/D1 scheduler records. Reuse their accepted receipts instead. Slurm-only remote inspection/processing; user handles login/2FA.

Failure, timeout, input mismatch, unexpected nonfinite values or output cap stops the affected job and preserves evidence. No retry, expanded budget, changed precision or threshold. After submission record IDs once and stop waiting; collect only after the user's completion notice.

Exact configuration: `configs/medical/medical-external.approved.json`. Private frozen101-row payload and hashes: `artifacts/bunya/medical-external-20260922/`. Private report remains `artifacts/bunya/ten-class-review/report/index.html`.
