# First medical HU-MCD discovery / 首次医学概念发现 — 2026-09-22

## Authorization / 本次批准

Explicit user approval: D0 + D1 + deployment, explaining the accepted epoch10classifier; no classifier improvement/refit, tuning, repeated SAM probes, external inference, retries or expansion. Main plan `NEXT_PHASE_MAIN_CHAT_PLAN_20260921.zh.md` retains all unsuperseded rules. Prior result and failure evidence are retained. The latest actual jobs/commit are in `artifacts/bunya/medical-discovery-20260922/handover.json`; absence means not yet submitted.

用户明确批准的是固定模型的首次源域发现。D0清单预计算不是远端图片验收，D1提交不是适配通过或结果验收。全部远端文件检查与计算均由Slurm执行；现有SSHmaster仅用于调度/传输。

## Corrected ambiguities / 已核实并纠正的说明 — MED26-015

1. **`outlier_percentile=1.0` is quantile q=1, NOT remove1%.** `utils/utils_mcd.py:get_outlier_mask` compares row-L1 strictly greater than `np.quantile(W_l1,q)`; `classes.py:load_or_comp_sparse_repr_matrix` only refits when q<1. For finite rows q1 removes none. Both the accepted HU reference config and executed discovery use1.0. The previous “1% outlier rule” was an assistant documentation error. We retain1.0; **do not change to0.99**. A targeted sparse-matrix check distinguishes these cases.
2. **No HU cluster-number search.** With `n_clusters=None`, `ClusterSpaceClass._approximate_num_clusters` uses `int(mean+std)` of nonzero retained-segment counts for images represented in the segment list. Keep the released heuristic; do not import MCD's3..19completeness search. Previous “automatic cluster search” wording was inaccurate.
3. SAM's postprocessing `min_area_ration=.01` **is a separate1% image-area rule** in `ImageClass._segment_sam`/`get_unique_concept_masks`; it is not an SSC outlier fraction. Keep SAM ViT-H points32/minregion256 and the original mask composition/postprocessing unchanged. Native224 input is not upscaled to300; the original resize300 is an upper bound.
4. Exact-zero features are excluded from clustering/final assignments by the released HU discovery logic, **not images from the frozen400/70**. Preserve every raw mask, effective mask, feature/logit row and zero/empty reason before exclusion, with stable segment IDs. Do not import benchmark all-zero-argmax behavior into discovery or hide this distinction.
5. PCA dimension retains released `ratio`/`alphaRatio=.8` behavior and per-concept `2048//K`cap, no fixed dimension or claim that the parameter establishes an optimal medical representation. Keep original kmeansseed43 and PCAseed42.

These are documentation clarifications plus a medical input/head adapter; **no scientific filtering rule has changed**. The unchanged shared scientific files are individually hash-bound in both approved configs. No optimality search is needed to execute this approved first case study.

## Frozen input and model identities / 固定输入与模型

- Classifier: accepted medical ResNet50, epoch10,7outputs, MEL index4, bias retained. SHA256 `07a6f65dbb23eb7c97510bef09c4b624a1a86007b27a64e324151fd1dfd2a655`; former classifier job28781915. No ImageNet class-ID lookup, no pretrained reset/download in D1.
- SAM ViT-H SHA256 `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`.
- D0 starts from accepted DermaMNIST-C10015-row source manifest. Train MEL1021images/533lesions; lexical sorted lesion IDs; independent `np.random.RandomState(43).choice(...,400,replace=False)`; in selected order choose one uniform index from each image-ID-sorted lesion using the same independent RNG. Preserve all70test MEL images/61lesions in original array order. Do not use classifier accuracy, image appearance or H candidates to select.
- Frozen metadata selection SHA256: `003c122b14f0ebad5eb1aab7a17b027ae8d784d242c6aff9e911930d32cd209a`. `planned-selection.json` has complete roles and candidate lesion order. D0 independently reproduces this selection from the accepted remote manifest, binds NPZ hash, checks selected pixel identity and lossless PNG serialization, then writes actual path/file/pixel hashes. Source rows/labels and original archives stay unchanged.
- Model inference: batch8,global_pool2048,FP32/noAMP; cuDNN TF32on, matmulTF32off; benchmark/deterministic false, float32 precision highest. Native224 classifier normalization equals HU tensor construction. No random augmentation at discovery inference. Samplingseed43, discovery stage RNGs43 and trainingseed43 are distinct roles.
- Region features: crop0/full image,masking_mode−1,conditional erosion when raw area>25%,released14pixel morphology; L2 for SSC only. Record raw and effective masks; images are never cropped for model features. Retainedmincluster50,mincoverage0,maxsamplesNone.

## New-path gates and outputs / 新路径门槛与中间结果

Local synthetic tests cover input-tensor equality across all256channel values, native geometry and mask ordering/erosion, q1 versusq.99, lesion-level independent RNG, released assigned-only/unique-image prototypes, authorization and afterokdependencies. No medical model inference or SAM is run locally.

D1first checks deployment/source/weight/D0 identities and exact actual HU-vs-classifier tensor equality, then reuses the approved same-batch ordinary ResNet50/full-one-mask/head algebra check on **first8fixed discovery training rows**. The new selection is explicitly recorded; it is not mislabeled classifier validation. Only after PASS does the complete470-image discovery path proceed. This is a gate inside one full job, not a separate SAM probe.

Save stages, all raw SAM masks and quality metadata, composed HU masks, post-erosion effective masks, raw masked features/logits including zeros, full-image features/logits, initial SSC labels and row L1/outlier flags, cluster filter decisions, centroid-sorted and actual diverse PCA member order, bases including complement, scores/completeness, final train/held-out assignments, and feature/logit reconstruction evidence. Distinguish global importance from per-region similarity.

Concept IDs `HUMCD-MEL-C001…`; complement `HUMCD-MEL-COMPLEMENT` is separate. Segment IDs include original image ID and raw mask index. Top prototypes call the original `get_top_concept_segms`: final assigned members ranked by score, up to10distinct images. A separate recorded presentation RNG4301 samples up to10final members per concept; this never changes the training list, fit or assignments. Images/masks and example indexes support later native-size visual review; they are not medical labels.

If zero concepts survive, numerical/identity gates fail or resources exhaust, preserve available evidence and stop; no rescue settings, automatic rerun or expansion. Zero concepts are not silently represented as successful complete discovery. No sensitivity comparison is executed without result-based evidence and later authorization.

## Resource caps and dependencies / 上限与依赖

| Stage | Cap | Dependency |
|---|---|---|
| Deployment | 1CPU / 1GiB / 5min / no GPU | pushed full commit; all remote preparation within Slurm |
| D0 | 1CPU / 4GiB / 10min / no GPU /0.5GiB output | afterok deployment; exact selection + source identity |
| D1 | 1L40S / 8CPU /32GiB /2h /15GiB outputs+new caches | afterok deployment AND D0; medical adapter passes inside job before SAM |

Existing classifier/SAM/data reused; **no existing medical SAM/features/SSC cache exists to reuse**. ImageNet HU caches are not interchangeable with these inputs/model. New caches remain isolated per Slurm job and signature. Python work ceilings540s/6900s leave room for final evidence; scheduler caps remain10min/2h. One output root per job, with existing collection machinery and immutable artifact hashes.

Engineering acceptance will separately check masks/features/cluster mapping/assignments/reconstruction. Interpretation must separate the known classifier errors, segmentation limitations and discovery behavior. No clinical-use or patient-independent conclusions; R101/S remain protocol-only and external inference is not included.
