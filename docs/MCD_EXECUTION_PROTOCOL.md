# Golden Retriever MCD execution protocol / 执行协议

This implements the **HU-MCD repository's MCD benchmark path**, not the separate
human-study/display run. User authorization covers feature extraction, one fit,
and both evaluation directions, independently of ACE. This document is an
implementation handover, **not evidence of a completed MCD experiment**.
本轮已授权三阶段；旧权限说明不再适用。当前交付代码，不声称已完成模型实验。

## Source audit / 原始依据

Local primary sources, read before implementation:

- `artifacts/bunya/final-28208840/report/support/paper.txt`, implementation details
  around lines 470–486 and faithfulness protocol around 727–747: ResNet50/timm,
  400 training images, minimum five MCD concepts specifically for the human study;
  Figure 4 aggregates ten classes × 50 validation images.
- `artifacts/bunya/final-28208840/report/support/supplement.txt`, Table 2 and its
  preceding paragraph: counts derive from SAM and filtering. These are **HU-MCD
  counts**, not MCD reference targets. No independent MCD FO hyperparameter table
  was found there; the supplement refers to Vielhaben et al. for algorithm details.
- `benchmark_methods.py`, MCD settings, the `range(3,20)` branch,
  `iter_mask_imgs_mcd`, and `calc_avg_and_std`; `run_mcd.py`,
  `prepare_imgs_for_mcd` and `range(5,20)`; `classes.py`, SSC, spectral clustering,
  `get_segments`, and `compute_pca_basis`; `concept_explainer.py`, subspace bases,
  activations/relevance and quantification; `utils/utils_mcd.py`, the actual
  oblique, completeness and SSC functions.
- Installed `skdim.id.lPCA(ver='FO')`: default `alphaFO=.05`; counts eigenvalues
  strictly greater than `.05 * largest_eigenvalue`. Runtime checks this default
  and pins library versions to Golden's saved launch evidence.

These sources establish the **local reference implementation**. They do not
establish byte equivalence with an independently retrieved Vielhaben repository.
No external source or paper result is substituted for the code actually executed.
以上为本地论文／补充／原仓库核对；不能把单类别结果称为论文十类汇总复现。

| Item / 项目 | Executed benchmark / 本实现 | Display or caveat / 展示与限制 |
|---|---|---|
| k search | 3,4,…,19; stop only when completeness **> .5** | `run_mcd.py` starts at **5**; human study requires ≥5 |
| Exhaustion | Keep k=19, `threshold_met=false`, explicit anomaly | Source continues downstream; no forced success or extra k |
| SSC | q=.75; first full nonzero rows, then recompute on inliers | Strict `row_L1 > quantile(.75)`; ties need not remove exactly 25% |
| Features | Unmasked ResNet50 layer4, 2048×7×7 per image | No SAM or HU pooled/masked-feature reuse |
| Clustering | Spatial L2 in `prepare_imgs_for_mcd`, then original clustering L2 again | Zero spatial rows excluded only as in source |
| Concept filtering | min_size=0, min_coverage=0, max_samples=None | HU threshold 50 does not apply |
| FO/PCA | FO dimension capped by `2048 // actual_concept_count`; PCA random_state=42 | Actual basis size also limited by available PCA rows |
| Members | Source centroid sorting, then `get_segments('diverse',None)` | All selected members; actual PCA call order captured without extra calls |
| Spectral step | Original normalized affinity/Laplacian/eigsh, k-means random_state=43 | No new eigsh `v0` or global RNG seed introduced |
| Validation | Frozen 50 actual inputs/order, full spatial maps | Display loader's shuffle is not adopted |
| Evaluation | Both sdc/deletion and ssc/insertion from **one fit** | “ssc” evaluation direction is distinct from SSC clustering |

## Fixed identity and staging / 固定身份与部署

Frozen input manifest SHA256:
`dafeafcaa64d2379a236500d443e18f8a27520b9a0ad0288e79bdab7c683cd74`.
ResNet50 `resnet50_a1_0-14fe96d1.pth` SHA256:
`14fe96d1f9fb311a60490082d2077e6e60427dcfe21839ddf934cce948f72b0f`.
Golden source run `28208840`, source commit
`eeb27e9fb31122edbfda1109c69ab7e64b094941`.

Preserve seed43's existing 400 selected training images and all original 50
validation IDs. Preserve the sole L→RGB PNG compatibility mapping. The new module
compares **input_sha256** to Golden's actual input hashes; original file identities
are separately checked by `workstream_runtime.verify_inputs`. No resampling,
conversion, replacement or grayscale comparison is performed here.

- Input image processing uses original `ImageClass` (max shortest side 300),
  full-image ones mask and `ConceptDatasetClass`: cropping=0, use_masks=False,
  masking_mode=None, erosion=1. No extra crop, feature normalization or pooling
  is introduced during extraction. Frozen prepared names determine image order.
- Model is created without downloading weights and loaded strictly from the
  hashed checkpoint. Installed timm=0.6.13 mask implementations must match
  Golden launch `masking` attestations and released source bytes.
- Fixed inference batch size **8**, separately reset for train/validation and
  each evaluation direction; prediction batches may span image boundaries.
- Fixed precision: cudnn TF32=true; matmul TF32=false; cudnn benchmark=false;
  cudnn deterministic=false; deterministic algorithms=false;
  float32_matmul_precision=`highest`. Flags are checked, not silently changed.
  Model/features/logits remain float32. Mixed basis dtype and float64 oblique
  calculations follow saved/original NumPy arrays. No AMP/autocast introduced.

| Runtime mode | config.stage | Allocation | Dependency |
|---|---|---|---|
| `mcd-features` | `features` | 1 GPU, 4 CPUs, 16 GiB, 30 min | frozen inputs, weights, Golden proof |
| `mcd-fit` | `fit` | CPU only, 8 CPUs, 64 GiB, 4 h | PASS feature manifest + its actual SHA256 |
| `mcd-evaluate` | `evaluation` | 1 GPU, 4 CPUs, 16 GiB, 30 min | same PASS feature and fit manifests |

These are authorized budgets, not measured completion-time guarantees. At most
19,600 spatial training rows: first dense coherence alone is ~1.43 GiB float32;
joblib workers also need feature/dictionary copies and fitting intermediates.
LOKY workers are capped to allocation, each BLAS/OpenMP pool to one thread. No
automatic timeout retry, alternative GPU resubmission or second fit.
All stages require a Slurm compute hostname/allocation. This module has no SSH,
Slurm submission or collection code. 提特征可独立先行；拟合等待真实缓存哈希。

## Parent integration contract / 主线接口

`hpc.workstream_runtime.MODES` maps the three names above to `hpc.mcd_reference`;
entry is `run(config: dict, output: Path) -> dict`. Parent owns dispatch, immutable
release, submission, generic worker logs, collection and final acceptance. Generic
MCD dispatch must not use HU/SAM prerequisite or weight checks. MCD performs its
own input, weight, source, installed model and GPU precision checks.

Required common config keys (copy fixed settings from Golden; change layer only):

- `stage`, `execution_commit` (full reviewed release SHA), `class_name`,
  `model_name`, `layer_name='layer4'`, `max_shortest_side`, `batch_size`, `precision`,
  `random_seeds={'sdc':43,'ssc':43}`. Device `cuda:0` for GPU stages, `cpu` for fit.
- `source_dir`, `dataset_manifest`, `dataset_manifest_sha256`,
  `resnet_checkpoint`, `resnet_checkpoint_sha256`.
- `golden_run_dir`; `golden_files_sha256` with keys `summary.json`,
  `run_manifest.json`, `resolved_config.json`, `scientific/discovery.json`.
- **`golden_launch_manifest`, `golden_launch_sha256`**: Golden execution launch
  proof (separate path allowed). Required because Golden's `run_manifest.json`
  does **not** contain the installed input-masking hashes; its launch proof does.
- Fit/evaluation add `feature_dir`, `feature_manifest_sha256`; evaluation also
  adds `fit_dir`, `fit_manifest_sha256`. Hash the finished `mcd_manifest.json`.
  Do not replace these with predicted hashes or a queued job's RUNNING manifest.

Helpers imported unchanged from `hpc.evaluate_reference`:
`checked_file(path,sha,size=None)`, `under(root,relative)`, `sha256(path)`,
`write_json(path,data)`, `runtime_precision(torch)`, `save_masks(path,masks)`,
`predict_stream(model,trajectories,batch_size,progress,on_batch=None)` and
`random_signature(config,model_default_cfg,rows,source_sha256)`.
The predictor returns `(logits, global_batch_identities)`; trajectories yield
`(image_index, ImageClass)`; `on_batch` receives logits and their identities.
No changes to these helper interfaces are required.

Cache identity requires the same MCD module byte hash, method/input/model/precision
signature, PASS status, and all owned file hashes. Therefore freeze/review this
module before extraction. Later engineering changes require explicit compatibility
review; do not weaken identity checks or automatically re-extract features.
Parent launcher fields may change after return: preexisting worker-owned files
are deliberately excluded from the MCD immutable inventory.

## Scientific arrays and traces / 科学数组与检查

- Features: `classifier.npz`; `{training,validation}_batches/*.npz` durable batch
  checkpoints; `{split}_features.npz` full maps + all1000 logits; `{split}_inputs.json`
  ordered mapping, loaded image sizes, batch membership and zero spatial counts.
  Training maps are ~153 MiB uncompressed, below collector's 256 MiB single-file cap.
- Fit: normalized `clustering_input.npz`, raw↔nonzero↔image/cell mappings;
  `ssc/pass_1.npz`, `pass_2.npz`, source SSC cache, first-pass outlier mask and rule;
  `ssc_trace.json`; every attempted `search/kNN/` labels/outliers, actual PCA member
  order, FO estimates and bases; `search_trace.json` including failure and RNG states.
  Top-level `discovery.npz` is the selected unchanged fit. Global importance and
  learned-only order are in `mcd_result.json`; complement is separate from concepts.
- Each `{split}_spatial.npz`: **all** raw spatial rows, normalized-batch oblique
  scores, argmax, raw-feature local relevance, feature reconstruction error and
  reconstructed all1000 image logits. Zero training rows excluded from fitting
  remain represented here. Stored mappings identify every image/row/column.
- Full basis square/rank, condition number, finite values; pooled-map→FC check;
  all-row feature reconstruction relative error ≤1e-4, all1000 pooled logit and
  target relevance reconstruction rtol=atol=1e-4. LU is factored once, but float32
  per-image normalization occurs **before** solving as in source. First image's
  49 scores in each split are compared to literal original per-row solves at
  rtol=atol=1e-5, with observed errors recorded. Small synthetic tests separately
  compare relevance and reconstruction to literal source projections.
- Evaluation: pixel argmax maps; packed cumulative masks; local importance/order,
  absent concept IDs, input and state mappings; streamed partial logits/batches;
  complete all1000 logits, correctness, offsets, visible pixel proportions,
  contributing image counts and original mean/std curves for both directions.
- Every stage: `mcd_config.json`, `mcd_inputs.json`, `mcd_manifest.json`,
  `mcd_progress.json`, `mcd_result.json` when successful and `mcd_anomalies.json`.
  Failures preserve traceback, completed files and attempted search history.

SSC `.npz` files >128 MiB are losslessly archived **after source cache use**, on
success or failure, into 64 MiB byte chunks. `ssc_archives.json` contains original
size/hash and ordered chunk size/hashes; concatenation restores the exact original
NPZ. Arrays are not quantized. No input cache is modified. Partial failure traces
remain available; hard kill cannot guarantee finally/manifest execution, so parent
must use Slurm final state and partial collector records for incomplete stages.

## Evaluation meaning and Random / 评价含义

Use original `iter_mask_imgs_mcd` directly. Compute 49-cell per-image activations;
resize score and relevance maps to 224×224 **before** argmax; include complement
in argmax but exclude it from learned concept masks. Local ordering is descending
mean resized relevance on assigned pixels, distinct from global squared oblique
weight importance `||w_concept||²/||w||²`. Completeness is original **norm** of the
orthogonal learned-union projection of normalized class weight, not its square.

Absent concepts have original NaN mean importance and empty masks; source sorting
and skipping are preserved, with JSON null and explicit absence records. Neither
absent concepts nor complement are silently turned into learned concepts. Initial
state is full image for deletion, zero mask for insertion. Original break omits a
fully flipped endpoint; no padding, smoothing, added predictions or re-ranking.
Original aggregate uses **count > .75 * 50**, i.e. ≥38 contributors. Paper says
“at least75%”; do not silently change code. Keep original `1-visible_fraction`
plot coordinate and identify direction. Golden-only curves do not reproduce
Figure 4's 500-image aggregation or a human-understandability study.

Random is optional cache reuse only. Without attestation: `NOT_REQUESTED`, zero
new Random predictions. Optional `random_reuse` has `directory`, `contract_path`,
`contract_sha256`; the separately derived JSON contract contains:

- `status='PASS'`, `source_job='28214892'`;
- `signature` equal to the **entire** helper result `{sha256,identity}`;
- `files`: relative paths → `{sha256,bytes}`, including original evaluation config,
  sources, curves, both Random predictions/states and all 100 packed mask files.

Identity binds ordered source IDs + actual hashes, classifier, model preprocessing,
precision, batch grouping, mode seeds, grid border rule, and benchmark/classes/
utils_general/inputmask source hashes. It intentionally excludes method feature
layer. Parent derives this attestation from existing B evidence without modifying
old outputs. A missing/mismatched claim never triggers Random recomputation. With
requested mismatched attestation, MCD predictions remain saved but stage fails
acceptance; parent can review evidence rather than resubmit automatically.

## Boundaries, anomalies and local validation / 边界与本地验收

- Neither original `eigsh` nor this wrapper supplies `v0`. Process RNG states and
  library versions are saved, but they do not attest ARPACK's internal starting
  vector. No cross-process bitwise fit guarantee. Earlier draft's extra global
  seed43 convention was removed before deployment; k-means43/PCA42 remain source.
- Source full-learned-rank complement returns an uninitialized `np.empty` row;
  unexpected rectangular/rank-deficient/nonfinite bases stop the affected stage.
  No silent complement or least-squares protocol repair. Source square-map index
  convention is safe for expected7×7; other feature shapes stop for review.
- Nonfinite active scores, invalid reconstruction, changed input identity or
  unsupported model/config/version stop the affected stage; no model tweaks.
- Grayscale exclusion and validation compatibility are inherited fixed-input
  limitations. They are not evidence of MCD inferiority or human interpretability.
- Resolved predeployment engineering errors: original/actual input hash confusion;
  mutable parent files in cache manifest; missing masking hashes in Golden science
  manifest. Tests cover these concrete provenance boundaries. Historical outputs
  were not changed. Deprecation `np.bool8` and local Matplotlib temporary-cache
  warnings occurred in tests; no scientific array discrepancy was observed.

Local checks use `humcd-upstream` Python with `OPENBLAS_NUM_THREADS=2`,
`OMP_NUM_THREADS=2`, `PYTHONDONTWRITEBYTECODE=1`:

```text
python -m unittest hpc.tests.test_mcd_reference hpc.tests.test_evaluation -q
```

Tests use synthetic CPU image/model data and stub expensive SSC/PCA where checking
control flow; they do not execute a pretrained model, real MCD fit, SSH or Slurm.
They cover literal feature hook/batches, actual Golden compatibility identity,
two SSC passes plus cache reuse and strict .5 search stopping, unchanged original
bidirectional trajectories/absent concepts/endpoints, literal oblique algebra,
parent-file mutation isolation, chunk roundtrip and Random signature rejection.
Latest local validation (2026-09-10): **21/21 PASS**, 4.479 seconds, including
the parent's new missing-masking-hash regression.
Local PASS is implementation evidence; remote feature/fit/evaluation acceptance
still requires the produced job, config, arrays and final manifest.
