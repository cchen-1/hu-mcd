# ACE — HU-MCD released-code reference

**Authorized arm: R; implementation handover, not completed experiment results.**
本轮用户已批准完整 R 组合、seed43、batch8 和三阶段资源上限；这覆盖
`ACE_PROTOCOL_DECISION_20260910.zh-en.md` 中早先的“等待批准”状态。
S1、重新抽负样本、补 SLIC 标签、logit 梯度或统计修正版均不在本轮。

The exact display name is **ACE — HU-MCD released-code reference**. This is the
HU-MCD repository's ACE path on the fixed Golden model/inputs. It is not claimed
to recover the original ACE paper's private run, dependencies or sample IDs.

## Fixed protocol and evidence / 固定协议与依据

Implementation: `hpc/ace_reference.py`; exported `PROTOCOL` and `PROTOCOL_SHA256`.
The reviewed execution configuration must bind this protocol before features:

```text
506a52a3feb6c7042403cfd22956267e48a2eae51f9db5a0a717855146472749
```

Fresh local source reads: `run_ace.py`, `benchmark_methods.py` ACE settings,
`run_benchmark` and `iter_mask_imgs_ace`; `concept_explainer.py` image loading,
zero-row linking, cluster checks, `compute_tcav_scores` and nearest-cluster
assignment; `classes.py` SLIC, dataset preprocessing, k_means, cluster ordering,
CAV and TCAV methods; `utils/utils_ace.py` estimator/split/test;
`utils/utils_general.py` hooks and CE backward. The detailed primary-paper and
pinned public repository comparison is in
[ACE_PROTOCOL_DECISION_20260910.zh-en.md](ACE_PROTOCOL_DECISION_20260910.zh-en.md).
No scientific operator in those sources is edited.

| Choice / 项目 | Exact R execution / 实际规则 |
|---|---|
| Discovery | Existing Golden400 ordered prefix50; unchanged identity/hash |
| Gradient | Original Python shuffle over the same ordered400; first50; possible discovery overlap retained |
| Final validation | Original50 actual inputs, including the one L→RGB PNG compatibility mapping; isolated from all fitting/gradient roles |
| Random population | Curie's frozen training-only label-blind RGB-compatible2000; independent preparation Random43; no target-class exclusion or new deduplication |
| Model | ResNet50/timm0.6.13, global_pool; fixed Golden weight SHA and complete precision dictionary |
| Feature batch | 8 globally across image boundaries, final remainder per original feature call; explicitly differs from source CLI default64 |
| SLIC | Source n_segments15/50/80, sigma1, compactness50; original default label indexing, `range(max_label)` omission, area>.001 and duplicate IoU>.5 |
| Discovery/validation input | Source tight rectangular crop (`cropping_mode=1`), padding0.4588, no input masking, no feature normalization |
| Random/gradient input | Whole-image ones mask, crop0, no masking; original resize/normalization |
| K-means | Original imported `sklearn.cluster.k_means`, k25, random_state43, no n_init override |
| Structural filtering | Source rejects size<50 or image coverage<.5; equality50 and25/50 passes; max_samples50 nearest-centroid order |
| CAV | Original20 calls per eligible concept plus control; first negative50 from full effective pool, then overwrite/permutations of those50 for19 rounds |
| Estimator | Actual `SGDClassifier(alpha=.01,max_iter=1000,tol=.001)`; other installed defaults retained; split80/20, random_state42 |
| TCAV | Actual mean cross-entropy backward on logits at global_pool; score fraction `dot(normalized CAV,gradient)<0` over50 gradient images |
| Nominal test | Original two-sided `ttest_rel`; p<.01 affects concept filtering **and both evaluation directions** |
| Ranking/evaluation | Original descending mean TCAV; nearest raw centroid among all clusters, then structural/p filter; original cumulative masks and endpoint break |

Golden classifier SHA256:
`14fe96d1f9fb311a60490082d2077e6e60427dcfe21839ddf934cce948f72b0f`.
Golden dataset SHA256:
`dafeafcaa64d2379a236500d443e18f8a27520b9a0ad0288e79bdab7c683cd74`.
Original Golden run28208840, commit `eeb27e9fb31122edbfda1109c69ab7e64b094941`.
Precision remains cuDNN TF32=true, matmul TF32=false, cuDNN benchmark=false,
cuDNN deterministic=false, deterministic_algorithms=false, float32 matmul
precision=`highest`; no AMP or alternate weights.

### Defaults are part of the evidence / 未显式参数也须记录

The pinned environment is sklearn1.5.2. The inspected function signature has
`init='k-means++', n_init='auto', max_iter=300, tol=.0001, copy_x=True,
algorithm='lloyd'`. With k-means++ this environment resolves auto to one init;
older default-ten behavior must not be silently restored. The worker records
`inspect.signature(classes.k_means)`, bound defaults/supplied keys, sklearn
version, and the **actual internal estimator's** `get_params()`, `_n_init` and
`n_iter_` during its existing fit. It does not substitute KMeans for the source
function or fit a second estimator for logging. Per-run SGD `get_params()`,
iterations, t counter, coefficient/intercept/classes and RNG states are likewise
recorded. The paper's unrecorded environment/defaults remain unknown.

## Stages, limits and parent configuration / 三阶段与主线接口

| Dispatch | `config.stage` | Authorized allocation | Dependency |
|---|---|---|---|
| `ace-features` | `features` | L40S×1, 4CPU,16GiB,1h | Accepted frozen input manifest + protocol/config |
| `ace-cav` | `cav` | CPU only,4CPU,16GiB,1h | PASS features and actual manifest SHA |
| `ace-evaluate` | `evaluation` | L40S×1,4CPU,16GiB,1h | Same features + same candidate/control CAV fit |

Entry: `hpc.ace_reference.run(config, output)`. Parent owns immutable release,
Slurm dispatch, input-preparation acceptance, resource/account scheduling,
collection and final acceptance. No SSH, submission, automatic retry, negative
replacement or resource expansion exists here. Short QOS/account's two-GPU limit
is a parent scheduling constraint, not permission to request extra GPUs.
These are caps, not measured completion estimates. No SAM probe or HU/SAM cache
is used. CPU CAV stage does not instantiate a CNN.

Common config keys:

```text
execution_commit: full reviewed release SHA
stage: features | cav | evaluation
class_name: golden_retriever
model_name: resnet50
layer_name: global_pool
max_shortest_side: 300
batch_size: 8
device: cuda:0 for GPU stages; cpu for cav
precision: complete unchanged Golden precision dictionary
random_seeds: {sdc: 43, ssc: 43}
ace_protocol_sha256: exported PROTOCOL_SHA256
ace_input_manifest: final ace_inputs_manifest.json from accepted preparation
ace_input_manifest_sha256: its actual final SHA256
source_dir, dataset_manifest, dataset_manifest_sha256: unchanged Golden
resnet_checkpoint, resnet_checkpoint_sha256: unchanged Golden
golden_run_dir, golden_files_sha256
golden_launch_manifest, golden_launch_sha256
```

`golden_files_sha256` keys are `summary.json`, `run_manifest.json`,
`resolved_config.json`, `scientific/discovery.json`. Installed input-masking
identity comes from Golden's launch attestation because its scientific manifest
omits those hashes. Do not treat copied Golden SAM/clustering configuration as
ACE settings; the actual ACE settings are the bound `PROTOCOL` above.

CAV/evaluation add `feature_dir`, `feature_manifest_sha256`. Evaluation also adds
`cav_dir`, `cav_manifest_sha256`; hash finished `ace_manifest.json` files, never
RUNNING placeholders. Same worker byte hash and signatures are required across
stages. Freeze this module before submitting features; changes later need explicit
compatibility review, not automatic recomputation.

### Curie input preparation contract

Consume schema `ace-r-inputs-v1`, status PASS; verify manifest SHA, owned file
hashes and role-file hashes. `roles.target400/discovery50/validation50/random2000`
are `{path,sha256,count}` pointing to ordered JSON arrays. Their paths are relative
to the preparation output. The worker **does not** rediscover/sort source paths.
`prepared_name` is a logical prefix, not a physical renaming requirement.
Golden rows may contain extra `source_id`, `selection_index`, `original_split`
fields: compare explicit original/actual identities, not full dictionaries.
Prepared target400/validation50 are checked against the original Golden manifest.
The original Golden validation rows are kept for the existing Random signature
and attestation, avoiding irrelevant metadata differences.

Input preparation's isolated `random.Random(43)` does not consume the scientific
stream. Its skipped candidates and duplicate-content evidence stay in its own
immutable report. Different random training IDs with duplicate content are
retained; validation intersections are not permitted. Known-validation-content
checking is not a claim that the entire ImageNet validation population was hashed.

## RNG and formal sequence / 随机流与执行顺序

The approved R convention seeds **Python, NumPy and Torch once at feature-stage
initialization, with43**, after imports. Later stages restore saved states; no
per-concept/per-round seed reset. This is distinct from MCD's unchanged unseeded
eigsh default. States include Python version, complete Python/NumPy state, Torch
CPU state and saved CUDA states; CPU stage carries CUDA state without using GPU.

1. Extract discovery and random features with original ordering/batching. Complete
   validation features as independent look-ahead work, saving/restoring the formal
   RNG around that call. The independent validation loader must not advance the
   subsequent CAV stream. No predictions are reused as gradients.
2. CPU stage restores the formal feature checkpoint and runs original k-means.
   The original k-means explicitly uses local random_state43; logging consumes no
   RNG. Clustering is moved to the CPU allocation; its scientific inputs are fixed.
3. Verify the **actual post-zero-deletion random feature pool** (identity/count/order).
   This fixed configuration requires2000 effective rows; otherwise preserve
   evidence and stop before CAV rather than silently refill/shrink. All raw rows
   and zero masks remain saved.
4. Clone the current Python state; clone-only `sample(pool,50)`, then original
   shuffle of the ordered400 indices. Save control selection, full candidate order,
   gradient50, inputs, pre/post states and hashes **before any CAV fit**.
5. Formal stream calls control `random.sample`, constructs the original sorted
   control concept, fits eligible concepts then control with original methods,
   and only then performs the formal gradient shuffle. Compare IDs/state with
   the frozen plan. The actual gradient CNN runs in the next GPU stage.
6. Instrument each real NumPy selection, split and SGD call. For split IDs the
   same `train_test_split` call carries one extra passive index array; no second
   RNG draw or resplit. Instrumentation preserves weights and final RNG exactly
   in synthetic tests. Any unexpected Python consumption across CAV fitting stops
   the affected task. No conflict-driven negative substitution is allowed.

The cloned control sample order and its centroid-sorted positive fitting order
are both saved. Each concept independently starts from the full negative pool;
rounds2–20 permute its first50. Different ordering changes split identities and
SGD trajectories; these are not20 independent negative datasets. Control positives
and negatives may overlap, yielding conflicting labels; preserve and report it.

## Saved evidence / 证据与内存

Extraction streams one decoded image/its SLIC masks at a time; original flat
batch8 spans image boundaries. It releases images/datasets before moving across
roles. It does not retain discovery masks, validation masks and2000 decoded random
images simultaneously. CPU fit reconstructs feature-only segment metadata.

- `ace_config/inputs/manifest/progress/result/anomalies.json`: actual configuration,
  identity, status and owned artifact hashes. Preexisting shared runtime files are
  excluded from the immutable stage inventory because parent updates them later.
- Feature roles `discovery/`, `random/`, `validation/`: original SLIC label arrays
  for every scale, omitted-label evidence, packed masks with original float32
  reconstruction, per-image row/source/mask hashes; batch feature/logit checkpoints,
  complete raw arrays, zero mask, ordered images/rows/batches. Random uses whole
  images; it has no SLIC. `classifier.npz` records actual FC weight/bias.
- All raw feature rows and all1000 logits are finite checked; FC reconstruction
  uses rtol=atol=1e-4, with maximum error recorded. Source deletes zero discovery
  and random rows for fitting; validation zeros remain and return no matching
  concept. No raw arrays or input lists are changed.
- `kmeans_execution.json`, `clusters.npz`, `all_candidates.json`: every requested
  label0…24, including unobserved/structurally filtered labels, raw memberships,
  centroid order, coverage, inclusive boundary behavior and filter reasons.
  No p filtering is applied before these candidates are saved.
- `frozen_roles.json`, recoverable `formal_rng.json`, control planning states,
  `role_overlaps.json`: discovery/gradient/random/control-positive/validation
  intersections, both original-source identity and content-hash counts.
- `cavs/Kxx/` and `cavs/control/`: actual positive IDs/order; every real negative
  local index and full-pool row/order; combined train/test indices; hashes of
  consumed X/y; per-round NumPy/Torch/Python states before/after selection and SGD;
  estimator defaults/iterations, coefficient/intercept/classes, original held-out
  labels/predictions/accuracy. Per-round files/checkpoints preserve partial fits;
  final `rounds.json` adds positive-negative and train-test image intersections.
  Source feature caches and ordered IDs reconstruct consumed arrays exactly.
- `gradients/` batch checkpoints, `gradients.npz`, `gradient_inputs.json`: actual
  CE gradients, input global_pool features, logits, batch sizes/losses/identities.
  There is no analytic-gradient replacement or second gradient sampling.
- `tcav_statistics.npz`, `nominal_tests.json`, `all_candidates_with_tests.json`:
  every pre-p-filter concept plus control's20 scores, all20×50 gradient dot
  products, p values/statistics, mean difference/direction and filter reasons.
  Raw NPZ preserves NaN/inf; JSON uses null plus explicit finiteness/representation.
- `validation_assignments.json`: raw nearest cluster (including filtered clusters),
  structural concept index, nominal gate, zero status and precise segment ID.
  `evaluation_data/`: both original cumulative mask sequences, partial/final
  all1000 logits, correctness/offsets, counts and global prediction batches.
  Curves include visible-pixel percentages and contributor counts. No endpoint
  filling, smoothing, rule changes or extra predictions.

Files are naturally split by role/image/batch/concept/round; expected largest
feature arrays are below collector's256MiB single-file cap. Full RNG records are
written per round while fitting, then assembled, avoiding repeated rewrites of
all completed rounds. Failures preserve their precise round and exception; a
hard kill can leave only partial progress, so parent must consult final Slurm
state rather than infer success from an old progress file.

## Nominal p values, evaluation and Random / 科研边界

Original strict p<.01 is retained. A statistically lower concept TCAV score can
pass this two-sided gate; no unapproved higher-than-control condition is added.
Degenerate identical score vectors can produce NaN p, which fails the original
comparison; signed infinite t with finite p is retained in raw output. These are
reported source behaviors, not silently repaired exceptions. Nonfinite/zero CAV
weights or nonfinite active gradients/scores stop the affected task.

The nominal p gate influences **both** C-Insertion and C-Deletion. Shared negative
sets across concept/control round labels are absent in R, so the output is not
claimed to be a calibrated independent-repeat significance result. Filtering,
TCAV predictive influence and human understandability are separate conclusions.
Global mean-TCAV order is not HU/MCD per-image local relevance order. Cumulative
flipping uses source masks/overlaps, includes the initial state, and omits a full
endpoint where source breaks. Aggregation retains steps with **count>.75×50**
(≥38); Golden-only curves are not the paper's ten-class500-image aggregate.

Evaluation invokes the locked read-only `mcd_reference.verify_random_reuse` with
the original Golden validation rows and full shared `random_signature`. Optional
`random_reuse={directory,contract_path,contract_sha256}` must attest existing job
28214892 sources/config/curves/both predictions/states/all100 masks. Parent derives
that attestation independently from existing B algorithm and data bytes. No new
Random predictions are performed, whether reuse is absent or mismatched; requested
mismatch fails acceptance while preserving completed ACE predictions.

Shared read-only dependencies: `evaluate_reference` hashing/JSON/mask helpers,
`predict_stream`, `runtime_precision`, `random_signature`; locked `mcd_reference`
`load`, `digest`, `make_model`, `verify_random_reuse`. Their byte hashes are part of
ACE's signature. No MCD, shared runtime or input-preparation files are edited by
this implementation task.

## Local validation / 本地验证

Tests use synthetic local CPU data, small CAVs and a toy differentiable image
model, not a pretrained CNN or Bunya experiment. They verify original versus
instrumented CAV coefficients/accuracy/all RNG, no retry on failure, actual
k_means defaults, SLIC masks/max-label omission, streamed flat feature batches,
original CE gradients including the final batch2, clone/formal role agreement,
all-candidate/control persistence, Curie metadata compatibility, and nominal
negative-direction/NaN filtering. Structural-filtered candidates remain visible.
Local provenance-fixture tests mock remote package-version observations; actual
Slurm execution checks the pinned installed versions rather than trusting mocks.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
/home/chen/miniforge3/envs/humcd-upstream/bin/python \
-m unittest hpc.tests.test_ace_reference hpc.tests.test_evaluation -v
```

Observed local dependency warnings: NumPy `np.bool8` deprecation in pinned
scikit-image, and Matplotlib selecting a temporary cache because the default
config directory is not writable. They do not indicate a measured scientific
array mismatch. Local test PASS does not substitute for real stage acceptance.

### Final local handover — 2026-09-10 / 最终本地交接

```text
Ran 22 tests in 5.842s
OK
```

This is the completed ACE + shared evaluation suite; the parent subsequently runs
its integrated runtime suite separately, without a duplicate run here.
Worker SHA256:
`8966d8799a6bc7bf6466c4a0694a2c42a91b7cbe14e347c1ebb7107672018f2c`.

Accepted input job28227379 manifest:
`/scratch/user/uqcche38/hu-mcd/outputs/workstreams/28227379/ace_inputs_manifest.json`
SHA256:
`b95a6aebbc27366f7cc79f6d76289024877706170bc04d6055b17031804aa145`.
The locally collected actual manifest passed this worker's complete file/role
hashes and ordered Golden/original/actual identity checks. Strict runtime version
verification then correctly rejected the local CPU wheels: torch2.2.2+cpu and
torchvision0.17.2+cpu versus Golden/Bunya2.2.2+cu121 and0.17.2+cu121. Other checked
versions matched. This is a **metadata-contract pass, not a local CUDA execution
pass**; no production guard was bypassed or weakened, and no original remote
image was read. Parent input acceptance is recorded separately in
`artifacts/bunya/evaluation-baselines-20260910/ACE-R-inputs.acceptance.json`.

No SSH, Slurm submission, commit, MCD modification or real pretrained-model
experiment was performed by this implementation task. The locked MCD module SHA
remains `fb2bc64b2307d2a16d64df2ef114a4d4e9b9d415781d9ebec48987a90414795f`.
