# HU-MCD reproduction audit — 8 September 2026

## Decision

The core discovery path works on the inspected smoke inputs. That does not
establish that the implementation is correct in every path, that the only
limitation is sample size, or that paper results have been reproduced.

Prior cache checks reproduced the saved assignments and found close CPU/GPU
features. They support weak held-out concept coverage rather than a
validation-specific failure. See [the diagnosis](VALIDATION_DIAGNOSIS.md).

The next milestone should be a one-class reference run with faithful inputs,
an auditable configuration, local relevance checks and a working evaluation
path. A larger Imagewoof smoke run alone is a sample-size diagnostic.

## Sources and pinned reference

I retrieved and read the final [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Grobrugge_Towards_Human-Understandable_Multi-Dimensional_Concept_Discovery_CVPR_2025_paper.pdf)
and its [supplement](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Grobrugge_Towards_Human-Understandable_Multi-Dimensional_CVPR_2025_supplemental.pdf).
The arXiv HTML was consulted while obtaining the final PDFs; the final PDFs
were then checked directly.

The live [upstream commit API](https://api.github.com/repos/grobruegge/hu-mcd/commits/main)
returned `168eb5bf1717ab46d1c84be86bd81f0c51861413`, matching the local upstream
reference. The core classes, concept explainer, mathematical/masking utilities,
ACE/MCD runners and benchmark file have no local differences from that revision.
Our wrappers, tracking and prototype rendering contain local additions.

Downloaded PDF SHA-256:

- Paper: `70843ef628f964b841a5b166021055b5ba83d85a2f332d17f84ee935c10c2f6a`
- Supplement: `95cd743caa0f8f5e42700fe5a8c36c62d7b0d84937e458f531c63c572367fe00`

## Reference specification

The paper uses ImageNet1k, 400 randomly sampled training images per class,
ten named classes, timm ResNet50 and SAM ViT-H. Faithfulness evaluation covers
50 validation images per class (500 total); understandability is evaluated in
a human study. See Sections 4–4.2 of the paper.

The released [HU-MCD runner](https://github.com/grobruegge/hu-mcd/blob/168eb5bf1717ab46d1c84be86bd81f0c51861413/run_humcd.py)
and [concept explainer](https://github.com/grobruegge/hu-mcd/blob/168eb5bf1717ab46d1c84be86bd81f0c51861413/concept_explainer.py)
use global-pool features, original-scale layer masking, erosion threshold 0.25,
32 SAM prompt points per side, minimum retained cluster size 50, normalized
SSC inputs and automatic PCA dimensionality. The code retains PCA components
using an 80% centered-variance heuristic. Prototype defaults are ten concepts
and ten examples per concept.

Supplement Table 2 reports golden retriever with **13 clusters and completeness
0.67**. Our Bunya smoke summary has seven retained concepts and completeness
0.20266. These are comparison references, not a reason to force the clustering
parameter to 13. Record both initial cluster count and retained concept count.

The paper's exact sampled image IDs, a full historical environment lock and
human-study response data are not supplied in the checked upstream tree.
Exact numerical identity is therefore not an assured reproduction criterion.

## Current pipeline versus remaining work

| Area | Current evidence | Missing or different |
|---|---|---|
| Dataset | Working Imagewoof2-320 golden-retriever subset; local train/validation files do not overlap | Canonical ImageNet training/validation inputs and manifests for the target classes. Imagewoof's resized/repartitioned subset is not the paper's original split protocol. Current preparation takes first sorted files. |
| SAM | ViT-B/16 points works on Bunya; checkpoint hash recorded | ViT-H checkpoint and 32-point configuration; resource profiling. Only ViT-B is present in the inspected local model folder. |
| Discovery | Masked activations, SSC, PCA and complement execute; original core code retained | Reference-scale sample support and min-size 50; record clustering inputs, labels, dimensions and memberships. |
| Model identity | Existing packaging uses resnet50_a1_0-14fe96d1.pth; timm is pinned locally | Record the loaded classifier-weight hash, preprocessing and installed masking-library hashes per run. The paper does not uniquely pin a historical timm checkpoint version. |
| Prototype evaluation | Training/validation sheets and assignment proxies work | Explicit complement labeling; faithful prototype selection; per-segment raw scores. Assignment coverage is a diagnostic, not a paper outcome metric. |
| Local explanations | concept_relevances exists upstream | Not exercised by run_smoke.py. Validate contribution reconstruction with classifier bias handled separately; save local relevance and produce reviewable maps. |
| Faithfulness | benchmark_methods.py exists upstream | Has not been validated end-to-end in our workflow. Needs controlled C-Deletion/C-Insertion runs, masks, curves, contributor counts and comparison exports. |
| Baselines | run_ace.py, run_mcd.py and random benchmark present | Unverified in our environment. ACE also needs its random negative-image pool. Baseline settings differ intentionally; do not impose HU-MCD settings on them. |
| Input masking ablations | Multiple underlying masking modes are implemented | Supplement's original-scale inpainting and cropped inpainting comparisons lack dedicated settings in the benchmark dispatcher. |
| Human evaluation | No study-generation/response-analysis pipeline in the checked upstream tree | Separate scope if reproducing human-understandability claims. Visual inspection or an AI rating is not a replication of the human study. |
| Run identity | Local RunTracker writes job-specific outputs/cache, config, Git/source/input hashes and progress; collector supports --run-layout | These are local changes, not evidence of a new remotely executed tracked run. No persisted fitted bases or full software/weight archive yet. |
| Git and Slurm | Git release preparation is documented as live-tested; compute-node guards exist | phase3_smoke.sbatch still hardcodes the old code path. Connect an exact prepared release to the launcher and collector. Pin joblib/BLAS workers to allocated CPUs. |
| Durable evidence | Local result snapshots exist and preserve old outputs | An explicit archive/backup policy for final scientific artifacts; scratch is not the archive. |

The current source retains methods; most missing work is integration,
instrumentation and evaluation, rather than rewriting the discovery algorithm.

## Paper/code ambiguities and checked benchmark behavior

These items must be recorded in a protocol. They do not establish the cause of
the existing smoke result.

1. **Cluster-count heuristic:** paper/supplement describe an average segment
   count; classes.py:334 actually uses int(mean + standard deviation).
2. **Erosion:** the paper refers to the first convolution's kernel size (7 for
   ResNet50). ConceptDatasetClass calls shrink_mask with its default 14-pixel
   square. Image resizing also affects the interpretation of that width.
3. **Minimum size:** supplement prose says more than 50 members, while the
   code keeps size 50 because it rejects only sizes below the threshold.
4. **75% benchmark rule:** benchmark_methods.py:406 averages the k-th removal/
   insertion step only when more than 75% of images have that step. It is not a
   test that a single globally named concept occurs in 75% of images. Exactly
   75% is excluded. The paper's prose is less precise here.
5. **Endpoints:** the HU-MCD benchmark breaks before appending a state that
   completely removes or restores the image. A local synthetic two-concept
   check produced mask areas [4, 2] for deletion and [0, 2] for insertion;
   the terminal 0/4 areas were absent. The test replaced dilation with the
   identity to isolate this control flow.
6. **Prototype count:** run_humcd.save_concepts includes the complement in the
   plot count. An empty or complement-only sheet does not show that learned
   activations were all zero.
7. **MCD variants:** the standalone MCD runner begins its cluster search at 5,
   whereas the benchmark starts at 3. Keep the intended protocol-specific
   difference explicit.

Also tested the real aggregation helper with 100 synthetic image trajectories:
a step present in 75 was excluded, while a step present in 76 was included.

**Recommended operational reference:** preserve the pinned released code for
the first reference reproduction and document these behaviors. Implement
paper-interpretation alternatives as separately named variants with narrow
tests. Silently changing these settings would make comparison harder.

Further evaluation checks should cover zero-feature segments (the benchmark
keeps them while the smoke runner removes them), all-zero/all-one masks,
positive/negative relevance, finite outputs, pixel-axis direction, empty
concept sets and per-step sample counts. There is no evidence yet that every
one of these cases fails; they are unverified paths.

## Execution order

### 1. Finish the reproducible experiment contract

Adapt a launcher to use a prepared commit-specific release. Provide a resolved
config and unique output directory. Extend saved artifacts with bases, cluster
labels, initial/retained counts, mask-to-feature row mappings, raw activations,
local relevance and residuals. Record model/environment identity.

Test this wiring without a new training or discovery experiment where possible.
Do not replace or overwrite existing results.

### 2. Prepare the reference inputs and narrow correctness checks

Stage authorized canonical ImageNet inputs, exact split manifests, SAM ViT-H,
the pinned ResNet weights and a frozen preprocessing specification. Use a
deterministic randomized subset selection recorded in the manifest.

Before expensive runs, verify:
- sum of concept feature projections plus complement reconstructs each feature;
- local relevance sums to the linear class score; add classifier bias to compare
  against stored logits;
- learned/complement geometry and finite zero-mask handling;
- prototype row-to-image correspondence;
- benchmark ordering, endpoints, pixel fractions and sample support.

### 3. Run one reference class when authorized

Use golden retriever first: 400 training / 50 validation images and released
settings. Run HU-MCD discovery, prototypes, completeness and its two
faithfulness curves. Keep raw outputs and compare with the supplement's
reference while investigating discrepancies rather than tuning to the number.

A smaller resource probe can precede it, but must remain labeled a probe.
The 10-image/ViT-B runtime is not a reliable linear predictor: ViT-H changes GPU
cost and SSC builds a dense N-by-N coherence matrix over segments.

### 4. Expand scientific coverage

Once the one-class pipeline is understood, process the other nine classes,
run ACE/MCD/random comparisons, then the two masking ablations if reproducing
the supplement. Aggregate per-image evidence into the reported plots.

Treat the human-study replication as a separate milestone. If only computational
results are reproduced, state that scope explicitly.

### 5. Reimplement against a reference

Keep the original implementation as a frozen oracle. For an independent
implementation, use identical masks/features to compare each stage before
comparing full runs. That separates implementation differences from stochastic
sampling and segmentation differences.

Do not start by rewriting SSC/PCA or suppressing the complement to improve
appearance. A training-size learning curve remains useful for diagnosing the
small-run result, but serves a different purpose from the faithful reference run.

## Scope of this audit

No experiment or HPC job was submitted. No scientific source was changed.
Only this audit document was added. Existing outputs and hpc/check_job.sh were
verified unchanged during the audit.

The final conclusion is: **engineering smoke validation is complete for the
tested path; computational paper reproduction and independent reimplementation
validation are still incomplete.**
