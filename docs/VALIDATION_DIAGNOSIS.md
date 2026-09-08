# Validation diagnosis — 8 September 2026

The saved runs show weak transfer of the discovered concept subspaces, not empty
validation features. No validation-specific implementation failure was found in
the paths checked. Tiny training clusters and a winner-takes-all comparison
against the orthogonal complement explain the observed pattern; the independent
effect of training size has not yet been measured.

## What the previous runs actually produced

Assignments below count segments whose highest activation belongs to a learned
concept rather than the orthogonal complement. They are not classification accuracy.

| Existing run | Class | Training / validation images | Learned assignments, training | Learned assignments, validation |
|---|---|---:|---:|---:|
| WSL initial smoke | garbage_truck | 3 / 2 | 3 / 23 (13.0%) | 0 / 9 (0%) |
| WSL fidelity | golden_retriever | 20 / 10 | 41 / 105 (39.0%) | 1 / 70 (1.4%) |
| Bunya job 28005092 | golden_retriever | 10 / 5 | 23 / 53 (43.4%) | 0 / 37 (0%) |

The WSL fidelity run's sole learned validation prototype is
`008_ILSVRC2012_val_00019212.jpeg`, concept index 4. Bunya used the first five
images. Applying the reconstructed WSL 20-image concept basis to those first
five WSL validation images also gives **0 / 37**. Applying that same basis to the
Bunya GPU validation features gives **0 / 37** too.

However, sample selection is not the whole explanation: the Bunya 10-image basis
also gives **0 / 70** when evaluated on all ten cached WSL validation images.
The basis itself differs. Training-image count, seed and numerical execution
differ between the two historical runs, so their causal effects cannot be
isolated from these runs alone. The initial garbage-truck run is a different
class and is not a point on a golden-retriever learning curve.

## What was verified

- Reused the existing authenticated Bunya SSH master and downloaded four named,
  existing activation/SSC cache files through SFTP. No remote file processing,
  model inference or Slurm job was run.
- All cached feature values were finite. WSL contained 109 raw training rows and
  73 validation rows; the existing zero-row filter retained 105 and 70. Bunya
  contained 54 and 40 raw rows; 53 and 37 survived. Thus the validation data did
  not all disappear during filtering.
- Reconstructed spectral clusters and PCA bases from the saved sparse matrices
  and features, without refitting sparse representations. Both runs reproduced
  their saved concept dimensions and per-concept training/validation assignment
  counts exactly.
- Checked the diagnostic projection calculation against the repository's
  full complement-basis decomposition on three validation vectors per run:
  maximum absolute activation errors were below 4e-8.
- Matched CPU/GPU feature rows within each shared image, allowing for segment
  ordering differences. Across 53 training and 37 validation segments, every
  matched cosine similarity exceeded 0.9998. Mean relative L2 differences were
  about 0.67% and 0.68%, respectively. This is strong evidence against a gross
  GPU feature-extraction mismatch, not a claim of bitwise reproducibility.
- Bunya segment boundaries for this cache analysis were inferred from the
  corresponding WSL per-image mask counts and checked against saved retained
  counts and feature matches. Remote source-image/mask byte identity was not
  independently checked.
- Local train and validation source image hashes had no overlap (20 unique
  training, 10 unique validation files).
- Source inspection confirmed a pretrained ResNet in evaluation mode, matching
  train/validation masking and feature settings, and a frozen concept basis
  during validation.
- `classes.py`, `concept_explainer.py` and `utils/utils_mcd.py` match the locally
  recorded upstream revision `168eb5bf1717ab46d1c84be86bd81f0c51861413`.
  This is a local Git comparison, not verification of the latest online revision.

## Why a validation sheet can show only the complement

This pipeline does not fine-tune ResNet. It discovers clusters and fits concept
subspaces using training-image segments. Validation measures how held-out
segments project into those frozen subspaces; it should not learn new concepts.

`run_smoke.assignment_quality` and `run_humcd.get_top_concept_segms` use
`argmax` over all learned concept activations **plus the complement**.
A segment can activate several concepts, yet have no learned-concept assignment
because the complement has the largest norm. The renderer omits concepts with
no winning prototypes. The final panel labeled “Concept 7” in the Bunya run is
the complement, not an eighth learned concept.

For Bunya validation:

| Quantity | Value |
|---|---:|
| Median strongest learned-concept activation | 0.3022 |
| Median complement activation | 0.8510 |
| Largest learned-concept activation across validation | 0.5589 |
| Smallest complement activation across validation | 0.7327 |
| Mean squared feature magnitude in the union of learned subspaces | 27.68% |
| Mean squared feature magnitude outside that union | 72.32% |

Thus these are not numerical ties. There is nonzero concept signal, but it is
too weak to win. The 27.68% figure is orthogonal projection energy into the
learned union, calculated as one minus the squared normalized complement norm.
It is not semantic correctness, classification accuracy, or the sum of individual
concept scores, since the learned concept subspaces need not be mutually orthogonal.

The approximately 0.96 complement score shown on the prototype figure comes
from decomposing the classifier's class-weight vector. It is distinct from the
per-segment validation activations above and is not a probability of failure.
Similarly, “PASS” in the smoke summary means execution completed, not scientific
validation succeeded.

## Likely limitation and what remains uncertain

Bunya's seven retained clusters have only **5–10 segments each**, supported by
**3–9 distinct training images**. Their learned union has rank **22 in a
2048-dimensional feature space**. The WSL fidelity run has rank 37 and clusters
of 13–27 segments. Both have limited held-out coverage; more concepts in the
smaller run does not mean better concepts.

The current PCA fits centered within-cluster variation, retains a heuristic
80% of that variance, then uses those basis directions as linear subspaces of
the raw features. Retaining 80% of centered variance does not guarantee 80%
coverage of raw validation features. This behavior matches the checked upstream
code; it is not a verified local regression and should not be silently changed.

The source runner defaults to 400 training images and uses a minimum cluster
size of 50, whereas these smoke runs use 10–20 images and a minimum size of 5.
That scale mismatch is a plausible major contributor. It does not prove that
increasing sample size alone will fix transfer. SAM settings, mask semantics,
class diversity and PCA geometry still warrant controlled checks.

Cache provenance is another future concern: segment cache names do not encode
all segmentation settings, and SSC cache names use class plus image count.
Changes in content/settings can silently reuse old results if cache directories
are recycled. No stale-cache mismatch was demonstrated in the caches inspected
here.

## Recommended next steps

1. **Keep these runs as the engineering baseline.** Retain this diagnosis, the
   exact source/config hashes, per-segment scores and reconstructed bases.
   Separate execution success from scientific quality in future reports, and
   label the complement explicitly.
2. **Make future runs diagnosable before scaling.** Save original image IDs,
   mask/feature row mappings, SSC labels, fitted bases, dimensions, cluster
   image coverage, raw concept activations and residual norms. Use fresh,
   configuration-specific cache directories and bind results to a Git commit.
3. **When a new experiment is authorized, vary training size alone first.**
   Use nested training subsets such as 20, 50 and 100 images, one fixed seed,
   one fixed held-out set, and unchanged backbone/SAM/clustering settings.
   Judge assignment rate together with continuous union coverage and prototype
   semantics. Keep a separate untouched test set if validation guides choices.
   These are diagnostic sizes, not a guarantee of paper-quality reproduction.
4. **If coverage stays poor, isolate the next factor.** Inspect dog/background
   masks and cluster examples, then assess SAM fidelity and PCA/subspace
   sensitivity separately. Treat changes to centering, dimensions or assignment
   rules as explicit methodological ablations, not an automatic bug fix.
   Removing the complement would force assignments without proving improvement.

No new experiment was submitted. Existing `hpc/check_job.sh` and all 34 files in
`outputs/` were checked against their prior hashes and preserved.

## Reproduce this cache-only check

From this repository in the existing WSL environment:

```bash
conda activate humcd-upstream
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python scripts/diagnose_cached_validation.py \
  --output /tmp/humcd-validation-recheck
```

Use a new output directory each time; the script deliberately refuses to
overwrite one. It needs the existing WSL caches and the four downloaded Bunya
cache files under `artifacts/bunya/validation-diagnosis-2026-09-08/cache/`.
It performs local spectral/PCA reconstruction and scoring, with no SSH,
segmentation or neural inference. Assertions require the checked historical
configurations and saved assignment counts; it is a diagnostic for these
specific runs, not a general experiment runner.

Saved evidence:

- `artifacts/bunya/validation-diagnosis-2026-09-08/results/diagnosis.json`
- `artifacts/bunya/validation-diagnosis-2026-09-08/results/*_segments.csv`
- `artifacts/bunya/validation-diagnosis-2026-09-08/results/*_reconstructed.npz`

Original local summaries:

- `/home/chen/results/hu-mcd-smoke/output/summary.json`
- `/home/chen/results/hu-mcd-fidelity-imagewoof320/output/summary.json`
- `outputs/bunya-phase3-smoke/summary.json`

The larger WSL run's existing `fidelity_report.md` already records poor validation
transfer. Its suggested numerical quality thresholds are provisional heuristics,
not validated paper acceptance criteria.
