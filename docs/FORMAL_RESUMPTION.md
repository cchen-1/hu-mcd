# Single-class reference resumption after probe diagnosis

The original probe 28204575 remains FAILED. Its all-one-mask test changed batch
size between its two forwards. L40S diagnosis 28206208 reproduced the exact error
on the plain model, established bitwise masked/plain equality at matched batch
sizes 1, 2 and 8, and isolated the error to the cuDNN TF32 numerical path.
See [the diagnosis](PROBE_FAILURE_DIAGNOSIS.md).

## Changes and evidence

- The probe now compares both paths at the same batch size and original tolerance.
  Boundary arrays are written before assertion, with incremental stage records.
- CPU Slurm **28208779** completed the unexecuted algebra/SSC checks using the
  original feature cache; no SAM rerun or new concept-fitting probe. Algebra
  reconstruction error was zero. SSC used 37 nonzero rows (one original zero row
  excluded), returned 295 nonzero coefficients, and passed finite/shape/no-self
  checks. Cached FC maximum error was 3.814697265625e-6.
- `hpc/manifests/probe_28204575_completed.json` preserves this PASS report.
  Its remote path is
  `/scratch/user/uqcche38/hu-mcd/launches/28208779/prerequisites/prerequisites.json`,
  SHA256 `77d2730874cc18db9d7547fc922adea14d174dc28d697985d3a69e854a08b82d`.
- Launcher arguments `--prerequisite-report` and `--prerequisite-report-sha256`
  select an explicit completed-evidence route. It verifies bound original probe,
  diagnosis, source/helper hashes, cached inputs, science configuration and
  precision, and the completed algebra/SSC checks. It records all three job IDs
  without relabeling the original probe or inheriting its failed dependency.
- No core algorithm files changed. Formal inputs remain the fixed seed-43
  400 training images and 50 validation IDs, including the single approved RGB
  compatibility PNG. Batch=8, ViT-H/32 points and released clustering settings
  remain fixed. Keep 1 L40S, 8 CPUs, 32 GiB, short QOS, 12-hour limit.
- The config pins the observed precision flags. The runner records actual flags
  in `run_manifest.json` before model work, failing if they differ. It does not
  silently change defaults: cuDNN TF32 on, matmul TF32 off, benchmark off,
  deterministic modes off, matmul precision highest.
- Existing per-job output/cache isolation and stage progress remain. Scientific
  arrays are saved before checks; failures preserve features, logits, assignment
  scores, assignments and image/mask mappings with failed-check details. Fitted
  bases, complement, labels and relevance reconstruction samples are retained.
  Scientific file hashes are published after their stages and on failure, making
  existing `--include-scientific` collection useful before final completion.

## Validation and operation

Focused regressions cover batch-dependent numerical drift versus an actual mask
bug; altered/failed prerequisite evidence; stale dependency exclusion; and raw
assignment preservation on FC failure. The existing producer/collector integration
suite also checks live and completed snapshots, exact job/commit identity and hashes.

The code version must be committed/pushed before deployment. Deployment uses a
CPU Slurm job, and verifies the real evidence gate from the prepared immutable
release. Only after successful deployment should one formal job be submitted.
Check `squeue --me` and accounting for old job **28205921** first: it was cancelled
without running. No automatic resubmission is configured.

Live submission/deployment IDs, current states, exact Git SHA and collection
snapshots are maintained in the ignored local directory
`artifacts/bunya/reference-resumption/STATUS.md`; immutable batch scripts and
submission receipts sit alongside it. This avoids changing a prepared release
just to update a job's status.

Collector invocation after obtaining the new job ID and SHA:

```bash
python3 hpc/collect_results.py JOB_ID --profile reference --run-layout \
  --expected-commit FULL_SHA --include-scientific --max-file-mib 256
```

A pending job has no research outputs yet; collector status `partial` is expected.
A successful engineering run is not by itself paper-metric reproduction. The
formal retained/complement assignments and reconstruction records will diagnose
held-out concept coverage; faithfulness benchmarks remain a subsequent milestone.
