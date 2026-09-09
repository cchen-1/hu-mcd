# Coordinated reproduction workstreams

This document supersedes planning-only status in older handovers. Current initial observation (2026-09-09): authenticated SSH master active; Slurm queue empty; Golden formal28208840 COMPLETED and audit28211056 PASS. No A/B/C experiment submissions were present in current accounting. HEAD before this work: eeb27e9fb31122edbfda1109c69ab7e64b094941.

A: audit all nine upstream classes using seed43 candidate ordering and existing training eligibility definition, freeze full50 validation IDs, and isolate class-level input issues before publication. No new validation conversions are implied. Discovery reuses shared verified model/batch/precision/core checks from28208840 plus its preserved prerequisite chain; each class gets independent input validation and output/cache directories.

B: cached Golden C-Insertion/C-Deletion+Random implementation, preserving upstream zero-feature retention, mean relevance ordering, dilation8, prediction mode1, endpoint and >75% rules. It is independent of A once its code/cache checks pass.

C: ACE/MCD preparation only; inspect exact method-specific inputs and cache needs. New negative-pool or expensive-fit decisions remain explicit. It does not wait for A/B metrics.

Shared worker: hpc/workstream_runtime.py. Launcher/collector: python -m hpc.workstreams. Tasks have exclusive local receipts; uncertain submissions block retries until scheduler reconciliation. All remote work including Git deployment and file audits uses Slurm. Only scheduler queries and named transfers run through login SSH.

Unified historical anomalies: docs/ANOMALIES.jsonl. New job-local anomalies.jsonl and original stderr are collected under artifacts/bunya/workstreams-20260909, then merged into the registry with evidence and disposition. User hpc/check_job.sh, outputs and unrelated NotebookLM work remain unchanged.

Live actual job IDs/configs/commits/collections: artifacts/bunya/workstreams-20260909/STATUS.json. Submitted is not completed, and worker PASS is not full-paper reproduction.

## Observed input audit and execution scope (2026-09-09)

CPU inventory/audit28214251 completed in49s; full candidate image checks took place on its compute node. Each new class has1300 candidates and50 frozen validation IDs. Cross-class original-file SHA256 check:4000 unique selected training and500 unique validation images, with no intersection. This does not test perceptual duplicates.

Initially, four classes cleared for exact-list publication: hummingbird, Siamese_cat, tailed_frog, container_ship. Jobs28214690–28214693 completed successfully. The other five classes were initially held at their input decision: airliner/police_van/beach_wagon have potentially historical subtype concentration among monochrome thumbnails; zebra has66/1300 grayscale candidates; ox28/1300, including repeated working/display scenes. These are qualitative warnings, not measured relative-to-colour biases. Eight validation L images across beach_wagon/zebra/ox retained their original IDs while awaiting explicit compatibility approval. The existing Golden input approval is not assumed to cover new classes.

B uses13 cached bases, all569 raw validation segments and existing verified masks; no SAM or discovery rerun. Allocation planned:1L40S,4CPU,16GiB,1h;10652 prediction states with global batch8. A planned per-class allocation:1L40S,8CPU,32GiB,2h, maximum two A jobs concurrently plus one B; afterany resource dependencies do not make independent classes scientifically dependent. Golden's33m50s is an observed reference, not a guaranteed duration for other classes.

C preparation and upstream protocol findings are in BASELINE_PREPARATION.md and hpc/configs/golden_retriever_baselines.plan.json. No baseline fit or random-pool selection has been submitted. Fourteen focused shared/evaluation tests passed locally; dependency/cache-location warnings are documented in the anomaly register.

## User decisions and current submissions

The user subsequently approved all five unchanged seed43 filtered training lists and the eight exact L-to-RGB-PNG validation adjustments, with unchanged dimensions and channel-by-channel pixel verification. Scene risks remain unverified. See NINE_CLASS_INPUT_PROTOCOL.md for counts and the paper/upstream difference, including the upstream loader accepting achromatic RGB files that our training rule excludes.

Execution release982045f0e392e2bea02334a48afb96e6dcd26fc9 deployed byCPU28214828. B evaluation28214892; A discoveries hummingbird28214893, Siamese_cat28214894, tailed_frog28214896, container_ship28214897, airliner28214983, police_van28214984. Remaining three require the newly approved input compatibility publication, then their own formal jobs. Each submission receipt and resource dependency is in the live STATUS directory. These are submission facts, not completed results.

Five focused conversion tests passed: exact whitelist/identity, atomic failure, pixel equality, unchanged original lists, and unaffected RGB publication. The compatibility helper is an engineering input adapter under the user's explicit authorization; it does not alter training selection or scientific core functions. A later release adds only this adapter/tests/documentation; earlier independent jobs stay bound to their original release.
