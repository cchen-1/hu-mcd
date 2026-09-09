# Coordinated reproduction workstreams

This document supersedes planning-only status in older handovers. Current initial observation (2026-09-09): authenticated SSH master active; Slurm queue empty; Golden formal28208840 COMPLETED and audit28211056 PASS. No A/B/C experiment submissions were present in current accounting. HEAD before this work: eeb27e9fb31122edbfda1109c69ab7e64b094941.

A: audit all nine upstream classes using seed43 candidate ordering and existing training eligibility definition, freeze full50 validation IDs, and isolate class-level input issues before publication. No new validation conversions are implied. Discovery reuses shared verified model/batch/precision/core checks from28208840 plus its preserved prerequisite chain; each class gets independent input validation and output/cache directories.

B: cached Golden C-Insertion/C-Deletion+Random implementation, preserving upstream zero-feature retention, mean relevance ordering, dilation8, prediction mode1, endpoint and >75% rules. It is independent of A once its code/cache checks pass.

C: ACE/MCD preparation only; inspect exact method-specific inputs and cache needs. New negative-pool or expensive-fit decisions remain explicit. It does not wait for A/B metrics.

Shared worker: hpc/workstream_runtime.py. Launcher/collector: python -m hpc.workstreams. Tasks have exclusive local receipts; uncertain submissions block retries until scheduler reconciliation. All remote work including Git deployment and file audits uses Slurm. Only scheduler queries and named transfers run through login SSH.

Unified historical anomalies: docs/ANOMALIES.jsonl. New job-local anomalies.jsonl and original stderr are collected under artifacts/bunya/workstreams-20260909, then merged into the registry with evidence and disposition. User hpc/check_job.sh, outputs and unrelated NotebookLM work remain unchanged.

Live actual job IDs/configs/commits/collections: artifacts/bunya/workstreams-20260909/STATUS.json. Submitted is not completed, and worker PASS is not full-paper reproduction.
