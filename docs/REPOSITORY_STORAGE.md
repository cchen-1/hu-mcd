# Repository and research storage

Recorded 14 October 2026. Git records code, configurations, protocols, concise
research conclusions and the append-only anomaly history. Private data, models,
scientific arrays, generated reports and backup snapshots belong in research
storage, not GitHub.

## Git and Bunya versions

Before this housekeeping, local `reproduction`, GitHub `origin/reproduction`
and the checked Bunya standard release all pointed to
`4551aa8f817a066e18424264407204f3c25aab17`. The Bunya release was clean.
HAM classifier job 28885816 used `4a7360c31e41fb88f94734b0cd8167ae497c9b42`;
Derm7pt fitting job 28889071 used `649ce176c7d2e4e17b9e9be196fb0e7c33ef3e42`.
Their archived source-package hashes matched their execution receipts during
14 October verification. These are historical job identities, not branch tips.

Committing reporting tools or updating GitHub does not deploy to Bunya or change
past experiment versions. Never replace historical release directories. Use a
new pinned release only when a future execution is requested. Remote scans,
hashing, compression and data processing must run within Slurm allocations;
login nodes are for lightweight access, transfer and scheduler commands.

## What is preserved outside Git

- `artifacts/bunya/`: collected scientific outputs, checkpoints, manifests,
  acceptance receipts, logs, the private report and prior versions.
- Other `artifacts/` directories: report staging, historical snapshots,
  source/review evidence and presentation exports. Do not discard them merely
  because their names contain cache, backup or staging.
- `outputs/`, local data/model directories and binary model/array/archive files:
  private research payloads. Existing tracked upstream assets remain tracked.
- NotebookLM authentication/browser state stays outside Git. Notebook source
  texts and query history under `artifacts/notebooklm/` remain local; they are
  not required to reproduce the HPC experiments.

Ignoring a file is not a backup. On 14 October the user selected Q9903 RDM for
long-term research archives while retaining local copies. Use a new dated
archive, inventory its contents, verify checksums after transfer, and retain
originals. Record completion receipts locally; a proposed destination or an
uploaded partial file is not a verified backup. Avoid duplicate raw-data imports
where a verified existing RDM archive already exists.

## Reusing the historical tools

The reporting and acceptance scripts preserve the work performed in September
2026. Several intentionally bind to specific job IDs, historical input manifests,
absolute workstation paths or dated staging directories. A fresh clone has the
source but not the private evidence necessary to execute those scripts.

Do not run every script to rebuild a report: some perform work at module import,
write acceptance receipts, replace the live local report, update an Obsidian
note, contact NotebookLM or collect named remote results. Review the entry point
and restore its required private inputs first. Historical dated documentation
records the state at that date; later accepted-result documents supersede earlier
submission-only statements without erasing the history.

For future changes, stage explicit source/document paths, inspect the staged
file list and diff, run relevant local tests, then commit and push in logical
batches. Do not force-add ignored research payloads or use a bulk upload of the
workspace as a substitute for a reviewed commit.
