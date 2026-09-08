# HU-MCD Phase 3 on UQ Bunya

This directory provides a deliberately small, gated deployment workflow. It is
designed to protect fair-share priority and to prevent computation or software
installation on Bunya login nodes.

## Fixed locations and account

| Item | Location/value |
|---|---|
| Slurm account | `a_ai_collab` |
| Remote project | `/scratch/user/uqcche38/hu-mcd` |
| Conda environment | `/home/uqcche38/.conda/envs/humcd-bunya` |
| Model files | `/scratch/user/uqcche38/hu-mcd/models` |
| Smoke data | `/scratch/user/uqcche38/hu-mcd/data/phase3_smoke` |
| Outputs | `/scratch/user/uqcche38/hu-mcd/outputs` |

The environment is kept in `/home` because it is durable. Data, caches, model
weights, and outputs are kept in `/scratch`, which has the larger quota and is
the correct location for running jobs. Scratch data must still be archived
because inactive files can expire.

## Login-node rule

A prompt containing `bunya1` through `bunya5` is a login node. On those nodes,
only use lightweight access and scheduler commands such as:

```bash
ssh uqcche38@bunya.rcc.uq.edu.au
sbatch ...
squeue --me
sacct ...
scancel JOB_ID
module spider ...
```

Do not run Python, Conda/Mamba installs, model loading, archive extraction, or
dataset processing there. Every supplied `sbatch` script checks that it is on a
`bunNNN` compute node before doing work.

## Resource ladder

| Gate | Requested resources | Purpose |
|---|---|---|
| Stage archive | 1 CPU, 2 GiB, 10 min, CPU | Verify and extract uploaded files |
| Build environment | 2 CPUs, 8 GiB, 45 min, CPU | Create the pinned CUDA environment |
| GPU probe | 1 CPU, 4 GiB, 5 min, 1 L40S | Test CUDA, SAM, and GPU visibility |
| Phase 3 smoke | 4 CPUs, 16 GiB, 30 min, 1 L40S | Run 10 train + 5 validation images |

No A100 or H100 is requested. The smoke run uses the existing SAM ViT-B model.
SAM ViT-H and the two-class 50/20 pilot are intentionally deferred until the
smoke metrics show the resources actually required.

## 1. Build the package locally

Run this only in local WSL, where the prompt is not a Bunya host:

```bash
cd /home/chen/projects/hu-mcd
bash hpc/package_local.sh
```

The command creates the following ignored files under `dist/`:

```text
dist/hu-mcd-phase3-smoke.tar.gz
dist/hu-mcd-phase3-smoke.tar.gz.sha256
```

The package includes the repository, real copies of the 20/10 Imagewoof images
behind the local symlinks, SAM ViT-B, and the cached pretrained ResNet50 weight.

## 2. Upload from local WSL

Still from local WSL:

```bash
scp dist/hu-mcd-phase3-smoke.tar.gz \
  dist/hu-mcd-phase3-smoke.tar.gz.sha256 \
  hpc/stage_bundle.sbatch \
  uqcche38@bunya.rcc.uq.edu.au:/scratch/user/uqcche38/
```

`scp` is a transfer operation supported by Bunya. It does not start the HU-MCD
workload on the login node.

## 3. Submit the CPU staging job

Log in to Bunya, then submit only:

```bash
ssh uqcche38@bunya.rcc.uq.edu.au
sbatch /scratch/user/uqcche38/stage_bundle.sbatch
squeue --me
```

Do not extract the archive manually. When the job finishes, inspect its status:

```bash
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode
```

The staging log is `/scratch/user/uqcche38/stage-humcd-JOB_ID.out`.

## 4. Submit the CPU environment job

Only after the staging job reports `COMPLETED`:

```bash
cd /scratch/user/uqcche38/hu-mcd
sbatch hpc/setup_environment.sbatch
squeue --me
```

This submission command runs on the login node, but installation runs inside a
small Slurm CPU allocation. Review `logs/setup-environment-JOB_ID.out` and the
`sacct` record before continuing.

## 5. Submit the five-minute GPU probe

Only after environment setup reports `COMPLETED`:

```bash
cd /scratch/user/uqcche38/hu-mcd
sbatch hpc/gpu_probe.sbatch
squeue --me
```

The probe loads SAM ViT-B and runs low-density automatic segmentation on one
synthetic image. It writes:

```text
outputs/gpu_probe/gpu_probe.json
logs/gpu-probe-JOB_ID.out
```

Continue only if CUDA is available, the device is L40S, SAM completes, and the
job exits successfully.

## 6. Submit the Phase 3 smoke test

```bash
cd /scratch/user/uqcche38/hu-mcd
sbatch hpc/phase3_smoke.sbatch
squeue --me
```

The smoke test writes English metrics and landscape prototype sheets under:

```text
outputs/phase3_smoke/
```

Inspect scheduler and utilisation metrics after completion:

```bash
bash hpc/check_job.sh JOB_ID
```

The helper only queries Slurm/jobstats and is safe on the login node.

## Stop conditions

Do not submit the next gate when:

- the previous job did not finish with `COMPLETED`;
- CUDA or SAM validation failed;
- peak CPU RAM is close to the request;
- peak GPU memory is close to 48 GiB;
- GPU utilisation indicates that the workload is blocked on CPU or I/O;
- outputs or metrics are missing.

Cancel an unexpected or wasteful job immediately:

```bash
scancel JOB_ID
```

## After the smoke gate

The current next milestone is a one-class reference reproduction using the
paper/released-code configuration after the deployment and evaluation checks.
The earlier 50/20 two-class pilot proposal is superseded by
[the main-chat handover](../docs/MAIN_CHAT_HANDOVER.md) and
[the reproduction audit](../docs/PAPER_REPRODUCTION_AUDIT.md).

The existing smoke sbatch still selects the old code directory and ViT-B smoke
configuration. A pushed commit or a prepared Git release alone does not update
that launcher. No paper-scale job is configured or submitted by this change.


## Collect existing results locally

After your manual WSL SSH login, use the [local results collector](COLLECTOR.md)
to retrieve existing job reports, logs and prototype sheets without submitting
a job. It reuses your authenticated connection and keeps separate snapshots.
