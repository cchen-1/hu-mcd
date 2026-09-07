#!/usr/bin/env python3
"""Minimal CUDA and SAM probe for the Bunya Phase 3 gate."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the allocated GPU job")

    checkpoint = Path(
        os.environ.get(
            "HUMCD_SAM_CHECKPOINT",
            "/scratch/user/uqcche38/hu-mcd/models/sam_vit_b_01ec64.pth",
        )
    )
    output_dir = Path(
        os.environ.get(
            "HUMCD_PROBE_OUTPUT",
            "/scratch/user/uqcche38/hu-mcd/outputs/gpu_probe",
        )
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    properties = torch.cuda.get_device_properties(device)

    started = time.perf_counter()
    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    model.to(device=device)
    generator = SamAutomaticMaskGenerator(
        model=model,
        points_per_side=4,
        points_per_batch=16,
        min_mask_region_area=256,
    )
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[64:192, 64:192, :] = 180
    masks = generator.generate(image)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    result = {
        "status": "PASS",
        "host": platform.node(),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_device": properties.name,
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_total_memory_mib": properties.total_memory / 1024**2,
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "sam_model": "vit_b",
        "sam_checkpoint": str(checkpoint),
        "sam_checkpoint_sha256": sha256(checkpoint),
        "generated_masks": len(masks),
        "elapsed_seconds": elapsed,
    }
    destination = output_dir / "gpu_probe.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
