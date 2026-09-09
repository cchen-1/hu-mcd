#!/usr/bin/env python3
"""Run a small, explicit HU-MCD end-to-end engineering smoke test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch
from joblib import parallel_backend

import classes
from concept_explainer import ConceptExplainer
from run_humcd import get_top_concept_segms, save_concepts
from utils import utils_general, utils_mcd, scientific_records
from utils.run_tracking import RunTracker, atomic_json


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "smoke_local.json",
    )
    parser.add_argument("--run-id", help="Unique run ID; required and equal to job ID under Slurm")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def segment_images(
    images: list[classes.ImageClass], config: dict, cache_dir: Path
) -> None:
    segmentation = config["segmentation"]
    algorithm = segmentation["algorithm"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    if algorithm == "sam":
        checkpoint = Path(segmentation["checkpoint"]).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing SAM checkpoint: {checkpoint}")
        mask_generator = utils_general.load_sam_mask_generator(
            sam_type=segmentation["model_type"],
            sam_checkpoint=str(checkpoint),
            points_per_side=int(segmentation["points_per_side"]),
            min_mask_region_area=int(segmentation["min_mask_region_area"]),
        )
    elif algorithm == "slic":
        mask_generator = None
    else:
        raise ValueError(f"Unsupported segmentation algorithm: {algorithm}")

    for image in images:
        image.load_segments(cache_dir=str(cache_dir), sam_model=mask_generator)

    del mask_generator
    gc.collect()


def collect_activations(
    images: list[classes.ImageClass],
) -> tuple[np.ndarray, list[int]]:
    activations: list[np.ndarray] = []
    segments_per_image: list[int] = []
    for image in images:
        segments_per_image.append(len(image.segments))
        activations.extend(segment.model_act for segment in image.segments)
    return np.stack(activations, axis=0), segments_per_image


def optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def assignment_quality(
    concept_activations: np.ndarray,
    top_segments: list[list[tuple[int, int]]],
    number_of_concepts: int,
) -> dict:
    assignments = np.argmax(concept_activations, axis=1)
    complement_index = number_of_concepts
    learned_counts = [int(np.sum(assignments == index)) for index in range(number_of_concepts)]
    complement_count = int(np.sum(assignments == complement_index))
    covered_concepts = sum(count > 0 for count in learned_counts)

    top_activations = []
    for concept_index, count in enumerate(learned_counts):
        if count > 0:
            top_activations.append(
                float(np.max(concept_activations[assignments == concept_index, concept_index]))
            )

    learned_prototypes = [
        prototype
        for concept_prototypes in top_segments[:number_of_concepts]
        for prototype in concept_prototypes
    ]
    unique_images = {image_index for image_index, _ in learned_prototypes}
    sorted_activations = np.sort(concept_activations, axis=1)
    assignment_margin = sorted_activations[:, -1] - sorted_activations[:, -2]

    return {
        "segment_count": int(len(assignments)),
        "learned_concept_assignment_count": int(np.sum(assignments != complement_index)),
        "orthogonal_complement_assignment_count": complement_count,
        "learned_concept_assignment_rate": float(np.mean(assignments != complement_index)),
        "concept_assignment_counts": learned_counts,
        "prototype_coverage_rate": float(covered_concepts / number_of_concepts),
        "selected_learned_prototypes": len(learned_prototypes),
        "unique_prototype_images": len(unique_images),
        "unique_prototype_image_rate": (
            float(len(unique_images) / len(learned_prototypes))
            if learned_prototypes
            else 0.0
        ),
        "mean_top_prototype_activation": (
            float(np.mean(top_activations)) if top_activations else None
        ),
        "mean_assignment_margin": float(np.mean(assignment_margin)),
    }


def write_metrics_report(summary: dict, path: Path) -> None:
    timing_rows = "\n".join(
        f"| `{name}` | {seconds:.3f} |"
        for name, seconds in summary["stage_elapsed_seconds"].items()
    )
    train_quality = summary["prototype_quality_proxy"]["training"]
    validation_quality = summary["prototype_quality_proxy"]["validation"]
    report = f"""# HU-MCD Local Fidelity Metrics

## Scope

This is an engineering fidelity run, not a paper-result reproduction. Prototype
quality values below are objective assignment/coverage proxies; semantic quality
still requires visual inspection of the prototype sheets.

## Configuration

| Metric | Value |
|---|---:|
| Experiment | `{summary['experiment_name']}` |
| Dataset | `{summary['dataset_name']}` |
| Target class | `{summary['class_name']}` |
| Device | `{summary['device']}` |
| SAM model | `{summary['sam_model_type']}` |
| SAM points per side | {summary['sam_points_per_side']} |
| Training images | {summary['train_images']} |
| Validation images | {summary['validation_images']} |

## Core metrics

| Metric | Value |
|---|---:|
| Status | `{summary['status']}` |
| Total elapsed time (s) | {summary['elapsed_seconds']:.3f} |
| Peak self RSS (MiB) | {summary['peak_self_rss_mib']:.1f} |
| Peak child RSS (MiB) | {summary['peak_children_rss_mib']:.1f} |
| GPU device | `{summary['gpu_device_name']}` |
| GPU total memory (MiB) | {summary['gpu_total_memory_mib']} |
| Peak GPU allocated memory (MiB) | {summary['peak_gpu_memory_allocated_mib']} |
| Peak GPU reserved memory (MiB) | {summary['peak_gpu_memory_reserved_mib']} |
| Training segments | {sum(summary['train_segments_per_image'])} |
| Validation segments | {sum(summary['validation_segments_per_image'])} |
| Retained concepts | {summary['concepts']} |
| Concept dimensions | `{summary['concept_dimensions']}` |
| Completeness | {summary['completeness']:.6f} |

## Stage timing

| Stage | Seconds |
|---|---:|
{timing_rows}

## Prototype quality proxies

| Metric | Training | Validation |
|---|---:|---:|
| Learned-concept assignment rate | {train_quality['learned_concept_assignment_rate']:.4f} | {validation_quality['learned_concept_assignment_rate']:.4f} |
| Prototype coverage rate | {train_quality['prototype_coverage_rate']:.4f} | {validation_quality['prototype_coverage_rate']:.4f} |
| Unique prototype image rate | {train_quality['unique_prototype_image_rate']:.4f} | {validation_quality['unique_prototype_image_rate']:.4f} |
| Mean top prototype activation | {train_quality['mean_top_prototype_activation']} | {validation_quality['mean_top_prototype_activation']} |
| Mean assignment margin | {train_quality['mean_assignment_margin']:.4f} | {validation_quality['mean_assignment_margin']:.4f} |

## Artifacts

- Training prototype sheet: `{summary['training_overview']}`
- Validation prototype sheet: `{summary['validation_overview']}`
- Machine-readable summary: `{summary['summary_path']}`
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with RunTracker(config, config_path, REPO_ROOT, args.run_id) as tracker:
        # Use allocated worker processes without multiplying BLAS threads per worker.
        with parallel_backend("loky", inner_max_num_threads=1):
            run(config_path, tracker.config, tracker)


def run(config_path: Path, config: dict, tracker: RunTracker) -> None:
    os.chdir(REPO_ROOT)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config["torch_num_threads"]))

    device = str(config["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    utils_general.DEVICE = device

    gpu_device_name = None
    gpu_total_memory_mib = None
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        properties = torch.cuda.get_device_properties(device)
        gpu_device_name = properties.name
        gpu_total_memory_mib = properties.total_memory / 1024**2

    precision = {
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
    }
    tracker.record_precision(precision)
    if config.get('precision') is not None and precision != config['precision']:
        raise ValueError('Runtime precision differs from the fixed reference configuration')

    source_dir = Path(config["source_dir"]).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    cache_root = Path(config["cache_root"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    stage_elapsed_seconds = tracker.stage_times
    class_name = config["class_name"]
    stage_started = time.perf_counter()
    explainer = ConceptExplainer(
        source_dir=str(source_dir),
        target_class=class_name,
        model_name=config["model_name"],
        layer_name=config["layer_name"],
    )
    explainer.load_class_images(
        num_imgs=int(config["train_images"]),
        max_shortest_side=int(config["max_shortest_side"]),
    )
    if len(explainer.class_imgs) != int(config["train_images"]):
        raise RuntimeError("The requested number of training images was not loaded")
    tracker.record_inputs("training", explainer.class_imgs)
    stage_elapsed_seconds["model_and_training_image_load"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    segment_images(explainer.class_imgs, config, cache_root / "segments_train")
    stage_elapsed_seconds["training_segmentation"] = time.perf_counter() - stage_started
    explainer.segm_algo = config["segmentation"]["algorithm"]
    stage_started = time.perf_counter()
    explainer.compute_cls_segms_acts(
        cache_dir=str(cache_root / "activations_train"),
        cropping_mode=0,
        use_masks=True,
        masking_mode=-1,
        erosion_threshold=0.25,
        batch_size=int(config["batch_size"]),
        norm_acts=False,
    )
    stage_elapsed_seconds["training_activations"] = time.perf_counter() - stage_started

    clustering = config["clustering"]
    stage_started = time.perf_counter()
    explainer.create_concepts(
        cluster_algo=clustering["algorithm"],
        n_clusters=optional_int(clustering["n_clusters"]),
        norm_acts=True,
        min_size=int(clustering["min_cluster_size"]),
        min_coverage=0.0,
        max_samples=None,
        outlier_percentile=float(clustering["outlier_percentile"]),
        folderpath=str(cache_root / "self_representation"),
    )
    if not explainer.concepts:
        raise RuntimeError("No concepts survived the smoke-test cluster filters")
    stage_elapsed_seconds["sparse_subspace_clustering"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    explainer.compute_concept_subspace_bases(
        subspace_dimensionality=optional_int(clustering["subspace_dimensionality"]),
        est_mode="ratio",
        compt_princ_angl=False,
    )
    stage_elapsed_seconds["concept_subspace_pca"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    class_index = utils_general.get_imagenet_class_index(class_name)
    weight_vector = (
        explainer.model.fc.weight.data.detach()[class_index].cpu().numpy()
    )
    if config.get("save_scientific_records", False):
        scientific_records.save_discovery(explainer, output_dir, class_index)
    concept_scores, _ = explainer.concept_quantification(weight_vector)
    completeness = float(
        utils_mcd.calc_completeness(weight_vector, explainer.concept_bases)
    )
    stage_elapsed_seconds["concept_scoring_and_completeness"] = time.perf_counter() - stage_started

    prototypes_per_concept = int(config.get("prototypes_per_concept", 1))
    concepts_to_plot = min(int(config.get("n_concepts_to_plot", 3)), len(concept_scores))
    stage_started = time.perf_counter()
    train_acts, train_batch_sizes = collect_activations(explainer.class_imgs)
    train_concept_activations = explainer.concept_activations(
        acts=train_acts,
        batch_sizes=train_batch_sizes,
        norm_batch=False,
        n_jobs=1,
    )
    if config.get("save_scientific_records", False):
        scientific_records.save_split(explainer, explainer.class_imgs, train_concept_activations,
                                      output_dir, "training", class_index)
    train_top_segments = get_top_concept_segms(
        concept_activations=train_concept_activations,
        n_segms_per_img=train_batch_sizes,
        n_prototypes=prototypes_per_concept,
    )
    train_prototype_dir = output_dir / "training_prototypes"
    train_prototype_dir.mkdir(parents=True, exist_ok=True)
    save_concepts(
        top_concept_segms=train_top_segments,
        imgs=explainer.class_imgs,
        concept_scores=concept_scores,
        n_concepts=concepts_to_plot,
        n_imgs_per_concepts=prototypes_per_concept,
        folderpath=str(train_prototype_dir),
    )
    train_quality = assignment_quality(
        train_concept_activations, train_top_segments, len(explainer.concepts)
    )
    stage_elapsed_seconds["training_assignment_and_prototypes"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    validation_images = explainer._load_images(
        folderpath=str(source_dir / "val_imgs" / f"{class_name}_val"),
        num_imgs=int(config["validation_images"]),
        max_shortest_side=int(config["max_shortest_side"]),
        shuffle=False,
    )
    if len(validation_images) != int(config["validation_images"]):
        raise RuntimeError("The requested number of validation images was not loaded")
    tracker.record_inputs("validation", validation_images)
    stage_elapsed_seconds["validation_image_load"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    segment_images(validation_images, config, cache_root / "segments_validation")
    stage_elapsed_seconds["validation_segmentation"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    explainer._load_or_calc_acts(
        cache_dir=str(cache_root / "activations_validation"),
        identifier=f"{class_name}_validation",
        images=validation_images,
        cropping_mode=0,
        use_masks=True,
        masking_mode=-1,
        erosion_threshold=0.25,
        batch_size=int(config["batch_size"]),
        norm_acts=False,
        delete_zeros=True,
    )
    stage_elapsed_seconds["validation_activations"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    validation_acts, validation_batch_sizes = collect_activations(validation_images)
    concept_activations = explainer.concept_activations(
        acts=validation_acts,
        batch_sizes=validation_batch_sizes,
        norm_batch=False,
        n_jobs=1,
    )
    if config.get("save_scientific_records", False):
        scientific_records.save_split(explainer, validation_images, concept_activations,
                                      output_dir, "validation", class_index)
    top_segments = get_top_concept_segms(
        concept_activations=concept_activations,
        n_segms_per_img=validation_batch_sizes,
        n_prototypes=prototypes_per_concept,
    )
    validation_prototype_dir = output_dir / "validation_prototypes"
    validation_prototype_dir.mkdir(parents=True, exist_ok=True)
    save_concepts(
        top_concept_segms=top_segments,
        imgs=validation_images,
        concept_scores=concept_scores,
        n_concepts=concepts_to_plot,
        n_imgs_per_concepts=prototypes_per_concept,
        folderpath=str(validation_prototype_dir),
    )
    validation_quality = assignment_quality(
        concept_activations, top_segments, len(explainer.concepts)
    )
    stage_elapsed_seconds["validation_assignment_and_prototypes"] = time.perf_counter() - stage_started

    checkpoint = Path(config["segmentation"]["checkpoint"]).expanduser()
    checkpoint_hash = sha256(checkpoint)
    peak_self_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    peak_children_rss_mib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    peak_gpu_memory_allocated_mib = None
    peak_gpu_memory_reserved_mib = None
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        peak_gpu_memory_allocated_mib = (
            torch.cuda.max_memory_allocated(device) / 1024**2
        )
        peak_gpu_memory_reserved_mib = (
            torch.cuda.max_memory_reserved(device) / 1024**2
        )
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "metrics_report.md"
    summary = {
        "status": "PASS",
        **tracker.identity,
        "git_commit": tracker.manifest["git_commit"],
        "resolved_config_sha256": tracker.manifest["resolved_config_sha256"],
        "metric_scope": "engineering_fidelity_not_paper_result",
        "experiment_name": config.get("experiment_name", config_path.stem),
        "dataset_name": config.get("dataset_name", "unspecified"),
        "config": str(config_path),
        "class_name": class_name,
        "imagenet_class_index": int(class_index),
        "device": device,
        "gpu_device_name": gpu_device_name,
        "gpu_total_memory_mib": gpu_total_memory_mib,
        "peak_gpu_memory_allocated_mib": peak_gpu_memory_allocated_mib,
        "peak_gpu_memory_reserved_mib": peak_gpu_memory_reserved_mib,
        "segmentation_algorithm": config["segmentation"]["algorithm"],
        "sam_model_type": config["segmentation"]["model_type"],
        "sam_points_per_side": int(config["segmentation"]["points_per_side"]),
        "sam_checkpoint": str(checkpoint.resolve()),
        "sam_checkpoint_sha256": checkpoint_hash,
        "train_images": len(explainer.class_imgs),
        "train_segments_per_image": [len(image.segments) for image in explainer.class_imgs],
        "validation_images": len(validation_images),
        "validation_segments_per_image": validation_batch_sizes,
        "concepts": len(explainer.concepts),
        "concept_dimensions": [int(basis.shape[0]) for basis in explainer.concept_bases],
        "concept_scores": [float(score) for score in concept_scores],
        "completeness": completeness,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_self_rss_mib": peak_self_rss_mib,
        "peak_children_rss_mib": peak_children_rss_mib,
        "stage_elapsed_seconds": stage_elapsed_seconds,
        "prototype_quality_proxy": {
            "training": train_quality,
            "validation": validation_quality,
        },
        "training_overview": str((train_prototype_dir / "overview.png").resolve()),
        "validation_overview": str((validation_prototype_dir / "overview.png").resolve()),
        "summary_path": str(summary_path.resolve()),
        "metrics_report": str(report_path.resolve()),
    }
    write_metrics_report(summary, report_path)
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
