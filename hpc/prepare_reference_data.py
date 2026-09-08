"""Prepare an immutable HU-MCD split from an explicitly verified ImageNet layout."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_rgb(path):
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB":
            raise ValueError(f"Selected image is {image.mode}, not RGB; no implicit conversion: {path}")


def prepare(dataset_root, output, seed, train_count=400, val_count=50, image_check=check_rgb, audit_dir=None, audit_only=False):
    root = Path(dataset_root).resolve(strict=True)
    output = Path(output)
    if not output.is_absolute() or output.exists():
        raise ValueError("Output must be a new absolute directory")
    if root == output or root in output.parents:
        raise ValueError("Never write inside the shared licensed dataset")
    candidates = {}
    for split in ("train", "val"):
        folder = root / split / "n02099601"
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        candidates[split] = sorted(p for p in folder.iterdir()
                                   if p.is_file() and p.suffix.lower() in (".jpeg", ".jpg", ".png"))
    if len(candidates["train"]) < train_count or len(candidates["val"]) != val_count:
        raise ValueError(f"Expected >= {train_count} train and exactly {val_count} validation images; "
                         f"found {len(candidates['train'])} and {len(candidates['val'])}")
    selected = {"training": random.Random(seed).sample(candidates["train"], train_count),
                "validation": candidates["val"]}
    manifest = {"schema_version": 1, "dataset": "ImageNet1k", "synset": "n02099601",
                "class_name": "golden_retriever", "seed": seed, "source_dir": str(output),
                "licensed_source_root": str(root), "python": sys.version,
                "sampling": "random.Random(seed).sample(sorted training filenames); all sorted validation files",
                "candidate_counts": {k: len(v) for k,v in candidates.items()},
                "candidate_filename_hashes": {
                    k: hashlib.sha256("\n".join(p.name for p in v).encode()).hexdigest()
                    for k,v in candidates.items()},
                "preparation_job_id": os.environ.get("SLURM_JOB_ID")}
    paths, hashes = set(), set()
    problems = []
    for split, items in selected.items():
        manifest[split] = []
        for i,path in enumerate(items,1):
            canonical = str(path.resolve(strict=True))
            try:
                image_check(path)
                image_error = None
            except (ValueError, OSError) as exc:
                image_error = f"{type(exc).__name__}: {exc}"
                problems.append(image_error)
            checksum = digest(path)
            if canonical in paths or checksum in hashes:
                problems.append(f"Duplicate content/path within or across splits: {path}")
            paths.add(canonical)
            hashes.add(checksum)
            manifest[split].append({"source": canonical, "sha256": checksum,
                                    "image_error": image_error, "bytes": path.stat().st_size,
                                    "prepared_name": f"{i:04d}_{path.name}"})
    manifest["image_validation_problems"] = problems
    if audit_dir:
        audit_dir = Path(audit_dir)
        (audit_dir/"selection_audit.json").write_text(json.dumps(manifest,indent=2)+"\n")
        for split in ("training", "validation"):
            (audit_dir/(split+"_images.txt")).write_text(
                "\n".join(e["source"] for e in manifest[split])+"\n")
    if audit_only:
        return {"status": "SELECTION_AUDITED", "training_count": train_count,
                "validation_count": val_count, "problems": problems}
    if problems:
        raise ValueError("; ".join(problems))
    manifest["overlap_check"] = {"duplicate_paths": 0, "duplicate_contents": 0,
                                "training_count": train_count, "validation_count": val_count}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-golden-", dir=output.parent) as td:
        staging = Path(td) / "split"
        for split,folder in (("training",staging/"golden_retriever"),
                             ("validation",staging/"val_imgs/golden_retriever_val")):
            folder.mkdir(parents=True)
            for entry in manifest[split]:
                (folder/entry["prepared_name"]).symlink_to(entry["source"])
        (staging/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        for split in ("training","validation"):
            (staging/(split+"_images.txt")).write_text(
                "\n".join(e["source"] for e in manifest[split])+"\n")
        # Refuse replacement even if another process creates the destination.
        lock = output.parent / (output.name + ".publish.lock")
        import fcntl
        with lock.open("a") as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            if output.exists():
                raise FileExistsError(output)
            os.rename(staging,output)
    return {"source_dir":str(output),"manifest_path":str(output/"dataset_manifest.json"),
            "manifest_sha256":digest(output/"dataset_manifest.json"),
            **manifest["overlap_check"]}


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,required=True)
    args=p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Remote data preparation must run under Slurm")
    print(json.dumps(prepare(args.dataset_root,args.output,args.seed),indent=2))
