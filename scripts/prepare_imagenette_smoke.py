#!/usr/bin/env python3
"""Create an HU-MCD-compatible smoke subset from Imagenette2-160."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


CLASS_TO_SYNSET = {
    "tench": "n01440764",
    "English_springer": "n02102040",
    "cassette_player": "n02979186",
    "chain_saw": "n03000684",
    "church": "n03028079",
    "French_horn": "n03394916",
    "garbage_truck": "n03417042",
    "gas_pump": "n03425413",
    "golf_ball": "n03445777",
    "parachute": "n03888257",
}


def rgb_images(folder: Path) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                if image.mode == "RGB":
                    selected.append(path.resolve())
        except OSError:
            continue
    return selected


def link_images(images: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for index, source in enumerate(images, start=1):
        suffix = source.suffix.lower() or ".jpeg"
        target = destination / f"{index:03d}_{source.stem}{suffix}"
        target.symlink_to(source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenette-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-name", choices=sorted(CLASS_TO_SYNSET), default="garbage_truck")
    parser.add_argument("--train-count", type=int, default=3)
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_count < 1 or args.val_count < 1:
        raise ValueError("train-count and val-count must both be positive")

    synset = CLASS_TO_SYNSET[args.class_name]
    train_source = args.imagenette_root / "train" / synset
    val_source = args.imagenette_root / "val" / synset
    if not train_source.is_dir() or not val_source.is_dir():
        raise FileNotFoundError(
            f"Expected Imagenette folders {train_source} and {val_source}"
        )

    train_images = rgb_images(train_source)[: args.train_count]
    val_images = rgb_images(val_source)[: args.val_count]
    if len(train_images) != args.train_count or len(val_images) != args.val_count:
        raise RuntimeError("Not enough RGB images for the requested subset")

    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
        shutil.rmtree(output)

    train_destination = output / args.class_name
    val_destination = output / "val_imgs" / f"{args.class_name}_val"
    link_images(train_images, train_destination)
    link_images(val_images, val_destination)

    manifest = {
        "dataset": "Imagenette2-160",
        "imagenette_root": str(args.imagenette_root.resolve()),
        "class_name": args.class_name,
        "imagenet_synset": synset,
        "train_images": [str(path) for path in train_images],
        "validation_images": [str(path) for path in val_images],
        "layout": {
            "train": str(train_destination),
            "validation": str(val_destination),
        },
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
