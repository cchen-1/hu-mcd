"""Raw-HAM input adapter preparation; no training, downloads or submission.

Delegate geometry and tensor arithmetic to the released HU-MCD classes instead
of introducing a second resize implementation. This helper is not yet wired to
a model-training launcher. Remote invocation must occur inside Slurm.
"""
from pathlib import Path

import numpy as np
from PIL import Image


def _released_dataset(path, model_cfg):
    """Build the released unmasked dataset and geometry metadata.

    Reject unexpected modes rather than silently converting or replacing inputs.
    This deliberately does not run SAM or load classifier weights.
    """
    from classes import ImageClass, SegmentClass, ConceptDatasetClass

    path = Path(path)
    with Image.open(path) as original:
        if original.mode != 'RGB':
            raise ValueError(f'Unexpected input mode {original.mode}: {path}')
        original_size = original.size
    image = ImageClass(str(path), max_shortest_side=300)
    image.segments = [SegmentClass(
        np.ones(image.img_numpy.shape[:2], dtype=np.float32), image)]
    dataset = ConceptDatasetClass(
        [image], model_cfg, cropping_mode=0, use_masks=False)
    return dataset, {
        'source_path': str(path),
        'original_size_wh': list(original_size),
        'pre_sam_size_wh': list(image.img_pil.size),
        'classifier_size_chw': list(model_cfg['input_size']),
        'max_shortest_side': 300,
        'cropping_mode': 0,
        'implementation': 'classes.ImageClass + ConceptDatasetClass',
    }


def released_full_image_input(path, model_cfg):
    dataset, identity = _released_dataset(path, model_cfg)
    return dataset[0], identity


def released_classifier_pixels(path, model_cfg):
    """Cache pre-normalization uint8 pixels, preserving the released transform.

    The classifier trainer applies normalization/augmentation to this cache.
    No alternate PIL interpolation or native-image crop is introduced here.
    """
    import torch
    dataset, identity = _released_dataset(path, model_cfg)
    segment = dataset.segments[0]
    if segment.max() <= 1:
        segment = (segment * 255).astype(np.uint8)
    tensor = dataset.resize(segment)
    pixels = (tensor * 255).round().to(dtype=torch.uint8)
    return pixels.permute(1, 2, 0).numpy(), identity


def save_released_presam_input(source, destination):
    """Lossless storage of ImageClass output, NEVER the classifier224 cache."""
    from classes import ImageClass
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError('Preserve existing input: '+str(destination))
    with Image.open(source) as original:
        if original.mode != 'RGB':
            raise ValueError('No implicit image-mode conversion')
        original_size = original.size
    image = ImageClass(str(source), max_shortest_side=300)
    expected_pixels = np.asarray(image.img_pil)
    image.img_pil.save(destination, format='PNG')
    with Image.open(destination) as saved:
        if saved.size != image.img_pil.size or not np.array_equal(np.asarray(saved), expected_pixels):
            raise ValueError('Lossless pre-SAM input changed pixels/geometry')
    return dict(original_size_wh=list(original_size),input_size_wh=list(image.img_pil.size),
                role='SAM_INPUT_NOT_CLASSIFIER_CACHE',
                preprocessing='released_ImageClass_short_side_cap300_LANCZOS_no_crop',
                extra_square_resize_before_sam=False)
