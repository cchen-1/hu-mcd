"""Raw-HAM input adapter preparation; no training, downloads or submission.

Delegate geometry and tensor arithmetic to the released HU-MCD classes instead
of introducing a second resize implementation. This helper is not yet wired to
an execution launcher. Remote invocation must occur inside a Slurm allocation.
"""
from pathlib import Path

import numpy as np
from PIL import Image


def released_full_image_input(path, model_cfg):
    """Return the unmasked classifier tensor and pre-SAM geometry metadata.

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
    return dataset[0], {
        'source_path': str(path),
        'original_size_wh': list(original_size),
        'pre_sam_size_wh': list(image.img_pil.size),
        'classifier_size_chw': list(model_cfg['input_size']),
        'max_shortest_side': 300,
        'cropping_mode': 0,
        'implementation': 'classes.ImageClass + ConceptDatasetClass',
    }
