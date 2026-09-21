"""Explicit final binary-mask geometry; no model, sampling or scientific defaults.

The caller must supply a previously approved operation/radius. This module does
not authorize experiments or reapply the source conditional erosion.
"""
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def intervene(mask, operation, radius):
    array = np.asarray(mask)
    if array.ndim != 2 or not np.isin(array, [0, 1]).all():
        raise ValueError('Expected one finite binary HxW effective input mask')
    if type(radius) is not int or radius < 0:
        raise ValueError('Radius must be an explicit nonnegative integer')
    if operation == 'identity':
        if radius != 0:
            raise ValueError('Identity requires radius 0')
        return array.astype(bool, copy=True)
    if operation not in ('erosion', 'dilation') or radius == 0:
        raise ValueError('Expected erosion/dilation with positive radius')
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    disk = x*x + y*y <= radius*radius
    fn = binary_erosion if operation == 'erosion' else binary_dilation
    return fn(array.astype(bool), structure=disk, border_value=0)


class FinalMaskInterventionDataset:
    """Wrap the original dataset AFTER its final resize/binarization.

    Preserves original RGB tensors, IDs, indexing and mode. Empty masks are
    returned explicitly for the execution caller to account for before CNN use;
    this wrapper never removes rows or substitutes features.
    """
    def __init__(self, original, operation, radius):
        intervene(np.ones((1, 1), dtype=bool), operation, radius)
        self.original, self.operation, self.radius = original, operation, radius

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        import torch
        rgb, mask, mode = self.original[index]
        if mask.ndim != 3 or mask.shape[0] != 1:
            raise ValueError('Expected original 1xHxW model input mask')
        changed = intervene(mask.detach().cpu().numpy()[0], self.operation, self.radius)
        result = torch.as_tensor(changed, dtype=mask.dtype, device=mask.device).unsqueeze(0)
        return rgb, result, mode
