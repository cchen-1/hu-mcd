"""Synthetic input checks only; no medical images, GPU, SAM or model weights."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from hpc.ham_input_preparation import released_full_image_input

CFG = dict(input_size=(3,224,224), mean=(.485,.456,.406), std=(.229,.224,.225))


class RawHamInputTests(unittest.TestCase):
    def test_large_and_small_geometry_and_reference_tensor(self):
        # Explicit reference pins the intended released two-resize semantics.
        cases = [((600,450),(400,300)), ((450,600),(300,400)),
                 ((180,240),(180,240)), ((224,224),(224,224))]
        rng = np.random.RandomState(43)
        with tempfile.TemporaryDirectory() as tmp:
            for (w,h), expected_size in cases:
                path = Path(tmp)/f'{w}-{h}.png'
                pixels = rng.randint(0,256,(h,w,3),dtype=np.uint8)
                Image.fromarray(pixels).save(path)
                before = path.read_bytes()
                actual, identity = released_full_image_input(path, CFG)
                self.assertEqual(identity['pre_sam_size_wh'], list(expected_size))
                reference = Image.fromarray(pixels)
                if min(w,h)>300:
                    reference = reference.resize(expected_size, Image.Resampling.LANCZOS)
                array = np.array(reference,dtype=np.float32)/255.
                reference = transforms.Compose([
                    transforms.ToPILImage(), transforms.Resize((224,224)),
                    transforms.ToTensor(), transforms.Normalize(CFG['mean'],CFG['std']),
                ])((array*255).astype(np.uint8))
                self.assertTrue(torch.equal(actual, reference))
                self.assertEqual(path.read_bytes(),before)

    def test_grayscale_requires_explicit_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gray.png'
            Image.fromarray(np.zeros((20,30),dtype=np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError,'Unexpected input mode L'):
                released_full_image_input(path,CFG)
