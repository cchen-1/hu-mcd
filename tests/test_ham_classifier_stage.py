"""No model execution: protect the new geometry/budget and existing authorization gate."""
import copy
import json
from pathlib import Path
import unittest
from hpc.medical_protocol import validate_config, digest

class HamClassifierStageTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('configs/medical/ham-classifier.approved.json').read_text())

    def test_old_and_new_protocols_keep_separate_budgets(self):
        validate_config(self.config, 'medical-classifier', approved=True)
        old = json.loads(Path('configs/medical/medical-classifier.approved.json').read_text())
        validate_config(old, 'medical-classifier', approved=True)
        self.config['resources'] = old['resources']
        with self.assertRaises(ValueError):
            validate_config(self.config, 'medical-classifier', approved=True)

    def test_geometry_and_processing_ceiling_cannot_be_relabelled(self):
        for mutate in (lambda p:p['input_geometry'].update(extra_square_resize_before_sam=True),
                       lambda p:p.update(work_seconds_ceiling=3601)):
            c = copy.deepcopy(self.config); mutate(c['protocol'])
            c['protocol_sha256'] = digest(c['protocol'])
            c['authorization']['approved_protocol_sha256'] = c['protocol_sha256']
            with self.assertRaises(ValueError):
                validate_config(c, 'medical-classifier', approved=True)

    def test_input_tampering_breaks_approval(self):
        self.config['protocol']['inputs']['source_npz']['sha256'] = '0'*64
        with self.assertRaises(ValueError):
            validate_config(self.config, 'medical-classifier', approved=True)
