"""Strict L-compatibility publication tests using private synthetic image fixtures."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from hpc import multiclass_inputs as publisher
from hpc.audit_training_candidates import inspect_image
from utils.run_tracking import sha256


class MulticlassCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.dataset = self.base/'licensed'
        self.record = dict(class_name='beach_wagon', synset='n02814533', seed=43,
            licensed_source_root=str(self.dataset), sampling='unchanged audited order',
            candidate_filename_sha256='a'*64, candidate_order_sha256='b'*64,
            statistics={}, skipped=[], scene_review='PENDING', issues=[], training=[], validation=[])
        for split, count, start in [('training',400,0), ('validation',50,400)]:
            folder = self.dataset/split
            folder.mkdir(parents=True)
            for i in range(count):
                path = folder/f'image_{start+i:04d}.png'
                if split == 'validation' and i == 7:
                    image = Image.frombytes('L', (5,3), bytes(range(15)))
                else:
                    image = Image.new('RGB', (5,3), (17, (start+i)%256, (start+i)//256))
                image.save(path)
                entry = dict(inspect_image(path), prepared_name=f'{i+1:04d}_{path.name}')
                self.record[split].append(entry)
                if entry['image_error']:
                    self.record['issues'].append(dict(kind='validation_format', entry=copy.deepcopy(entry)))
        self.l_entry = self.record['validation'][7]
        self.audit = self.base/'selection_audit.json'
        self.config = dict(class_name='beach_wagon', selection_audit=str(self.audit),
            prepared_root=str(self.base/'private-inputs'), scene_review_evidence='approved keep audited400 unchanged',
            validation_compatibility_allowlist=[dict(source_basename=Path(self.l_entry['source']).name,
                                                   source_sha256=self.l_entry['sha256'])])
        self.save_audit()

    def save_audit(self):
        self.audit.write_text(json.dumps(self.record))
        self.config['selection_audit_sha256'] = sha256(self.audit)

    def publish(self):
        with patch.dict(os.environ, SLURM_JOB_ID='synthetic-test'):
            return publisher.publish(self.config, self.base/'worker-output')

    def test_private_lossless_png_keeps_50_ids_and_400_training_inputs(self):
        original_audit = self.audit.read_bytes()
        original_validation = copy.deepcopy(self.record['validation'])
        original_training = copy.deepcopy(self.record['training'])
        result = self.publish()
        dest = Path(self.config['prepared_root'])
        manifest = json.loads(Path(result['dataset_manifest']).read_text())
        mapping = json.loads((dest/'input_mapping.json').read_text())
        changes = [r for r in mapping if r['adjustment']]
        self.assertEqual(len(changes), 1)
        changed = changes[0]
        self.assertEqual(changed['source_basename'], Path(self.l_entry['source']).name)
        self.assertEqual(changed['original_prepared_name'], self.l_entry['prepared_name'])
        self.assertEqual(changed['prepared_name'], self.l_entry['prepared_name']+'.rgb.png')
        self.assertEqual(changed['source_sha256'], sha256(self.l_entry['source']))
        self.assertEqual(changed['input_sha256'], sha256(changed['input_path']))
        self.assertTrue(changed['adjustment']['pixel_equality_verified'])
        self.assertTrue(changed['adjustment']['reason'])
        self.assertEqual(manifest['input_mapping_sha256'], sha256(dest/'input_mapping.json'))
        actual_path = Path(changed['input_path'])
        self.assertFalse(actual_path.is_symlink())
        with Image.open(self.l_entry['source']) as original, Image.open(actual_path) as actual:
            self.assertEqual((original.mode, actual.mode, actual.format), ('L', 'RGB', 'PNG'))
            self.assertEqual(original.size, actual.size)
            self.assertTrue(all(channel.tobytes() == original.tobytes() for channel in actual.split()))
        self.assertEqual([e['source'] for e in manifest['validation']], [e['source'] for e in original_validation])
        self.assertEqual([e['sha256'] for e in manifest['validation']], [e['sha256'] for e in original_validation])
        self.assertEqual((dest/'validation_images.txt').read_text(), '\n'.join(e['source'] for e in original_validation)+'\n')
        self.assertEqual((dest/'validation_original_prepared_names.txt').read_text(),
                         '\n'.join(e['prepared_name'] for e in original_validation)+'\n')
        self.assertEqual(len(list((dest/'val_imgs/beach_wagon_val').iterdir())), 50)
        self.assertEqual(manifest['resolved_validation_issues'], self.record['issues'])
        self.assertEqual(manifest['validation_issues'], [])
        for before, after in zip(original_training, manifest['training']):
            for key in before:
                self.assertEqual(before[key], after[key])
            self.assertIsNone(after['compatibility_adjustment'])
            self.assertTrue(Path(after['input_path']).is_symlink())
            self.assertEqual(sha256(after['input_path']), before['sha256'])
        self.assertEqual(self.audit.read_bytes(), original_audit)
        self.assertEqual(sha256(result['dataset_manifest']), result['dataset_manifest_sha256'])
        with self.assertRaisesRegex(ValueError, 'new private'):
            self.publish()

    def test_exact_allowlist_rejects_wrong_id_hash_mode_count_and_other_issues(self):
        valid = self.config['validation_compatibility_allowlist'][0]
        cases = [[], [dict(valid, source_sha256='0'*64)],
                 [dict(valid, source_basename='not_selected.JPEG')], [valid, valid],
                 [dict(valid, source_basename='../escape')], [dict(valid, extra='unapproved')]]
        for entries in cases:
            with self.subTest(allowlist=entries), self.assertRaises(ValueError):
                publisher.validation_compatibility(self.record, dict(self.config,
                    validation_compatibility_allowlist=entries))
        for mode in ('RGB', 'LA', 'I;16'):
            record = copy.deepcopy(self.record)
            record['validation'][7]['mode'] = mode
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                publisher.validation_compatibility(record, self.config)
        record = copy.deepcopy(self.record)
        record['issues'].append({'kind':'training_input_invalid_or_duplicate'})
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            publisher.validation_compatibility(record, self.config)
        record = copy.deepcopy(self.record)
        record['class_name'] = 'airliner'
        with self.assertRaisesRegex(ValueError, 'class/count'):
            publisher.validation_compatibility(record, self.config)
        # An allowlisted RGB entry is not an authorization to replace a valid image.
        rgb = self.record['validation'][0]
        with self.assertRaisesRegex(ValueError, 'hash/mode/ID'):
            publisher.validation_compatibility(self.record, dict(self.config,
                validation_compatibility_allowlist=[dict(source_basename=Path(rgb['source']).name,
                                                        source_sha256=rgb['sha256'])]))

    def test_changed_source_and_unreviewed_training_refuse_publication(self):
        with self.assertRaisesRegex(ValueError, 'scene review'):
            with patch.dict(os.environ, SLURM_JOB_ID='test'):
                publisher.publish(dict(self.config, scene_review_evidence=None), self.base)
        Image.new('L', (5,3), 251).save(self.l_entry['source'])
        with self.assertRaisesRegex(ValueError, 'Changed'):
            self.publish()
        self.assertFalse(Path(self.config['prepared_root']).exists())
        with self.assertRaisesRegex(ValueError, 'source hash'):
            publisher.lossless_l_to_rgb(self.l_entry, self.base/'unexpected.png')

    def test_pixel_equality_failure_aborts_private_publication_atomically(self):
        wrong = Image.new('RGB', (5,3), (99,99,99))
        with patch.object(Image.Image, 'convert', return_value=wrong):
            with self.assertRaisesRegex(ValueError, 'pixel verification'):
                self.publish()
        self.assertFalse(Path(self.config['prepared_root']).exists())
        self.assertFalse(list(self.base.glob('.publish-*')))
        self.assertEqual(sha256(self.l_entry['source']), self.l_entry['sha256'])

    def test_unaffected_rgb_publication_retains_existing_layout_and_schema(self):
        Image.new('RGB', (5,3), (201,202,203)).save(self.l_entry['source'])
        self.record['validation'][7] = dict(inspect_image(Path(self.l_entry['source'])),
                                           prepared_name=self.l_entry['prepared_name'])
        self.record.update(issues=[], scene_review='NOT_NEEDED')
        self.config.pop('validation_compatibility_allowlist')
        self.save_audit()
        result = self.publish()
        manifest = json.loads(Path(result['dataset_manifest']).read_text())
        self.assertEqual(manifest['schema_version'], 4)
        self.assertNotIn('resolved_validation_issues', manifest)
        self.assertFalse((Path(self.config['prepared_root'])/'input_mapping.json').exists())
        self.assertTrue(all(Path(e['input_path']).is_symlink() for e in manifest['validation']))
        self.assertEqual([e['prepared_name'] for e in manifest['validation']],
                         [e['prepared_name'] for e in self.record['validation']])


if __name__ == '__main__':
    unittest.main()
