"""Synthetic local input/identity tests only; no licensed images, models or SSH."""
import gzip
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hpc import ace_inputs as a


def picture(path, value, mode='RGB'):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new(mode, (13, 17), value if mode == 'L' else (value, value//2, 255-value))
    image.save(path, format='PNG')


class ACEInputsTests(unittest.TestCase):
    def fixture(self, root):
        licensed = root/'licensed'
        private = root/'private'
        data = dict(class_name=a.CLASS_NAME, synset=a.SYNSET, seed=43,
                    licensed_source_root=str(licensed), source_dir=str(private),
                    ready_for_reference=True, validation_issues=[], training=[], validation=[])
        for split, count in [('training', 4), ('validation', 2)]:
            for i in range(count):
                name = (f'{a.SYNSET}_{i}.png' if split == 'training' else f'ILSVRC2012_val_{i:08d}.png')
                source = licensed/('train' if split == 'training' else 'val')/a.SYNSET/name
                picture(source, 20+i if split == 'training' else 130+i,
                        mode='L' if split == 'validation' and i == 1 else 'RGB')
                prepared = f'{i+1:04d}_{name}'
                dest = private/(a.CLASS_NAME if split == 'training' else 'val_imgs/'+a.CLASS_NAME+'_val')/prepared
                dest.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source) as image:
                    image.convert('RGB').save(dest)
                data[split].append(dict(source=str(source), sha256=a.sha256(source),
                    input_path=str(dest), input_sha256=a.sha256(dest), prepared_name=prepared,
                    compatibility_adjustment='fixture L to RGB' if split == 'validation' and i == 1 else None))
        other = licensed/'train/n10000000'
        for i in range(8):
            picture(other/f'n10000000_{i}.png', 60+i)
        picture(other/'gray.png', 12, mode='L')
        (other/'bad.jpg').write_bytes(b'not an image')
        manifest = root/'golden.json'
        manifest.write_text(json.dumps(data))
        config = dict(dataset_manifest=str(manifest), dataset_manifest_sha256=a.sha256(manifest),
                      licensed_root=str(licensed), execution_commit='a'*40)
        return config, data

    def small_counts(self):
        return patch.multiple(a, TARGET_COUNT=4, DISCOVERY_COUNT=2, VALIDATION_COUNT=2,
                              RANDOM_COUNT=5, SYNSET_COUNT=2)

    def test_slurm_guard_precedes_any_dataset_access(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(a.socket, 'gethostname', return_value='login'):
            with self.assertRaisesRegex(RuntimeError, 'Slurm compute'):
                a.run({'dataset_manifest': '/not-accessible'}, Path('/not-created'))

    def test_canonical_inventory_permutation_and_compression_preserve_global_rng(self):
        with tempfile.TemporaryDirectory() as tmp, self.small_counts():
            config, _ = self.fixture(Path(tmp))
            before = random.getstate()
            ids, counts = a.enumerate_candidates(Path(config['licensed_root'])/'train', lambda *x: None)
            self.assertEqual(ids, sorted(ids))
            self.assertEqual(sum(counts.values()), len(ids))
            self.assertIn('n10000000/bad.jpg', ids)  # no hidden extension filtering
            order = a.candidate_order(len(ids))
            expected = list(range(len(ids))); random.Random(43).shuffle(expected)
            self.assertEqual(order, expected)
            self.assertEqual(random.getstate(), before)
            one, two = Path(tmp)/'one.gz', Path(tmp)/'two.gz'
            a.write_candidate_ids(one, ids, order); a.write_candidate_ids(two, ids, order)
            self.assertEqual(one.read_bytes(), two.read_bytes())
            self.assertEqual(gzip.decompress(one.read_bytes()).decode().splitlines(), [ids[i] for i in order])

    def test_frozen_order_and_original_L_actual_RGB_mapping(self):
        with tempfile.TemporaryDirectory() as tmp, self.small_counts():
            config, original = self.fixture(Path(tmp))
            # Deliberately reverse frozen manifest order: never substitute sorted order.
            original['training'].reverse()
            Path(config['dataset_manifest']).write_text(json.dumps(original))
            config['dataset_manifest_sha256'] = a.sha256(config['dataset_manifest'])
            _, roles = a.verify_golden(config, Path(config['licensed_root']))
            self.assertEqual([r['input_path'] for r in roles['discovery50']],
                             [r['input_path'] for r in original['training'][:2]])
            converted = roles['validation50'][1]
            self.assertNotEqual(converted['sha256'], converted['input_sha256'])
            self.assertEqual(converted['input_path'], original['validation'][1]['input_path'])
            Path(original['training'][0]['input_path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'actual image'):
                a.verify_golden(config, Path(config['licensed_root']))

    def test_validation_identity_content_symlink_and_bad_format_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp, self.small_counts():
            config, original = self.fixture(Path(tmp)); train = Path(config['licensed_root'])/'train'
            validation = dict(ids=set(), basenames=set(), hashes={original['validation'][0]['sha256']})
            other = train/'n10000000'
            cases = [('gray.png', 'unsupported_mode:L'), ('bad.jpg', 'read_or_decode_error:UnidentifiedImageError')]
            (other/'copy.png').write_bytes(Path(original['validation'][0]['source']).read_bytes())
            cases.append(('copy.png', 'known_validation_content'))
            (other/'outside.png').symlink_to(original['validation'][0]['source'])
            cases.append(('outside.png', 'resolved_path_outside_training_root'))
            picture(other/'ILSVRC2012_val_12345678.png', 45)
            cases.append(('ILSVRC2012_val_12345678.png', 'validation_image_identity'))
            for name, reason in cases:
                with self.subTest(name=name):
                    result = a.inspect_candidate(other/name, 'n10000000/'+name, train, validation, 0)
                    self.assertIn(reason, result['reasons'])
            truncated = other/'truncated.jpg'
            Image.new('RGB', (40, 40)).save(truncated)
            truncated.write_bytes(truncated.read_bytes()[:-30])
            self.assertTrue(a.inspect_candidate(truncated, 'n10000000/truncated.jpg', train, validation, 0)['reasons'])

    def test_end_to_end_role_counts_skips_rng_and_owned_manifest(self):
        with tempfile.TemporaryDirectory() as tmp, self.small_counts(), patch.dict(os.environ, {'SLURM_JOB_ID': '123'}):
            root = Path(tmp); config, original = self.fixture(root)
            out = root/'output'; out.mkdir()
            (out/'launch_manifest.json').write_text('mutable worker file')
            before = random.getstate()
            with patch.object(a, '_assert_runtime', return_value='a'*40):
                result = a.run(config, out)
            self.assertEqual(before, random.getstate())
            manifest = json.loads((out/'ace_inputs_manifest.json').read_text())
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual({k: v['count'] for k, v in manifest['roles'].items()},
                             {'target400': 4, 'discovery50': 2, 'validation50': 2, 'random2000': 5})
            self.assertEqual(manifest['deferred_roles']['gradient'], 'DEFERRED_EFFECTIVE_FEATURE_POOL')
            self.assertNotIn('launch_manifest.json', manifest['files'])
            for relative, entry in manifest['files'].items():
                self.assertTrue(relative.startswith('ace_inputs/'))
                self.assertEqual(a.sha256(out/relative), entry['sha256'])
                self.assertLess(entry['bytes'], 256*1024**2)
            attempts = [json.loads(line) for line in (out/'ace_inputs/selection_attempts.jsonl').read_text().splitlines()]
            selected = json.loads((out/'ace_inputs/random2000.json').read_text())
            self.assertEqual([x['source_id'] for x in attempts if x['decision']=='selected'], [x['source_id'] for x in selected])
            self.assertEqual([x['selection_index'] for x in selected], list(range(5)))
            self.assertFalse(list((out/'ace_inputs').rglob('*.png')))
            with patch.object(a, '_assert_runtime', return_value='a'*40):
                with self.assertRaises(FileExistsError):
                    a.run(config, out)

    def test_overlap_records_keep_duplicate_content_and_role_members(self):
        row = dict(source='/train/n10000000/a.png', input_path='/train/n10000000/a.png',
                   sha256='a'*64, input_sha256='a'*64)
        duplicate = dict(row, source='/train/n10000000/b.png')
        records = a.overlap_records({'target400': [row], 'random2000': [row, duplicate]})
        self.assertEqual(records['intra_role']['random2000']['sha256'][0]['indices'], [0, 1])
        self.assertEqual(records['cross_role']['target400__random2000']['id'][0]['right_indices'], [0])
        self.assertEqual(records['cross_role']['target400__random2000']['sha256'][0]['right_indices'], [0, 1])

    def test_insufficient_candidates_preserve_failed_stage_and_attempts(self):
        with tempfile.TemporaryDirectory() as tmp, self.small_counts(), patch.dict(os.environ, {'SLURM_JOB_ID': '124'}):
            root = Path(tmp); config, _ = self.fixture(root)
            out = root/'failed'
            with patch.object(a, '_assert_runtime', return_value='a'*40), patch.object(a, 'RANDOM_COUNT', 100):
                with self.assertRaisesRegex(ValueError, 'Only .* eligible'):
                    a.run(config, out)
            manifest = json.loads((out/'ace_inputs_manifest.json').read_text())
            self.assertEqual(manifest['status'], 'FAILED')
            self.assertGreater(manifest['statistics']['skipped'], 0)
            self.assertIn('ace_inputs/skipped.jsonl', manifest['files'])
            self.assertTrue((out/'ace_inputs/selection_attempts.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
