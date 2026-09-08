import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

HPC = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, HPC / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

a = load('audit_training_candidates')
w = load('release_worker')
s = load('submit_release')
try:
    from PIL import Image
except ImportError:
    Image = None

@unittest.skipIf(Image is None, 'Run with the research environment for Pillow image tests')
class TrainingAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'imagenet'
        paths = []
        for split, count, offset in [('train', 1300, 0), ('val', 50, 5000)]:
            folder = self.root / split / 'n02099601'
            folder.mkdir(parents=True)
            for i in range(count):
                path = folder / f'{i:05d}.png'
                number = i + offset
                Image.new('RGB', (2, 2), (number % 251, number // 251, 254)).save(path)
                if split == 'train':
                    paths.append(path)
        order = random.Random(43).sample(paths, len(paths))
        # Two encountered grayscale images and a third outside the selected prefix.
        for index in (3, 399, 1100):
            Image.new('L', (2, 2), index % 251).save(order[index])
        self.val = sorted((self.root / 'val/n02099601').iterdir())
        Image.new('L', (2, 2), 222).save(self.val[1])
        self.order = order
        self.frozen = {'seed':43, 'synset':'n02099601', 'licensed_source_root':str(self.root),
            'candidate_filename_hashes':{'train':hashlib.sha256('\n'.join(p.name for p in paths).encode()).hexdigest()}}
        for key, selected in [('training', order[:400]), ('validation', self.val)]:
            self.frozen[key] = [{'source':str(p), 'sha256':a.digest(p), 'prepared_name':f'{i:04d}_{p.name}'}
                                for i,p in enumerate(selected,1)]
        self.fp = self.base / 'frozen.json'
        self.fp.write_text(json.dumps(self.frozen))
        self.report = self.base / 'audit'
        self.report.mkdir()

    def audit(self):
        a.audit(self.root, self.fp, a.digest(self.fp), self.report, previews=False)
        return json.loads((self.report / 'training_audit.json').read_text())

    def test_refill_full_population_counts_fixed_validation_and_publish_guard(self):
        manifest = self.audit()
        stats = manifest['statistics']
        self.assertEqual((stats['training_candidates'],stats['grayscale_count'],stats['skipped_grayscale_count']), (1300,3,2))
        self.assertEqual(stats['examined_for_selection'], 402)
        self.assertEqual([e['source'] for e in manifest['training']],
                         [str(p) for i,p in enumerate(self.order[:402]) if i not in (3,399)])
        self.assertEqual([e['source'] for e in manifest['validation']], [e['source'] for e in self.frozen['validation']])
        self.assertEqual([e['sha256'] for e in manifest['validation']], [e['sha256'] for e in self.frozen['validation']])
        self.assertEqual(len(manifest['validation_issues']),1)
        self.assertFalse(manifest['ready_for_reference'])
        self.assertEqual(len(manifest['candidates']),1300)
        self.assertEqual(manifest['skipped'][0]['skip_reasons'], ['grayscale:grayscale_mode','unsupported_mode:L'])
        ap = self.report / 'training_audit.json'
        destination = self.base / 'prepared'
        a.publish(ap, a.digest(ap), destination)
        self.assertEqual(len(list((destination/'golden_retriever').iterdir())),400)
        self.assertEqual((destination/'validation_images.txt').read_text(), (self.report/'validation_images.txt').read_text())
        with self.assertRaisesRegex(ValueError,'unresolved validation'):
            w.verify_inputs({'dataset_manifest':str(destination/'dataset_manifest.json'), 'source_dir':str(destination)})
        with self.assertRaises(ValueError):
            a.publish(ap,a.digest(ap),destination)

    def test_frozen_identity_and_population_changes_rejected(self):
        with self.assertRaisesRegex(ValueError,'identity changed'):
            a.audit(self.root,self.fp,'0'*64,self.report,previews=False)
        self.order[-1].unlink()
        with self.assertRaisesRegex(ValueError,'population changed'):
            self.audit()

    def test_validation_change_is_never_replaced(self):
        Image.new('RGB',(2,2),(40,50,60)).save(self.val[1])
        with self.assertRaisesRegex(ValueError,'validation content changed'):
            self.audit()
        self.assertFalse((self.report/'training_audit.json').exists())

    def test_cross_split_duplicate_is_skipped_and_order_continues(self):
        # A later refill candidate duplicates a fixed validation image.
        self.order[400].write_bytes(self.val[2].read_bytes())
        manifest = self.audit()
        self.assertEqual(manifest['statistics']['examined_for_selection'],403)
        self.assertEqual(manifest['statistics']['skipped_grayscale_count'],2)
        self.assertIn('duplicate_selected_or_validation_content_or_path',manifest['skipped'][-1]['skip_reasons'])
        self.assertFalse({e['sha256'] for e in manifest['training']} & {e['sha256'] for e in manifest['validation']})

    def test_exact_rgb_gray_and_decode_error_distinguished(self):
        path = self.base/'rgb.png'
        Image.new('RGB',(3,3),(40,40,40)).save(path)
        self.assertEqual(a.inspect_image(path)['grayscale_kind'],'exact_achromatic_RGB')
        path.write_bytes(b'broken')
        self.assertTrue(a.inspect_image(path)['image_error'].startswith('decode_error:'))

    def test_changed_input_before_publish_rejected(self):
        manifest=self.audit()
        Path(manifest['training'][0]['source']).write_bytes(b'changed')
        ap=self.report/'training_audit.json'
        with self.assertRaisesRegex(ValueError,'Audited input changed'):
            a.publish(ap,a.digest(ap),self.base/'prepared')
        self.assertFalse((self.base/'prepared').exists())

class TrainingLauncherTests(unittest.TestCase):
    def test_audit_launcher_binds_manifest_and_cpu_only(self):
        args = ['--commit','a5e7f843856d17b7647f8af6571319b45719f669','--mode','audit-training',
            '--dataset-root','/scratch/licenseddata/imagenet/imagenet-1k','--seed','43',
            '--frozen-selection','/scratch/frozen.json','--frozen-selection-sha256','a'*64,
            '--cpus','1','--memory','2G','--time','00:05:00','--partition','general','--qos','debug']
        plan,worker=s.build_plan(s.arguments(args))
        script=s.render(plan,worker)
        self.assertEqual(plan['selection_sha256'],'a'*64)
        self.assertNotIn('#SBATCH --gres',script)
        self.assertLess(len(script.encode()),64000)
        import subprocess
        subprocess.run(['bash','-n'],input=script,text=True,check=True)
        with self.assertRaises(ValueError):
            s.build_plan(s.arguments(args+['--gpu','l40s:1']))

class CompatibilityTests(unittest.TestCase):
    setUp = TrainingAuditTests.setUp
    audit = TrainingAuditTests.audit

    @unittest.skipIf(Image is None, 'Requires Pillow')
    def test_one_png_adjustment_preserves_lists_and_checks_originals(self):
        from unittest.mock import patch
        prep = load('reference_preparation')
        self.audit()
        ap = self.report/'training_audit.json'
        original = self.base/'original'
        a.publish(ap, a.digest(ap), original)
        mp = original/'dataset_manifest.json'
        destination = self.base/'compatible'
        with patch.object(prep,'L_ID',self.val[1].name), patch.object(prep,'L_SHA',a.digest(self.val[1])):
            result=prep.compatibility(mp,a.digest(mp),destination)
        self.assertEqual(result['conversions'],1)
        for split in ('training','validation'):
            self.assertEqual((destination/(split+'_images.txt')).read_bytes(),(original/(split+'_images.txt')).read_bytes())
        mapping=json.loads((destination/'input_mapping.json').read_text())
        changed=[e for e in mapping if e['adjustment']]
        self.assertEqual(len(changed),1)
        self.assertTrue(changed[0]['input_path'].endswith('.png'))
        with Image.open(changed[0]['input_path']) as actual, Image.open(self.val[1]) as source:
            self.assertTrue(all(c.tobytes()==source.tobytes() for c in actual.split()))
        config={'source_dir':str(destination),'dataset_manifest':str(destination/'dataset_manifest.json'),
                'dataset_manifest_sha256':result['dataset_manifest_sha256']}
        self.assertEqual(w.verify_inputs(config)['validation_count'],50)
        self.val[1].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'mapping changed'):
            w.verify_inputs(config)

if __name__=='__main__':
    unittest.main()
