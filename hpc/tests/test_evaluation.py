"""Focused cache-identity and upstream protocol regression checks; no GPU/SAM."""
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hpc.evaluate_reference import (CachedScores, audited_cache, checked_file,
    predict_stream, save_masks, sha256, under, upstream_scores, validate_alignment,
    verify_sources, random_signature, random_code_identity, SOURCE_FILES, SOURCE_JOB, SOURCE_COMMIT)


class CacheIdentityTests(unittest.TestCase):
    def test_hash_tampering_missing_identity_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / 'cache.npy'
            file.write_bytes(b'original')
            digest = sha256(file)
            source = {'inventory': {'cache.npy': {'sha256': digest, 'bytes': 8}}}
            config = {'source_cache_dir': tmp}
            self.assertEqual(audited_cache(config, source, 'cache.npy'), file)
            file.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                audited_cache(config, source, 'cache.npy')
            with self.assertRaisesRegex(ValueError, 'absent'):
                audited_cache(config, source, 'missing.npy')
            for name in ('../cache.npy', '/etc/passwd'):
                with self.assertRaises(ValueError):
                    under(root, name)
            (root/'link').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):
                under(root, 'link')

    def test_pinned_source_manifest_chain_rejects_rehashed_wrong_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'scientific').mkdir()
            precision = {'cudnn_allow_tf32': True}
            dataset = dict(class_name='golden_retriever',
                training=[{'input_sha256': 'a'*64} for _ in range(400)],
                validation=[{'input_sha256': 'b'*64} for _ in range(50)])
            (root/'dataset.json').write_text(json.dumps(dataset))
            (root/'weights').write_bytes(b'weights')
            resolved = dict(precision=precision, batch_size=8, validation_images=50,
                train_images=400, class_name='golden_retriever',
                dataset_manifest_sha256=sha256(root/'dataset.json'),
                resnet_checkpoint_sha256=sha256(root/'weights'))
            (root/'resolved_config.json').write_text(json.dumps(resolved))
            summary = dict(status='PASS', run_id=SOURCE_JOB, git_commit=SOURCE_COMMIT,
                resolved_config_sha256=sha256(root/'resolved_config.json'))
            manifest = dict(summary, git_dirty=False, precision=precision,
                input_files={k:[{'sha256':r['input_sha256']} for r in dataset[k]]
                             for k in ('training','validation')},
                source_sha256={k:sha256(Path(__file__).resolve().parents[2]/k) for k in
                    ('benchmark_methods.py','classes.py','concept_explainer.py',
                     'utils/utils_general.py','utils/utils_mcd.py')})
            (root/'summary.json').write_text(json.dumps(summary))
            (root/'run_manifest.json').write_text(json.dumps(manifest))
            for name in SOURCE_FILES[3:]:
                (root/name).write_text('{}')
            (root/'audit.json').write_text(json.dumps(dict(status='PASS', source_job=SOURCE_JOB,
                audit_job='28211056', cache_inventory=[])))
            config = dict(resolved, source_run_dir=str(root), source_cache_dir=str(root),
                dataset_manifest=str(root/'dataset.json'), resnet_checkpoint=str(root/'weights'),
                source_audit_path=str(root/'audit.json'), source_audit_sha256=sha256(root/'audit.json'),
                source_files_sha256={k:sha256(root/k) for k in SOURCE_FILES},
                random_seeds={'sdc':43,'ssc':43})
            verify_sources(config)
            # Same verification path supports a separately pinned non-Golden discovery.
            new_job, new_commit = '28214983', 'f'*40
            for item in (summary, manifest):
                item.update(run_id=new_job, git_commit=new_commit)
            dataset['class_name']=resolved['class_name']=config['class_name']='airliner'
            (root/'dataset.json').write_text(json.dumps(dataset))
            resolved['dataset_manifest_sha256']=config['dataset_manifest_sha256']=sha256(root/'dataset.json')
            (root/'resolved_config.json').write_text(json.dumps(resolved))
            for item,name in [(summary,'summary.json'),(manifest,'run_manifest.json')]:
                item['resolved_config_sha256']=sha256(root/'resolved_config.json')
                (root/name).write_text(json.dumps(item))
            audit=dict(status='PASS',source_job=new_job,expected_commit=new_commit,code_commit=new_commit,
                class_name='airliner',config_sha256=sha256(root/'resolved_config.json'),
                dataset_sha256=sha256(root/'dataset.json'),source_cache=str(root),cache_inventory=[],
                model_identity={'resnet_sha256':resolved['resnet_checkpoint_sha256'],'precision':precision},
                splits={k:dict(images=n,masks_features_mapping='PASS',saved_reconstruction_checks={'status':'PASS'})
                        for k,n in [('training',400),('validation',50)]})
            (root/'audit.json').write_text(json.dumps(audit))
            config.update(source_job=new_job,source_commit=new_commit,source_audit_schema='ten-class-visual-v1',
                          source_audit_sha256=sha256(root/'audit.json'),
                          source_files_sha256={k:sha256(root/k) for k in SOURCE_FILES})
            self.assertEqual(verify_sources(config)['source_job'],new_job)
            config['class_name']='golden_retriever'
            with self.assertRaisesRegex(ValueError,'audit identity'):
                verify_sources(config)
            config['class_name']='airliner'
            manifest['run_id'] = 'other-run'
            (root/'run_manifest.json').write_text(json.dumps(manifest))
            config['source_files_sha256']['run_manifest.json'] = sha256(root/'run_manifest.json')
            with self.assertRaisesRegex(ValueError, 'identity/status'):
                verify_sources(config)

    def test_real_multiclass_manifests_missing_masking_hashes(self):
        base=Path(__file__).resolve().parents[2]/'artifacts/bunya/ten-class-review/evidence'
        if not base.exists():self.skipTest('Private audit fixtures absent')
        for job in ('28214893','28214894','28214896','28214897','28214983','28214984','28215019','28215020','28215042'):
            folder=base/job
            manifest=json.loads((folder/'run/run_manifest.json').read_text())
            audit=json.loads((folder/'audit.json').read_text())
            self.assertNotIn('input_masking/resnet.py',manifest['source_sha256'])
            source=dict(manifest=manifest,audit=audit,audit_schema='ten-class-visual-v1')
            cfg=json.loads((folder/'run/resolved_config.json').read_text())
            cfg['random_seeds']={'sdc':43,'ssc':43}
            rows=json.loads((folder/'inputs/dataset_manifest.json').read_text())['validation']
            model_cfg=json.loads((folder/'run/scientific/discovery.json').read_text())['model_default_cfg']
            sig=random_signature(cfg,model_cfg,rows,random_code_identity(source))
            self.assertEqual(len(sig['sha256']),64)
            audit['core_sha256']['input_masking/resnet.py']='0'*64
            with self.assertRaisesRegex(ValueError,'input-mask code'):
                random_code_identity(source)

    def test_random_reuse_requires_complete_input_and_protocol_identity(self):
        cfg=dict(resnet_checkpoint_sha256='a'*64,model_name='resnet50',max_shortest_side=300,
                 precision={'cudnn_allow_tf32':True},batch_size=8,random_seeds={'sdc':43,'ssc':43})
        rows=[dict(source='/remote/image.JPEG',input_sha256='b'*64)]
        code={k:'c'*64 for k in ('benchmark_methods.py','classes.py','utils/utils_general.py',
                                 'input_masking/resnet.py','input_masking/sal_layers.py')}
        original=random_signature(cfg,{'crop_pct':.95},rows,code)
        moved=[dict(source='/other/image.JPEG',input_sha256='b'*64)]
        self.assertEqual(original,random_signature(cfg,{'crop_pct':.95},moved,code))
        for changed in (dict(cfg,batch_size=64),dict(cfg,max_shortest_side=224),
                        dict(cfg,random_seeds={'sdc':42,'ssc':43}),dict(cfg,resnet_checkpoint_sha256='d'*64)):
            self.assertNotEqual(original,random_signature(changed,{'crop_pct':.95},rows,code))
        self.assertNotEqual(original,random_signature(cfg,{'crop_pct':.95},
                            [dict(source='/remote/another.JPEG',input_sha256='e'*64)],code))

    def test_mask_and_feature_order_including_zero_rows(self):
        raw = np.array([[1., 2.], [0., 0.], [3., 4.]])
        logits = np.zeros((3, 1000))
        masks = np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 0]]], dtype=np.float32)
        retained = {'features': raw[[0, 2]], 'logits': logits[[0, 2]]}
        records = [dict(image_index=0, segment_index=i, image_path='image',
            mask_shape=[2, 2], mask_dtype='float32',
            mask_sha256=hashlib.sha256(masks[j].tobytes()).hexdigest()) for i,j in enumerate((0, 2))]
        np.testing.assert_array_equal(validate_alignment(raw, logits, [3], retained, records,
                                      [masks], ['image']), [True, False, True])
        with self.assertRaisesRegex(ValueError, 'ordering'):
            validate_alignment(raw[[2, 1, 0]], logits, [3], retained, records, [masks], ['image'])
        with self.assertRaisesRegex(ValueError, 'mask ordering'):
            validate_alignment(raw, logits, [3], retained, records, [masks[[2, 1, 0]]], ['image'])
        with self.assertRaisesRegex(ValueError, 'counts'):
            validate_alignment(raw, logits, [2], retained, records, [masks], ['image'])

    def test_packed_masks_are_lossless_not_float_thresholded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'masks.npz'
            masks = np.random.RandomState(43).randint(0, 2, (3, 13, 17))
            save_masks(path, masks)
            with np.load(path) as saved:
                restored = np.unpackbits(saved['bits'], count=masks.size).reshape(saved['shape'])
            np.testing.assert_array_equal(masks, restored)
            with self.assertRaisesRegex(ValueError, 'binary'):
                save_masks(path, masks + .1)


class UpstreamProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import benchmark_methods
        import classes
        from concept_explainer import ConceptExplainer
        cls.benchmark, cls.classes, cls.Explainer = benchmark_methods, classes, ConceptExplainer

    def image(self, root, shape=(13, 17)):
        from PIL import Image
        path = Path(root) / 'image.png'
        Image.fromarray(np.full((*shape, 3), 128, dtype=np.uint8)).save(path)
        return self.classes.ImageClass(str(path), min(shape))

    def test_humcd_endpoint_break_zero_tie_and_normalization_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = self.image(tmp)
            image.segments = [self.classes.SegmentClass(np.ones((13, 17)), image)]
            image.segments[0].model_act = np.zeros(2)
            adapter = CachedScores(SimpleNamespace(concept_bases=[np.array([[1., 0.]])],
                max_shortest_side=13), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)))
            # C0 owns the zero row, hence the full mask would immediately reach
            # the endpoint; original benchmark breaks BEFORE adding that state.
            for mode, initial in [('sdc', 1), ('ssc', 0)]:
                result = self.benchmark.iter_mask_imgs_humcd(adapter, [image], mode)
                self.assertEqual(len(result[0].segments), 1)
                self.assertTrue(np.all(result[0].segments[0].mask == initial))
            with self.assertRaisesRegex(ValueError, 'normalization'):
                adapter.concept_activations(np.zeros((1,2)), [1], False)
            with self.assertRaisesRegex(ValueError, 'identity'):
                adapter.concept_relevances(np.ones((1,2)))

    def test_random_grid_seed_order_border_and_endpoint_are_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = self.image(tmp)
            previous = random.getstate()
            try:
                for mode in ('sdc', 'ssc'):
                    random.seed(43)
                    result = self.benchmark.iter_mask_imgs_rdm([image], mode)[0]
                    self.assertEqual(len(result.segments), 101)
                    order = list(range(100))
                    random.Random(43).shuffle(order)
                    for step, cell in enumerate(order):
                        difference = np.abs(result.segments[step+1].mask - result.segments[step].mask.astype(float))
                        expected = np.zeros((13, 17))
                        expected[cell//10, cell%10] = 1
                        np.testing.assert_array_equal(difference, expected)
                    final = result.segments[-1].mask
                    self.assertEqual(int(final.sum()), 121 if mode == 'sdc' else 100)
            finally:
                random.setstate(previous)

    def test_strict_greater_than_75_percent_no_padding(self):
        averages, std = self.benchmark.calc_avg_and_std([[1, 0], [1, 1], [0, 1], [0]], 4)
        np.testing.assert_array_equal(averages, [.5])
        np.testing.assert_array_equal(std, [.5])

    def test_cached_scores_equal_unmodified_upstream_including_float32_normalization(self):
        import torch
        from utils import utils_general
        explainer = self.Explainer.__new__(self.Explainer)
        explainer.concept_bases = [np.array([[1., 0., 0.]]), np.array([[.6, .8, 0.]])]
        explainer.compl_basis = np.array([[0., 0., 1.]])
        explainer.target_class = 'golden_retriever'
        explainer.max_shortest_side = 13
        weight = torch.zeros((1000, 3))
        weight[207] = torch.tensor([.1, -2., .3])
        explainer.model = SimpleNamespace(fc=SimpleNamespace(weight=weight))
        acts = np.array([[1.,2.,3.], [0.,0.,0.], [-1.,3.,1.], [3.,2.,1.]], dtype=np.float32)
        with patch.object(utils_general, 'get_imagenet_class_index', return_value=207):
            scores, relevance = upstream_scores(explainer, acts, [3, 1])
            expected = explainer.concept_activations(acts, [3, 1], norm_batch=True, n_jobs=1)
            expected_rel = explainer.concept_relevances(acts, n_jobs=1)
        np.testing.assert_array_equal(scores, expected)
        np.testing.assert_array_equal(relevance, expected_rel)
        self.assertEqual(scores[1].argmax(), 0)
        with self.assertRaisesRegex(ValueError, 'all-zero image'):
            upstream_scores(explainer, acts[1:2], [1])

    def test_stream_preserves_global_batch_boundaries_and_full_upstream_logits(self):
        import torch
        from torch.utils.data import DataLoader
        from utils import utils_general
        with tempfile.TemporaryDirectory() as tmp:
            first = self.image(tmp)
            second = self.classes.ImageClass(first.filename, 13)
            for image, count in [(first, 5), (second, 6)]:
                image.segments = [self.classes.SegmentClass(np.full((13,17), i%2), image)
                                  for i in range(count)]
            class BatchSensitiveModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.global_pool = torch.nn.Identity()
                    self.default_cfg = dict(input_size=(3,8,8), mean=(.5,)*3, std=(.5,)*3)
                def forward(self, inputs):
                    pixels, masks, mode = inputs
                    features = self.global_pool((pixels*masks).mean((2,3)))
                    # A batch-size-dependent term makes accidental image-local
                    # grouping observable, analogous to the TF32 regression.
                    return (features.sum(1) + len(pixels)).reshape(-1,1).expand(-1,1000)
            model = BatchSensitiveModel()
            with patch.object(utils_general, 'DEVICE', 'cpu'):
                checkpoints = []
                actual, groups = predict_stream(model, iter(enumerate([first, second])), 8,
                    lambda *a: None, on_batch=lambda values, ids: checkpoints.append((values.copy(), ids)))
                dataset = self.classes.ConceptDatasetClass([first, second], model.default_cfg,
                    cropping_mode=0, use_masks=True, masking_mode=1, erosion_threshold=1.0)
                loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=utils_general.custom_collate)
                _, expected = utils_general.compute_activations(model, 'global_pool', loader)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(np.concatenate([c[0] for c in checkpoints]), expected)
            self.assertEqual([len(g) for g in groups], [8,3])
            self.assertEqual(groups[0][-1], [1,2])
            self.assertEqual(groups[1][0], [1,3])


if __name__ == '__main__':
    unittest.main()
