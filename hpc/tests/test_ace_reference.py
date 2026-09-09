"""ACE-R scientific instrumentation checks; synthetic local CPU data only."""
import inspect
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from hpc import ace_reference as a


def segments(X,role='random'):
    result=[]
    for i,x in enumerate(X):
        row=dict(id=f'{role}/{i}',input=dict(source=f'/training/{i}.png',input_path=f'/training/{i}.png',input_sha256=f'{i:064x}'))
        result.append(SimpleNamespace(model_act=x,org_img=SimpleNamespace(filename=f'img{i}'),ace_row=i,ace_record=row))
    return result


def tiny_model():
    import torch
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.global_pool=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1,1)),torch.nn.Flatten(),torch.nn.Linear(3,2048))
            self.fc=torch.nn.Linear(2048,1000);self.batches=[]
            self.default_cfg={'input_size':(3,224,224),'mean':(.485,.456,.406),'std':(.229,.224,.225)}
            with torch.no_grad():
                self.global_pool[2].weight.fill_(.02);self.global_pool[2].bias.fill_(.1)
                self.fc.weight.copy_(torch.linspace(-.01,.01,1000)[:,None].expand(1000,2048));self.fc.bias.zero_()
        def forward(self,x):self.batches.append(len(x));return self.fc(self.global_pool(x))
    return Model().eval()


def images(tmp,n):
    from PIL import Image
    rows=[]
    for i in range(n):
        p=Path(tmp)/f'{i}.png';Image.fromarray(np.random.default_rng(i).integers(0,256,(23,31,3),dtype=np.uint8)).save(p)
        rows.append(dict(source=str(p),input_path=str(p),input_sha256=a.sha256(p)))
    return rows


class ACEReferenceTests(unittest.TestCase):
    def test_cloned_roles_do_not_consume_any_formal_rng(self):
        import torch
        random.seed(43);np.random.seed(43);torch.manual_seed(43)
        before=a.rng_snapshot();plan=a.clone_roles(before['python'],range(2000),list(range(400)))
        self.assertEqual(a.digest(a.rng_snapshot()),a.digest(before))
        self.assertEqual(random.sample(list(range(2000)),50),plan['control_indices'])
        actual=list(range(400));random.shuffle(actual)
        self.assertEqual(actual,plan['gradient_candidate_order'])
        self.assertEqual(a.tuples(random.getstate()),a.tuples(plan['python_after_gradient']))
        self.assertEqual(len(plan['gradient_indices']),50)
        self.assertEqual(a.digest(a.rng_snapshot()['numpy']),a.digest(before['numpy']))

    def test_observation_preserves_all_twenty_actual_cavs_and_rng(self):
        import classes
        X=np.random.default_rng(11).normal(size=(50,6)).astype(np.float32);R=np.random.default_rng(19).normal(size=(80,6)).astype(np.float32)
        c1=classes.ConceptClass(0,segments(X,'discovery'),False,50)
        c2=classes.ConceptClass(0,segments(X,'discovery'),False,50)
        random.seed(43);np.random.seed(43);initial=a.rng_snapshot()
        c1.train_cavs(R,mode='sklearn',n_runs=20,n_epochs=None,batch_size=8);expected_rng=a.rng_snapshot()
        a.restore_rng(initial)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'trace'
            with a.instrument_cav(c2,np.arange(80),out,lambda **kw:None,[]) as trace:
                c2.train_cavs(R,mode='sklearn',n_runs=20,n_epochs=None,batch_size=8)
            self.assertEqual(a.digest(a.rng_snapshot()),a.digest(expected_rng))
            for i in range(20):
                np.testing.assert_array_equal(c1.cavs[i]['weight_vector'],c2.cavs[i]['weight_vector'])
                self.assertEqual(c1.cavs[i]['accuracy'],c2.cavs[i]['accuracy'])
                self.assertEqual(set(trace[i]['negative_pool_rows']),set(trace[0]['negative_pool_rows']))
                self.assertEqual(sorted(trace[i]['train_indices']+trace[i]['test_indices']),list(range(100)))
                with np.load(out/f'round_{i:02d}.npz') as z:
                    self.assertEqual(float(np.mean(z['test_predictions']==z['test_labels'])),c2.cavs[i]['accuracy'])
                self.assertIsNone(trace[i]['sgd_params']['random_state'])
            self.assertEqual(len(trace),20)
            self.assertNotEqual(trace[0]['negative_pool_rows'],trace[1]['negative_pool_rows'])
            self.assertEqual(a.tuples(random.getstate()),a.tuples(initial['python']))

    def test_cav_failure_preserves_choice_and_does_not_retry(self):
        import classes
        from utils import utils_ace
        c=classes.ConceptClass(-1,segments(np.ones((50,3))),False,50)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'failed'
            with patch.object(utils_ace,'cav_sklearn_training',side_effect=RuntimeError('specific solver failure')) as solver:
                with self.assertRaisesRegex(RuntimeError,'specific solver'):
                    with a.instrument_cav(c,np.arange(60),out,lambda **kw:None,[]):
                        c.train_cavs(np.ones((60,3)),n_runs=20)
            trace=a.load(out/'rounds.json');self.assertEqual(len(trace),1)
            self.assertEqual(trace[0]['status'],'FAILED');self.assertEqual(len(trace[0]['negative_pool_rows']),50)
            self.assertEqual(solver.call_count,1)

    def test_kmeans_real_function_defaults_no_extra_rng(self):
        import classes
        X=np.random.default_rng(7).normal(size=(40,4));sg=segments(X)
        im=SimpleNamespace(segments=sg)
        plain=classes.ClusterSpaceClass([im],False);observed=classes.ClusterSpaceClass([im],False)
        initial=a.rng_snapshot();plain.k_means_clustering(n_clusters=3);end=a.rng_snapshot();a.restore_rng(initial)
        with tempfile.TemporaryDirectory() as tmp:
            with a.observe_kmeans(Path(tmp)):observed.k_means_clustering(n_clusters=3)
            trace=a.load(Path(tmp)/'kmeans_execution.json')
        np.testing.assert_array_equal(plain.labels,observed.labels)
        self.assertEqual(a.digest(end),a.digest(a.rng_snapshot()))
        self.assertEqual(trace['effective_arguments']['n_init'],inspect.signature(classes.k_means).parameters['n_init'].default)
        self.assertEqual(trace['estimator_params']['random_state'],43)
        self.assertGreaterEqual(trace['resolved_n_init'],1)

    def test_slic_observer_keeps_exact_masks_and_omitted_label(self):
        import classes
        with tempfile.TemporaryDirectory() as tmp:
            rows=images(tmp,1);plain=a.image_from_row(rows[0]);observed=a.image_from_row(rows[0])
            plain._segment_slic();out=Path(tmp)/'slic';out.mkdir()
            a.slic_observed(observed,out)
            self.assertEqual(len(plain.segments),len(observed.segments))
            for x,y in zip(plain.segments,observed.segments):np.testing.assert_array_equal(x.mask,y.mask)
            record=a.load(out/'slic.json')
            for i,r in enumerate(record['calls']):
                with np.load(out/f'slic_{i:02d}.npz') as z:labels=z['labels']
                self.assertNotIn(int(labels.max()),r['labels_considered'])
                self.assertGreater(r['omitted_max_pixels'],0)

    def test_streamed_feature_batches_match_original_flat_call(self):
        import torch
        import classes
        from utils import utils_general
        from torch.utils.data import DataLoader
        def mask_twice(im):
            mask=np.zeros(im.img_numpy.shape[:2],np.float32);mask[:12,:]=1
            im.segments=[classes.SegmentClass(mask,im),classes.SegmentClass(1-mask,im)]
        def observed(im,folder):mask_twice(im)
        with tempfile.TemporaryDirectory() as tmp,patch.object(utils_general,'DEVICE','cpu'):
            rows=images(tmp,9);model=tiny_model();out=Path(tmp)/'out';out.mkdir()
            with patch.object(a,'slic_observed',side_effect=observed):
                a.extract_role(model,rows,'discovery',out,lambda **kw:None,lambda *args:None)
            self.assertEqual(model.batches,[8,8,2])
            ims=[a.image_from_row(r) for r in rows]
            for im in ims:mask_twice(im)
            dataset=classes.ConceptDatasetClass(ims,model.default_cfg,1,False,None,1.)
            expected,y=utils_general.compute_activations(model,'global_pool',DataLoader(dataset,batch_size=8,shuffle=False,collate_fn=utils_general.custom_collate))
            actual,logits,_,records=a.load_role(out,'discovery')
            np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(logits,y)
            self.assertEqual([(r['image_index'],r['segment_index']) for r in records],[(i,j) for i in range(9) for j in range(2)])
            masks=a.unpack_masks(out/'discovery/image_0000/masks.npz')
            for j in range(2):np.testing.assert_array_equal(masks[j],ims[0].segments[j].mask)

    def test_gradient_hook_is_original_ce_with_batch_remainder(self):
        import classes
        from utils import utils_general
        from torch.utils.data import DataLoader
        with tempfile.TemporaryDirectory() as tmp,patch.object(utils_general,'DEVICE','cpu'):
            rows=images(tmp,2)*25;model=tiny_model();out=Path(tmp)/'out';out.mkdir()
            actual=a.gradient_cache(model,rows,out,lambda **kw:None)
            self.assertEqual(model.batches,[8,8,8,8,8,8,2])
            ims=[a.image_from_row(r) for r in rows]
            for im in ims:im.segments=[classes.SegmentClass(np.ones(im.img_numpy.shape[:2]),im)]
            data=classes.ConceptDatasetClass(ims,model.default_cfg,0,False,None)
            expected,_=utils_general.compute_activations(model,'global_pool',DataLoader(data,batch_size=8,shuffle=False,collate_fn=utils_general.custom_collate),cls_idx=207)
            np.testing.assert_array_equal(actual,expected)
            self.assertTrue(np.any(actual!=0))

    def test_nominal_p_retains_negative_difference_and_nan_excludes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'cavs').mkdir();out=root/'out';out.mkdir()
            for name,sign in (('K00',1),('K01',-1),('control',-1)):
                (root/'cavs'/name).mkdir();np.savez_compressed(root/'cavs'/name/'weights.npz',weights=np.ones((20,6))*sign)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                concepts,tests=a.score_candidates(root,{'result':{'eligible_cluster_ids':[0,1]}},np.ones((50,6)),out,lambda *args:None)
            self.assertEqual(tests[0]['mean_difference'],-1)
            self.assertTrue(tests[0]['retained'])
            self.assertFalse(tests[1]['retained']);self.assertFalse(tests[1]['pvalue_finite'])
            stored=a.load(out/'nominal_tests.json');self.assertIsNone(stored[1]['pvalue'])
            with np.load(out/'tcav_statistics.npz') as z:
                self.assertTrue(np.isnan(z['p_values'][1]));self.assertEqual(z['K00_dot_products'].shape,(20,50))

    def test_cav_stage_freezes_roles_preserves_all_candidates_and_control(self):
        # Full CPU orchestration on tiny-dimensional synthetic caches: no CNN or real data.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'fit';out.mkdir()
            random.seed(43);np.random.seed(43);a.write_json(root/'formal_rng.json',a.rng_snapshot())
            target=[dict(source=f'/train/target{i}',input_path=f'/train/target{i}',input_sha256=f'{i:064x}') for i in range(400)]
            pool=[dict(source=f'/train/random{i}',input_path=f'/train/random{i}',input_sha256=f'{i+1000:064x}') for i in range(2000)]
            val=[dict(source=f'/val/{i}',input_path=f'/val/{i}',input_sha256=f'{i+4000:064x}') for i in range(50)]
            X=np.random.default_rng(11).normal(size=(100,6));R=np.random.default_rng(12).normal(size=(2000,6))
            discovery=[dict(id=f'discovery/{i}',row=i,image_index=i//2,segment_index=i%2,input=target[i//2]) for i in range(100)]
            random_rows=[dict(id=f'random/{i}',row=i,image_index=i,segment_index=0,input=pool[i]) for i in range(2000)]
            def cached(root,role):
                data,records=(X,discovery) if role=='discovery' else (R,random_rows)
                return data,np.zeros((len(data),1000)),np.zeros(len(data),bool),records
            source=dict(inputs={'discovery':target[:50],'random':pool},dataset={'training':target,'validation':val})
            with patch.object(a,'stage_cache',return_value=(root,{})),patch.object(a,'load_role',side_effect=cached):
                result=a.cav({'feature_manifest_sha256':'a'*64},out,source,lambda **kw:None,lambda *args:None)
            candidates=a.load(out/'all_candidates.json');self.assertEqual(len(candidates),25)
            self.assertEqual(result['cav_fits'],20)  # <50 rows per cluster; original size filter
            self.assertEqual(result['structurally_eligible'],0)
            self.assertEqual(len(a.load(out/'cavs/control/rounds.json')),20)
            plan=a.load(out/'frozen_roles.json')
            self.assertEqual(len(plan['gradient_inputs']),50)
            self.assertEqual(a.tuples(a.load(out/'formal_rng.json')['python']),a.tuples(plan['python_after_gradient']))
            control_round=a.load(out/'cavs/control/rounds.json')[0]
            self.assertIn('positive_negative_image_overlap',control_round)
            self.assertIn('train_test_image_overlap',control_round)

    def test_curie_manifest_metadata_accepted_without_reordering(self):
        base=a.ROOT/'artifacts/bunya/final-28208840/report/support/formal_results'
        frozen=a.ROOT/'artifacts/bunya/reference-preparation/28204479/dataset_manifest.json'
        if not base.exists() or not frozen.exists():self.skipTest('Private source evidence not in checkout')
        dataset=a.load(frozen);launch=a.load(base/'launch_manifest.json')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'roles').mkdir()
            def extra(rows,split):
                return [dict(r,source_id=Path(r['source']).parent.name+'/'+Path(r['source']).name,original_split=split,selection_index=i) for i,r in enumerate(rows)]
            roles={'target400':extra(dataset['training'],'training'),'validation50':extra(dataset['validation'],'validation')}
            roles['discovery50']=roles['target400'][:50]
            roles['random2000']=[dict(source=f'/licensed/train/n00000001/{i}.JPEG',input_path=f'/licensed/train/n00000001/{i}.JPEG',sha256='1'*64,input_sha256='1'*64,prepared_name=f'{i+1:04d}_{i}.JPEG') for i in range(2000)]
            manifest=dict(schema='ace-r-inputs-v1',status='PASS',config={'dataset_manifest_sha256':a.sha256(frozen),'licensed_root':'/licensed'},selection={'seed':43},roles={},files={})
            for name,rows in roles.items():
                path=root/'roles'/(name+'.json');a.write_json(path,rows);relative=str(path.relative_to(root))
                manifest['roles'][name]=dict(path=relative,sha256=a.sha256(path),count=len(rows))
                manifest['files'][relative]=dict(sha256=a.sha256(path),bytes=path.stat().st_size)
            path=root/'ace_inputs_manifest.json';a.write_json(path,manifest)
            cfg=a.load(base/'resolved_config.json');cfg.update(stage='cav',ace_protocol_sha256=a.PROTOCOL_SHA256,
                ace_input_manifest=str(path),ace_input_manifest_sha256=a.sha256(path),random_seeds={'sdc':43,'ssc':43},
                golden_run_dir=str(base),golden_files_sha256={f:a.sha256(base/f) for f in a.GOLDEN_FILES},
                golden_launch_manifest=str(base/'launch_manifest.json'),golden_launch_sha256=a.sha256(base/'launch_manifest.json'),dataset_manifest=str(frozen))
            with patch.object(a.importlib.metadata,'version',side_effect=lambda key:launch['versions'][key]):
                observed=a.verify_sources(cfg)
            self.assertEqual(observed['dataset']['validation'],dataset['validation'])
            self.assertNotIn('selection_index',observed['dataset']['validation'][0])
            self.assertEqual(observed['inputs']['random'],roles['random2000'])
            self.assertIn('source_id',observed['inputs']['discovery'][0])

    def test_overlap_distinguishes_id_and_content_without_dedup(self):
        left=[dict(source='a',input_sha256='same'),dict(source='b',input_sha256='same')]
        right=[dict(source='b',input_sha256='same')]
        value=a.role_overlap(left,right)
        self.assertEqual(value['left_unique_images'],2);self.assertEqual(value['count'],1)
        self.assertEqual(value['content_count'],1)


if __name__=='__main__':unittest.main()
