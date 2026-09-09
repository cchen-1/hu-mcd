"""Local synthetic checks: no pretrained model, real experiment, network or Slurm."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from hpc import mcd_reference as m


class MCDReferenceTests(unittest.TestCase):
    def test_frozen_actual_input_and_golden_mask_attestation(self):
        # Use actual saved Golden provenance. The sole L->RGB conversion MUST pass.
        base=m.ROOT/'artifacts/bunya/final-28208840/report/support/formal_results'
        if not base.exists():self.skipTest('Private Golden evidence is not in this checkout')
        cfg=m.load(base/'resolved_config.json')
        cfg.update(golden_run_dir=str(base),golden_files_sha256={f:m.sha256(base/f) for f in m.GOLDEN_FILES},
            golden_launch_manifest=str(base/'launch_manifest.json'),golden_launch_sha256=m.sha256(base/'launch_manifest.json'),
            layer_name='layer4',random_seeds={'sdc':43,'ssc':43},
            dataset_manifest=str(m.ROOT/'artifacts/bunya/reference-preparation/28204479/dataset_manifest.json'))
        original_check=m.checked_file
        def local_check(path,expected,size=None):
            if str(path)==cfg['resnet_checkpoint']:return path  # remote weights not touched
            return original_check(path,expected,size)
        with patch('hpc.workstream_runtime.verify_inputs'),patch.object(m,'checked_file',side_effect=local_check):
            source=m.verify_common(cfg)
        self.assertEqual(source['signature']['input_hashes']['validation'],
            [r['sha256'] for r in m.load(base/'run_manifest.json')['input_files']['validation']])
        self.assertTrue(any(r['sha256']!=r['input_sha256'] for r in source['dataset']['validation']))
        self.assertEqual(len(source['signature']['science_sha256']),8)

    def test_feature_hook_batches_match_original_extraction(self):
        import torch
        import classes
        from PIL import Image
        from torch.utils.data import DataLoader
        from utils import utils_general
        class Spatial(torch.nn.Module):
            def forward(self,x):
                return torch.nn.functional.adaptive_avg_pool2d(x.mean(1,keepdim=True),(7,7)).repeat(1,2048,1,1)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.layer4=Spatial();self.fc=torch.nn.Linear(2048,1000)
                self.default_cfg={'input_size':(3,224,224),'mean':(.485,.456,.406),'std':(.229,.224,.225)}
                self.batches=[]
                with torch.no_grad():self.fc.weight.fill_(1/2048);self.fc.bias.zero_()
            def forward(self,x):
                self.batches.append(len(x));return self.fc(self.layer4(x).mean((2,3)))
        with tempfile.TemporaryDirectory() as tmp,patch.object(utils_general,'DEVICE','cpu'):
            root=Path(tmp);rows=[]
            for i in range(9):
                path=root/f'{i}.png';Image.fromarray(np.full((18,23,3),20*i,np.uint8)).save(path)
                rows.append({'input_path':str(path)})
            out=root/'out';out.mkdir();model=Model().eval()
            source={'dataset':{'training':rows,'validation':rows[:3]},'model_cfg':model.default_cfg}
            with patch.object(m,'make_model',return_value=model),patch.object(torch.cuda,'get_device_name',return_value='synthetic CPU'),patch.object(torch.cuda,'max_memory_allocated',return_value=0):
                m.features({},out,source,lambda **kw:None,lambda *a:None)
            self.assertEqual(model.batches,[8,1,3])
            for split in ('training','validation'):
                images=m.load_images(source['dataset'][split])
                ds=classes.ConceptDatasetClass(images,model.default_cfg,0,False,None,1.0)
                acts,logits=utils_general.compute_activations(model,'layer4',DataLoader(ds,batch_size=8,shuffle=False,collate_fn=utils_general.custom_collate))
                observed,y=m.load_features(out,split)
                np.testing.assert_array_equal(observed,acts);np.testing.assert_array_equal(y,logits)
            mapped=m.load(out/'training_inputs.json')
            self.assertEqual(mapped['global_batches'],[list(range(8)),[8]])

    def test_cached_adapter_rejects_row_and_normalization_changes(self):
        x=np.arange(98).reshape(49,2);s=np.zeros((49,3))
        adapter=m.MCDCachedScores(None,[None,None],x,s,s)
        np.testing.assert_array_equal(adapter.concept_activations(x,49,True),s)
        for args in ((x[::-1],49,True),(x,1,True),(x,49,False)):
            with self.assertRaises(ValueError):adapter.concept_activations(*args)
        with self.assertRaises(ValueError):adapter.concept_relevances(x[::-1])

    def test_original_batched_and_streamed_trajectories_match(self):
        import benchmark_methods as b
        from PIL import Image
        rng=np.random.default_rng(43)
        with tempfile.TemporaryDirectory() as tmp:
            rows=[]
            for i in range(3):
                p=Path(tmp)/f'{i}.png';Image.fromarray(np.full((25,32,3),40+i,np.uint8)).save(p);rows.append({'input_path':str(p)})
            images=m.load_images(rows);maps=rng.random((3,2,7,7)).astype(np.float32)
            raw=maps.transpose(0,2,3,1).reshape(-1,2)
            scores=rng.random((147,4));relevance=rng.normal(size=(147,4))
            scores[:,1]=0  # truly absent learned concept; upstream empty means remain NaN
            scores[49:98,3]=0  # one image with no complement, endpoint skipped
            for i,im in enumerate(images):im.segments[0].model_act=maps[i]
            model=SimpleNamespace(default_cfg={'input_size':(3,224,224)})
            whole=m.MCDCachedScores(model,[None]*3,raw,scores,relevance)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                for mode in ('sdc','ssc'):
                    expected=b.iter_mask_imgs_mcd(whole,images,mode)
                    for i,im in enumerate(images):
                        one=m.MCDCachedScores(model,[None]*3,raw[i*49:(i+1)*49],scores[i*49:(i+1)*49],relevance[i*49:(i+1)*49])
                        observed=b.iter_mask_imgs_mcd(one,[im],mode)[0]
                        self.assertEqual(len(expected[i].segments),len(observed.segments))
                        for a,c in zip(expected[i].segments,observed.segments):np.testing.assert_array_equal(a.mask,c.mask)
                        assignment,detail=m.trajectory_details(one,im)
                        self.assertIn(1,detail['absent_concepts']);self.assertIsNone(detail['importance'][1])
                        self.assertEqual(detail['concept_order'][0],1)
                        self.assertEqual(sum(detail['assignment_counts']),224*224)
                        # Full final endpoint not manufactured; state0 is original rule.
                        if mode=='sdc':self.assertGreater(observed.segments[-1].mask.sum(),0)
                        else:self.assertLess(observed.segments[-1].mask.sum(),224*224)

    def test_cached_oblique_algebra_matches_literal_source(self):
        # Small nonsymmetric full-rank bases test actual formula, using generic dimension
        # via the factored kernel; no PCA, SSC, image prediction or model fitting.
        from scipy.linalg import lu_factor,lu_solve,norm
        from utils import utils_mcd
        bases=[np.array([[1.,0.,0.],[.2,1.,0.]]),np.array([[.1,.3,1.]])]
        raw=np.random.default_rng(43).random((49,3)).astype(np.float32)
        w=np.array([.3,.7,1.1])
        scores,relevance,reconstruction=m.project_spatial_image(raw,bases,lu_factor(np.concatenate(bases).T),w)
        np.testing.assert_allclose(scores,utils_mcd.batch_concept_activations(raw,bases,True),rtol=1e-12,atol=1e-12)
        literal=np.stack([utils_mcd.subspace_projection(bases,r) for r in raw])
        np.testing.assert_allclose(relevance,literal@w,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(reconstruction,raw,rtol=1e-12,atol=1e-12)
        with self.assertRaisesRegex(ValueError,'All-zero'):
            m.project_spatial_image(np.zeros_like(raw),bases,lu_factor(np.concatenate(bases).T),w)

    def test_fit_uses_two_ssc_passes_and_strict_benchmark_search(self):
        # Exercise real source clustering/cache control; stub numerical fitting, not rules.
        import classes
        from concept_explainer import ConceptExplainer
        from utils import utils_mcd
        from scipy import sparse
        maps=np.random.default_rng(43).random((1,2048,7,7)).astype(np.float32)
        calls=[]
        def ssc(*,acts):
            calls.append(len(acts));return sparse.diags(np.arange(1,len(acts)+1,dtype=float)).tocsr()
        def bases(explainer,**kwargs):
            eye=np.eye(2048,dtype=np.float32);explainer.concept_bases=[]
            for i,c in enumerate(explainer.concepts):
                members=c.get_segments(mode='diverse',num=None)
                c._estimate_dim(np.stack([s.model_act for s in members]),'FO')
                explainer.concept_bases.append(eye[i:i+1])
            explainer.compl_basis=eye[len(explainer.concepts):].astype(float)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'fit';out.mkdir()
            np.savez_compressed(root/'classifier.npz',weight=np.ones((1000,2048),np.float32),bias=np.zeros(1000,np.float32))
            source={'dataset':{k:[{'input_path':'fixed.jpg'}] for k in ('training','validation')},'model_cfg':{}}
            with patch.object(m,'verified_stage',return_value=(root,{})),patch.object(m,'load_features',return_value=(maps,np.zeros((1,1000)))),patch.object(utils_mcd,'compute_sparse_repr_matrix',side_effect=ssc),patch.object(classes.ClusterSpaceClass,'_spectral_clustering',side_effect=lambda affinity,k,max_samples:np.arange(affinity.shape[0])%k),patch.object(ConceptExplainer,'compute_concept_subspace_bases',new=bases),patch.object(classes.ConceptClass,'_estimate_dim',return_value=1),patch.object(utils_mcd,'calc_completeness',side_effect=[.5,.6]),patch.object(m,'basis_gate',return_value=(None,{'rank':2048})),patch.object(m,'cached_spatial_replay',return_value={'synthetic':True}),patch.object(ConceptExplainer,'concept_quantification',return_value=(np.ones(5),None)):
                result=m.fit({'feature_manifest_sha256':'a'*64},out,source,lambda **kw:None,lambda *args:None)
            self.assertEqual(calls,[49,37])
            self.assertEqual(result['selected_k'],4)
            self.assertEqual([r['k'] for r in m.load(out/'search_trace.json')],[3,4])
            self.assertEqual([r['threshold_met'] for r in m.load(out/'search_trace.json')],[False,True])
            self.assertEqual(len(m.load(out/'search/k04/pca_member_order.json')),4)
            self.assertEqual(m.load(out/'ssc/outlier_rule.json')['outliers'],12)
            self.assertTrue((out/'ssc/first_pass_outlier_mask.npy').exists())

    def test_stage_manifest_survives_parent_status_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'launch_manifest.json').write_text('RUNNING')
            (root/'actual_config.json').write_text('{}');(root/'anomalies.jsonl').touch()
            source=dict(dataset={},signature={'fixed':True},signature_sha256=m.digest({'fixed':True}),golden_versions={})
            def synthetic(config,out,source,mark,note):
                (out/'science.npz').write_bytes(b'scientific');return {'ok':True}
            cfg={'stage':'features','execution_commit':'abc'}
            with patch.dict(os.environ,{'SLURM_JOB_ID':'123','SLURM_CPUS_PER_TASK':'2'}),patch.object(m.socket,'gethostname',return_value='bun123'),patch.object(m.subprocess,'check_output',side_effect=['abc\n','']),patch.object(m,'verify_common',return_value=source),patch.object(m,'features',side_effect=synthetic):
                m.run(cfg,root)
            manifest=m.load(root/'mcd_manifest.json')
            self.assertNotIn('launch_manifest.json',manifest['files'])
            self.assertNotIn('anomalies.jsonl',manifest['files'])
            (root/'launch_manifest.json').write_text('PASS')
            (root/'artifacts.json').write_text('{}')
            c={'feature_dir':str(root),'feature_manifest_sha256':m.sha256(root/'mcd_manifest.json')}
            m.verified_stage(c,'feature',source,'features')
            (root/'science.npz').write_bytes(b'tampered')
            with self.assertRaises(ValueError):m.verified_stage(c,'feature',source,'features')

    def test_ssc_archive_roundtrip_below_collector_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'ssc').mkdir();p=out/'ssc/original.npz';data=bytes(range(251))*4;p.write_bytes(data)
            archives=m.archive_large_ssc(out,limit=300,chunk_bytes=200)
            self.assertFalse(p.exists())
            restored=b''.join((out/r['path']).read_bytes() for r in archives[0]['chunks'])
            self.assertEqual(restored,data);self.assertEqual(hashlib.sha256(restored).hexdigest(),archives[0]['sha256'])
            self.assertTrue(all(r['bytes']<=200 for r in archives[0]['chunks']))

    def test_random_full_signature_and_incomplete_attestation_rejected(self):
        cfg=dict(resnet_checkpoint_sha256='a'*64,model_name='resnet50',max_shortest_side=300,
            precision={'matmul_allow_tf32':False},batch_size=8,random_seeds={'sdc':43,'ssc':43})
        source=dict(dataset={'validation':[dict(source='x.jpg',input_sha256='b'*64)]},model_cfg={'input_size':[3,224,224]},
            signature={'science_sha256':{k:'c'*64 for k in m.SCIENCE_FILES}})
        self.assertEqual(m.verify_random_reuse(cfg,source)['new_random_predictions'],0)
        expected=m.random_signature(cfg,source['model_cfg'],source['dataset']['validation'],source['signature']['science_sha256'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'attestation.json'
            contract=dict(signature=expected,status='PASS',source_job='28214892',files={});m.write_json(path,contract)
            cfg['random_reuse']=dict(directory=tmp,contract_path=str(path),contract_sha256=m.sha256(path))
            with self.assertRaisesRegex(ValueError,'omits required'):m.verify_random_reuse(cfg,source)
            cfg['batch_size']=7
            with self.assertRaisesRegex(ValueError,'signature differs'):m.verify_random_reuse(cfg,source)

    def test_local_entry_denied_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'SLURM_JOB_ID':''}):
            path=Path(tmp)/'never'
            with self.assertRaisesRegex(RuntimeError,'Slurm'):m.run({'stage':'features'},path)
            self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()
