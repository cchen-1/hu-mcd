"""Targeted new-path checks: synthetic tensors/metadata, never model inference or SAM."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from scipy import sparse
from hpc.medical_discovery import select_inputs,validate
from hpc.workstreams import submit
from utils.utils_mcd import get_outlier_mask
from classes import ImageClass,SegmentClass,ConceptDatasetClass
from run_humcd import get_top_concept_segms

ROOT=Path(__file__).resolve().parents[1]
class MedicalDiscoveryTests(unittest.TestCase):
    def test_quantile_one_is_no_outlier_removal_not_one_percent(self):
        a=sparse.diags([1.,2.,4.,100.])
        np.testing.assert_array_equal(get_outlier_mask(a,1.),[False]*4)
        np.testing.assert_array_equal(get_outlier_mask(a,.99),[False,False,False,True])

    def test_selection_uses_lesions_and_independent_rng(self):
        train=[dict(label='4',lesion_id=f'L{i:04d}',image_id=f'I{i:04d}-{j}') for i in range(533) for j in range(2)]
        rows=dict(train=train,test=[dict(label='4',image_id='T1',lesion_id='TL1'),dict(label='1',image_id='T2',lesion_id='TL2')])
        np.random.seed(2);before=np.random.get_state();a=select_inputs(rows);after=np.random.get_state()
        self.assertEqual(len({r['lesion_id'] for r in a['training']}),400)
        self.assertEqual(len(a['held_out']),1);np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])
        rows['train']=list(reversed(train));self.assertEqual(a,select_inputs(rows))
        np.random.seed(200);self.assertEqual(a,select_inputs(rows))

    def test_uint8_hu_tensor_matches_classifier_and_mask_mapping(self):
        # Every possible channel value exercises float32 -> uint8 roundtrip.
        rgb=np.resize(np.arange(256,dtype=np.uint8),(224,224,3))
        cfg=json.loads((ROOT/'configs/medical/medical-discovery.approved.json').read_text())['protocol']
        with TemporaryDirectory() as d:
            f=Path(d)/'synthetic.png';Image.fromarray(rgb).save(f);im=ImageClass(str(f),300)
            mask=np.zeros((224,224),np.float32);mask[10:40,20:60]=1
            im.segments=[SegmentClass(mask,im)]
            c=dict(input_size=(3,224,224),mean=cfg['mean'],std=cfg['std'])
            ds=ConceptDatasetClass([im],c,0,True,-1,.25);x,m,mode=ds[0]
            expected=(torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255-torch.tensor(cfg['mean'])[:,None,None])/torch.tensor(cfg['std'])[:,None,None]
            self.assertTrue(torch.equal(x,expected));np.testing.assert_array_equal(m.numpy()[0],mask);self.assertEqual(mode,-1)
            # Large raw region undergoes the existing conditional erosion; smaller one does not.
            large=np.zeros((224,224),np.float32);large[20:204,20:204]=1;im.segments=[SegmentClass(large,im)]
            ds=ConceptDatasetClass([im],c,0,True,-1,.25);self.assertLess(ds[0][1].sum().item(),large.sum())

    def test_prototypes_use_assigned_members_and_unique_images(self):
        acts=np.array([[.8,.9],[.6,.5],[.7,.1],[.9,.1]])
        top=get_top_concept_segms(acts,[2,2],10)
        self.assertEqual(top[0],[(1,1),(0,1)]);self.assertEqual(top[1],[(0,0)])

    def test_scope_gate_and_scientific_dependencies(self):
        cfg=json.loads((ROOT/'configs/medical/medical-discovery.approved.json').read_text());validate(cfg)
        bad=copy.deepcopy(cfg);bad['protocol']['outlier_quantile']=.99
        with self.assertRaises(ValueError):validate(bad)
        cfg['authorization']['status']='PENDING'
        plan=dict(mode=cfg['mode'],config=cfg,commit='a'*40,task_key='medical-synthetic',runtime_root='/scratch/test',release='/scratch/test/release',resources={**cfg['resources'],'partition':'gpu_cuda','qos':'normal'},deployment_afterok='11',prerequisite_afterok=['12'])
        with TemporaryDirectory() as d,patch('hpc.workstreams.Transport') as t:
            with self.assertRaises(ValueError):submit(plan,d,True)
            t.assert_not_called()
        cfg['authorization']['status']='APPROVED'
        with TemporaryDirectory() as d:
            submit(plan,d,False);s=(Path(d)/'medical-synthetic.sbatch').read_text()
            self.assertIn('--dependency=afterok:11,afterok:12',s);self.assertIn('--kill-on-invalid-dep=yes',s)
if __name__=='__main__':unittest.main()
