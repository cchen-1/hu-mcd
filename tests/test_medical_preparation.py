"""Synthetic software checks only: no medical images, weights, network or Slurm."""
import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hpc.medical_protocol import (workload, learning_rate, improves, digest,
                                  validate_config, execution_gate, require_frozen_test)
from hpc.medical_overlap import hashes, orientations, distances, candidate_indices, pixel_key
from hpc.medical_classifier import SourceImages, metrics, infer, save_state
from hpc.medical_sign_analysis import coverage, contrast, CONTRASTS
from hpc.workstreams import submit

ROOT=Path(__file__).resolve().parents[1]


def config(mode='medical-classifier'):
    return json.loads((ROOT/'configs/medical'/f'{mode}.proposed.json').read_text())


class MedicalPreparationTests(unittest.TestCase):
    def test_workload_and_schedule_boundaries(self):
        a=workload(8215,50,64); b=workload(8215,100,128)
        self.assertEqual((a['optimizer_updates'],a['training_image_visits']), (6450,410750))
        self.assertEqual((b['optimizer_updates'],b['training_image_visits']), (6500,821500))
        self.assertEqual((a['final_batch_size'],b['final_batch_size']), (23,23))
        p=config()['protocol']
        np.testing.assert_allclose([learning_rate(i,p) for i in (1,25,26,38,39,50)],
                                   [1e-4,1e-4,1e-5,1e-5,1e-6,1e-6])
        with self.assertRaises(ValueError):learning_rate(51,p)

    def test_unapproved_blocks_before_transport_or_data_access(self):
        for mode in ('medical-overlap','medical-classifier'):
            c=config(mode);validate_config(c,mode)
            plan=dict(mode=mode,config=c,resources={**c['resources'], 'partition':'general','qos':'debug'},
                      commit='a'*40,task_key='synthetic-preparation-test',runtime_root='/scratch/test',release='/scratch/test/release')
            with TemporaryDirectory() as tmp,patch('hpc.workstreams.Transport') as remote:
                with self.assertRaisesRegex(ValueError,'explicit approval'):submit(plan,tmp,execute=True)
                remote.assert_not_called()
                self.assertFalse(list(Path(tmp).iterdir()))
            with patch.dict(os.environ,{'SLURM_JOB_ID':'123'}),patch('hpc.medical_protocol.socket.gethostname',return_value='bun148'):
                with self.assertRaisesRegex(ValueError,'explicit approval'):execution_gate(c,mode)

    def test_approved_config_still_requires_compute_node_and_exact_digest(self):
        c=config(); c['authorization']=dict(status='APPROVED',user_decision='SYNTHETIC TEST ONLY',approved_protocol_sha256=c['protocol_sha256'])
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123'}),patch('hpc.medical_protocol.socket.gethostname',return_value='bunya-login'):
            with self.assertRaisesRegex(RuntimeError,'compute allocation'):execution_gate(c,'medical-classifier')
        c['protocol']['classifier_training_seed']=44
        with self.assertRaisesRegex(ValueError,'digest'):validate_config(c,'medical-classifier',True)

    def test_selection_tie_nonfinite_and_test_gate(self):
        self.assertTrue(improves(.7,None));self.assertFalse(improves(.7,.7));self.assertTrue(improves(.71,.7))
        for score in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):improves(score,.7)
        f=dict(checkpoint_sha256='a'*64,test_outcomes_used_for_selection=False)
        require_frozen_test(f,'a'*64,50,50)
        for sha,epochs in [('b'*64,50),('a'*64,49)]:
            with self.assertRaises(ValueError):require_frozen_test(f,sha,epochs,50)

    def test_decoded_identity_distinct_from_canonical_and_hash_candidates(self):
        a=np.random.default_rng(1).integers(0,256,(32,48,3),dtype=np.uint8)
        self.assertNotEqual(pixel_key(a),pixel_key(a.reshape(48,32,3)))
        self.assertEqual(len(list(orientations(a))),8)
        source=np.array([hashes(a)[0],hashes(a)[0]])
        dist=distances(source,np.array([hashes(a)[0]]))
        self.assertEqual(dist.tolist(),[[0,0]])
        # A far nearest neighbour is still retained, while threshold hits are complete.
        chosen,pm,dm=candidate_indices(np.array([[8,8,9]]),np.array([[10,9,7]]),[1],6)
        self.assertEqual(chosen,[0,1,2])

    def test_dataset_row_alignment_and_no_validation_augmentation(self):
        p=config()['protocol']; a=np.zeros((2,224,224,3),dtype=np.uint8)
        a[0,:,0]=255; rows=[dict(label='0'),dict(label='1')]
        ds=SourceImages(a,np.array([0,1]),rows,p,False)
        x,y,i=ds[0];self.assertEqual((y,i),(0,0))
        self.assertTrue(torch.equal(x,ds[0][0]))
        with self.assertRaises(ValueError):SourceImages(a,np.array([1,0]),rows,p)

    def test_metrics_use_all_classes_and_record_constant_argmax(self):
        labels=np.tile(np.arange(7),2); logits=np.eye(7)[labels]*4
        m=metrics(labels,logits);self.assertEqual(m['macro_ovr_auc'],1);self.assertEqual(m['accuracy'],1)
        constant=np.zeros((14,7));constant[:,0]=1
        m=metrics(labels,constant);self.assertTrue(m['constant_argmax']);self.assertEqual(m['macro_ovr_auc'],.5)
        logits[0,0]=np.nan
        with self.assertRaises(ValueError):metrics(labels,logits)
        with self.assertRaises(ValueError):metrics(np.zeros(3,dtype=int),np.zeros((3,7)))

    def test_synthetic_checkpoint_components_and_rng_are_preserved(self):
        model=torch.nn.Linear(2,7); opt=torch.optim.Adam(model.parameters());g=torch.Generator().manual_seed(43)
        with TemporaryDirectory() as tmp:
            save_state(Path(tmp)/'last',model,opt,g,1,.5)
            saved=json.loads((Path(tmp)/'last/identity.json').read_text())
            self.assertEqual(set(saved['files']),{'model.pt','optimizer.pt','rng.pt'})
            self.assertEqual(saved['epoch'],1)

    def test_coverage_union_zero_invalid_missing_and_overlap(self):
        masks=np.array([[[1,1],[0,0]],[[0,1],[1,0]],[[1,1],[1,1]]],dtype=bool)
        c=coverage(masks,np.array(['C1','C1','C2']),np.array([True,True,False]),['C1','C2'])
        self.assertEqual(c['C1']['value'],.75);self.assertEqual(c['C2']['value'],0)
        self.assertEqual(c['C2']['status'],'NO_VALID_ASSIGNED_MEMBER')
        self.assertIsNone(coverage(None,None,None,['C1'],False)['C1']['value'])
        with self.assertRaises(ValueError):coverage(masks.astype(float),np.array([0,0,1]),np.ones(3,dtype=bool),[0])

    def test_twelve_contrasts_do_not_flip_negative_association_or_fill_missing(self):
        self.assertEqual(len(CONTRASTS),12)
        r=contrast([.8,.1,np.nan,.7],['ABS','REG','REG','IR'],'REG')
        self.assertEqual((r['group_n'],r['reference_n'],r['missing_n'],r['other_category_n']),(1,1,1,1))
        self.assertEqual(r['auc'],0);self.assertIsNone(r['p_value'])
        self.assertIsNone(contrast([.1],['REG'],'REG')['auc'])


if __name__=='__main__':unittest.main()
