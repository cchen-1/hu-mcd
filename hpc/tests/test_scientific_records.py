import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from utils.scientific_records import numerical_checks

class ReconstructionTests(unittest.TestCase):
    def test_nonorthogonal_concepts_complement_negative_relevance_and_bias(self):
        bases=[np.array([[1.,0.,0.]]),np.array([[.6,.8,0.]]),np.array([[0.,0.,1.]])]
        acts=np.array([[1.,2.,3.],[-4.,1.,0.],[0.,-2.,5.]])
        weight=np.array([2.,-3.,1.]);bias=4.
        logits=acts@weight+bias
        checks,data=numerical_checks(acts,logits,weight,bias,bases)
        self.assertLess(checks['feature_reconstruction_max_relative_error'],1e-10)
        self.assertTrue((data['sample_local_relevance']<0).any())
        np.testing.assert_allclose(data['sample_local_relevance'].sum(1)+bias,logits)

    def test_wrong_fc_bias_or_missing_complement_rejected(self):
        acts=np.array([[1.,2.,3.]])
        weight=np.array([2.,-3.,1.]);bias=4.;logits=acts@weight+bias
        with self.assertRaises(AssertionError):
            numerical_checks(acts,logits,weight,0.,None)
        with self.assertRaises(ValueError):
            numerical_checks(acts,logits,weight,bias,[np.array([[1.,0.,0.],[0.,1.,0.]])])

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            numerical_checks(np.array([[np.nan]]),np.array([1.]),np.array([1.]),0,None)
