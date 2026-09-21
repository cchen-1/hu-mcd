import unittest
import numpy as np
from hpc.final_mask_intervention import intervene, FinalMaskInterventionDataset

class FinalMaskTests(unittest.TestCase):
    def test_disk_and_zero_boundary(self):
        mask = np.zeros((7, 7), dtype=bool);mask[3,3] = True
        self.assertEqual(int(intervene(mask, 'dilation', 1).sum()), 5)
        self.assertEqual(int(intervene(mask, 'dilation', 2).sum()), 13)
        self.assertFalse(intervene(mask, 'erosion', 1).any())
        self.assertEqual(int(intervene(np.ones((7,7)), 'erosion', 1).sum()),25)
        mask[:]=False;mask[0,0]=True
        self.assertEqual(int(intervene(mask, 'dilation', 1).sum()),3)

    def test_invalid_and_identity(self):
        mask=np.array([[1,0],[0,1]],dtype=np.float32)
        np.testing.assert_array_equal(intervene(mask,'identity',0),mask)
        with self.assertRaises(ValueError):intervene(mask*.5,'dilation',1)
        with self.assertRaises(ValueError):intervene(mask,'identity',1)

    def test_source_dataset_identity_and_no_mutation(self):
        import torch
        from types import SimpleNamespace
        from classes import ConceptDatasetClass
        raw=np.zeros((35,47),dtype=np.float32);raw[2:29,5:40]=1
        original=SimpleNamespace(img_numpy=np.full((35,47,3),.3,dtype=np.float32))
        original.segments=[SimpleNamespace(mask=raw,org_img=original)]
        dataset=ConceptDatasetClass([original],{'input_size':(3,224,224),'mean':(.485,.456,.406),'std':(.229,.224,.225)},cropping_mode=0,use_masks=True,masking_mode=-1,erosion_threshold=.25)
        before=dataset[0];wrapped=FinalMaskInterventionDataset(dataset,'identity',0)
        self.assertEqual(len(wrapped),1)
        after=wrapped[0]
        self.assertTrue(torch.equal(before[0],after[0]));self.assertTrue(torch.equal(before[1],after[1]));self.assertEqual(before[2],after[2])
        FinalMaskInterventionDataset(dataset,'erosion',2)[0]
        self.assertTrue(torch.equal(dataset[0][1],before[1]))

if __name__=='__main__':unittest.main()
