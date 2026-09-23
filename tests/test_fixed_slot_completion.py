"""Targeted empty-mask batch tests; no SAM or scientific model predictions."""
import unittest
import numpy as np
from hpc.final_mask_intervention import FixedSlotDataset

class FixedSlots(unittest.TestCase):
    def test_original_slot_and_discarded_output(self):
        original=[('original',i) for i in range(10)]
        changed=[('changed',i) for i in range(10)]
        empty=np.zeros(10,dtype=bool);empty[[3,9]]=True
        ds=FixedSlotDataset(original,changed,empty)
        for ids in [list(range(8)),[8,9]]:
            samples=[ds[i] for i in ids]
            self.assertEqual(len(samples),len(ids))
            self.assertEqual([s[1] for s in samples],ids)
            self.assertEqual([s[0] for s in samples],['original' if empty[i] else 'changed' for i in ids])
            fake_output=np.array([i*10 for i in ids],float)
            result=np.full(len(ids),np.nan);positions=[j for j,i in enumerate(ids) if not empty[i]]
            result[positions]=fake_output[positions]
            np.testing.assert_array_equal(np.isnan(result),empty[ids])
    def test_bad_mapping_rejected(self):
        with self.assertRaises(ValueError):FixedSlotDataset([1,2],[1],[False,True])

if __name__=='__main__':unittest.main()
