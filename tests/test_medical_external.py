"""New E224 path only: synthetic pixels, frozen cohort metadata, no inference."""
import copy,csv,json,unittest,hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from PIL import Image
from hpc.medical_external import cohort,prepare_pixels,validate,inventory_pixel_digest
from hpc.medical_sign_analysis import coverage,contrast
from hpc.workstreams import submit
ROOT=Path(__file__).resolve().parents[1]

class ExternalTests(unittest.TestCase):
    def test_pixel_geometry_is_explicit_and_grayscale_is_not_silently_converted(self):
        a=np.arange(8*12*3,dtype=np.uint8).reshape(8,12,3);im=Image.fromarray(a)
        np.testing.assert_array_equal(prepare_pixels(im),np.array(im.resize((224,224),Image.Resampling.BICUBIC)))
        self.assertEqual(im.size,(12,8))
        self.assertEqual(inventory_pixel_digest(im),hashlib.sha256(str(('RGB',a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest())
        with self.assertRaises(ValueError):prepare_pixels(im.convert('L'))

    def test_cohort_retains_metastasis_and_uses_test_only(self):
        cases=[dict(row_index=str(i),case_num=str(i+1),official_split='test',diagnosis='melanoma',derm=f'd/{i}.jpg') for i in range(101)]
        cases[100]['diagnosis']='melanoma metastasis'
        inv=[dict(row_index=i,case_num=str(i+1),role='derm',mode='RGB',relative_path=f'd/{i}.jpg') for i in range(101)]
        result=cohort(list(reversed(cases)),inv,'/scratch/data')
        self.assertEqual(result[-1]['record']['diagnosis'],'melanoma metastasis');self.assertEqual(result[0]['record']['row_index'],'0')
        cases[0]['official_split']='train'
        with self.assertRaises(ValueError):cohort(cases,inv,'/scratch/data')

    def test_union_invalid_and_signed_contrast_are_preserved(self):
        masks=np.ones((3,4,4),bool);masks[1,:2]=False
        got=coverage(masks,np.array([0,0,-1]),np.array([True,True,False]),[0,1])
        self.assertEqual(got[0]['value'],1);self.assertEqual(got[1]['value'],0)
        c=contrast([0.,.1,.9,1.],['IR','IR','ABS','ABS'],'IR')
        self.assertEqual(c['auc'],0);self.assertIsNone(c['p_value'])

    def test_approval_and_scheduler_caps_fail_before_transport(self):
        c=json.loads((ROOT/'configs/medical/medical-external.approved.json').read_text());validate(c)
        c['authorization']['status']='PENDING'
        plan=dict(mode='medical-external',config=c,resources={**c['resources'],'partition':'gpu_cuda','qos':'short'})
        with TemporaryDirectory() as d,patch('hpc.workstreams.Transport') as t:
            with self.assertRaises(ValueError):submit(plan,d,True)
            t.assert_not_called()
        c['authorization']['status']='APPROVED';plan['resources']['time']='01:00:00'
        with TemporaryDirectory() as d,patch('hpc.workstreams.Transport') as t:
            with self.assertRaises(ValueError):submit(plan,d,True)
            t.assert_not_called()
if __name__=='__main__':unittest.main()
