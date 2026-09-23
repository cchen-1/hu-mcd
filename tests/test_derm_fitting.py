import copy,json,unittest
from pathlib import Path
from hpc.derm_fitting import validate,fit_rows
from hpc.medical_protocol import digest

class DermFittingTests(unittest.TestCase):
    def test_protocol_rejects_square_warp_old_model_and_scope_change(self):
        config=json.loads(Path('configs/medical/derm-r101-fit.approved.json').read_text());validate(config)
        for field,value in [('extra_square_resize_before_sam',True),('held_out_images',70),('min_cluster_size',10),('outlier_quantile',.99)]:
            c=copy.deepcopy(config);c['protocol'][field]=value;c['protocol_sha256']=digest(c['protocol']);c['authorization']['approved_protocol_sha256']=c['protocol_sha256']
            with self.assertRaises(ValueError):validate(c)
    def test_fixed_order_and_roles(self):
        d=dict(count=101,held_out=0,rows=[dict(fit_order=i,image_id=str(i),role='DISCOVERY_FIT_NO_HELD_OUT',extra_square_resize_before_sam=False,uses_classifier_cache=False,input_size_wh=[450,300]) for i in range(101)])
        self.assertEqual(fit_rows(d)['held_out'],[])
        for change in ['order','geometry','cache']:
            m=copy.deepcopy(d)
            if change=='order':m['rows'][0]['fit_order']=1
            if change=='geometry':m['rows'][0]['input_size_wh']=[224,224]
            if change=='cache':m['rows'][0]['uses_classifier_cache']=True
            with self.assertRaises(ValueError):fit_rows(m)
