import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from hpc import prerequisites as p
from hpc.reference_probe import check_mask_boundary
from utils.scientific_records import save_split
from utils.run_tracking import RunTracker

class BoundaryTests(unittest.TestCase):
    def test_batch_drift_is_not_a_mask_failure_but_real_mask_bug_is(self):
        class Model:
            bug=False
            def __call__(self,value):
                masked=isinstance(value,tuple)
                x=value[0] if masked else value
                return torch.ones((len(x),3))*(1+len(x)*0.01+(0.1 if self.bug and masked else 0))
        batch=(torch.ones(2,3,2,2),torch.ones(2,1,2,2),-1)
        model=Model()
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(check_mask_boundary(model,batch,td)['max_abs_error'],0)
            model.bug=True
            with self.assertRaises(AssertionError):check_mask_boundary(model,batch,td)
            self.assertTrue((Path(td)/'mask_boundary.npz').exists())
            self.assertGreater(json.loads((Path(td)/'mask_boundary.json').read_text())['max_abs_error'],0)

class PrerequisiteTests(unittest.TestCase):
    def test_completed_evidence_cannot_mask_changed_precision_source_or_failure(self):
        cfg=json.loads((ROOT/'hpc/configs/golden_retriever_reference.json').read_text())
        original={'status':'FAILED','slurm_job_id':'28204575','actual_commit':'a'*40,'plan':{'config':cfg}}
        comparisons={f'cuda_default_same_batch{n}_masked_vs_plain':{'exact_equal':True} for n in (1,2,8)}
        comparisons.update(cuda_cudnn_tf32_off_original_mixed2_vs_plain1={'passes_original_tolerance':True},
                           cuda_default_vs_restored={'exact_equal':True},cuda_default_original_mixed2_vs_plain1={'max_abs':.01295},
                           cuda_default_plain2_vs_plain1={'max_abs':.01295})
        diagnosis={'status':'DIAGNOSIS_COMPLETED','diagnosis_job_id':'28206208','research_commit':'a'*40,
                   'gpu_name':'NVIDIA L40S','model_state_unchanged':True,'model_training':False,
                   'comparisons':comparisons,'default_precision':cfg['precision']}
        evidence={'original_probe':{'path':'original','sha256':'x'},'diagnosis':{'path':'diagnosis','sha256':'x'},
                  'core_sha256':{'classes.py':'x'},'numerics_sha256':'x',
                  'features':{'path':'acts','sha256':'x'},'logits':{'path':'logits','sha256':'x'}}
        report={'status':'PASS','evidence':evidence,'completion_helper_sha256':'x','boundary_helper_sha256':'x',
                'job_id':'300','checks':{'ssc':{'passed':True},'algebra':{'passed':True}}}
        records={'report':report,'original':original,'diagnosis':diagnosis}
        with patch.object(p,'digest',return_value='x'),patch.object(p,'read_bound',side_effect=lambda i:records[i['path']]):
            good=p.verify_completed('report','x',ROOT,cfg)
            self.assertEqual(good['original_probe_status'],'FAILED')
            for change in ['status','precision','batch','source','ssc']:
                c=copy.deepcopy(cfg);r=copy.deepcopy(report)
                if change=='status':r['status']='FAILED'
                if change=='precision':c['precision']['cudnn_allow_tf32']=False
                if change=='batch':c['batch_size']=16
                if change=='source':r['evidence']['core_sha256']['classes.py']='wrong'
                if change=='ssc':r['checks']['ssc']['passed']=False
                records['report']=r
                with self.subTest(change=change),self.assertRaises(ValueError):p.verify_completed('report','x',ROOT,c)

    def test_bound_file_hash_changes_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'report.json';f.write_text('{}')
            bound={'path':str(f),'sha256':p.digest(f)}
            f.write_text('{"status":"PASS"}')
            with self.assertRaises(ValueError):p.read_bound(bound)

class FailureEvidenceTests(unittest.TestCase):
    def test_raw_assignments_survive_fc_failure_and_are_published(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            cfg={'source_dir':str(root/'data'),'output_dir':str(root/'out/run'),'cache_root':str(root/'cache/run'),
                 'segmentation':{'checkpoint':str(root/'model')}}
            with self.assertRaises(AssertionError):
                with RunTracker(cfg,root/'config',root,'sample') as tracker:
                    (tracker.output/'scientific').mkdir()
                    seg=SimpleNamespace(model_act=np.array([1.,2.]),model_pred=np.array([999.]),mask=np.ones((2,2)))
                    im=SimpleNamespace(segments=[seg],filename='fixed-image')
                    fc=SimpleNamespace(weight=torch.tensor([[1.,1.]]),bias=torch.tensor([0.]))
                    explainer=SimpleNamespace(model=SimpleNamespace(fc=fc))
                    save_split(explainer,[im],np.array([[.2,.8]]),tracker.output,'validation',0)
            with np.load(tracker.output/'scientific/validation.npz') as a:
                self.assertEqual(a['assignments'].tolist(),[1])
                self.assertEqual(a['logits'].tolist(),[[999.]])
            manifest=json.loads((tracker.output/'artifacts.json').read_text())
            self.assertIn('scientific/validation.npz',manifest['files'])
            self.assertEqual(json.loads((tracker.output/'scientific/validation_checks.json').read_text())['status'],'FAILED')

class LauncherRouteTests(unittest.TestCase):
    def test_resolved_route_has_no_stale_dependency_and_requires_hash(self):
        from hpc import submit_release as s
        argv=['--commit','a'*40,'--mode','reference','--config',str(ROOT/'hpc/configs/golden_retriever_reference.json'),
              '--cpus','8','--memory','32G','--time','12:00:00','--partition','gpu_cuda','--qos','short','--gpu','l40s:1',
              '--prerequisite-report','/scratch/completed.json','--prerequisite-report-sha256','b'*64]
        plan,worker=s.build_plan(s.arguments(argv))
        self.assertNotIn('after_probe',plan)
        self.assertNotIn('#SBATCH --dependency=',s.render(plan,worker))
        self.assertEqual(plan['config']['precision']['cudnn_allow_tf32'],True)
        with self.assertRaises(ValueError):s.build_plan(s.arguments(argv+['--after-probe','28204575']))
        with self.assertRaises(ValueError):s.build_plan(s.arguments(argv[:-1]+['bad-hash']))
