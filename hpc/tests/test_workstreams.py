import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from hpc.workstreams import submit
from hpc.workstream_runtime import verify_inputs

class WorkstreamTests(unittest.TestCase):
    def plan(self):
        return {'task_key':'A-input-audit','commit':'a'*40,'runtime_root':'/scratch/user/uqcche38/hu-mcd','release':'/scratch/user/uqcche38/releases/'+'a'*40+'/code','mode':'inventory','config':{},'resources':{'partition':'general','qos':'debug','cpus':2,'memory':'8G','time':'00:10:00','gpu':None}}
    def test_render_cannot_submit_or_compute_on_login(self):
        with tempfile.TemporaryDirectory() as tmp, patch('hpc.workstreams.Transport') as transport:
            result=submit(self.plan(),tmp,False);self.assertEqual(result['status'],'PLANNED');transport.assert_not_called()
            script=Path(tmp,'A-input-audit.sbatch').read_text();self.assertIn('python -m hpc.workstream_runtime',script);self.assertIn('SLURM_JOB_ID',script)
    def test_uncertain_or_previous_receipt_blocks_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'A-input-audit.submission.json').write_text('{"status":"SUBMISSION_UNCERTAIN"}')
            with self.assertRaises(FileExistsError):submit(self.plan(),tmp,True)
    def test_inspection_cannot_request_gpu(self):
        plan=self.plan();plan['resources']['gpu']='l40s:1'
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):submit(plan,tmp,False)
    def test_unsafe_scheduler_text_rejected(self):
        plan=self.plan();plan['resources']['qos']='debug\n#SBATCH --gres=gpu:1'
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):submit(plan,tmp,False)
    def test_inputs_hash_mismatch_fails_before_opening_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp,'manifest.json');path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'manifest hash'):verify_inputs({'dataset_manifest':str(path),'dataset_manifest_sha256':'0'*64})

if __name__=='__main__':unittest.main()
