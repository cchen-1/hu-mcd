"""New-path controls: fail closed without discarding invalid region identities."""
import unittest
import zipfile
from unittest.mock import patch
import numpy as np
from hpc.mask_sensitivity import identity_check
from hpc.derm7pt_audit import validate_members
from hpc.workstreams import submit
from tempfile import TemporaryDirectory
from pathlib import Path


class ExecutionControls(unittest.TestCase):
    def test_identity_gate_rejects_changed_zero_or_assignment(self):
        x=np.array([[1.,0.],[0.,0.]],dtype=np.float32)
        scores=np.array([[1.,0.],[np.nan,np.nan]])
        score=lambda a:(a/np.linalg.norm(a,axis=1)[:,None],np.zeros(len(a)))
        self.assertEqual(identity_check(x,x,score,scores,np.array([0,-1]))['zero_rows'],1)
        y=x.copy();y[1,0]=1e-8
        with self.assertRaises(AssertionError):identity_check(y,x,score,scores,np.array([0,-1]))
        with self.assertRaises(AssertionError):identity_check(x,x,score,scores,np.array([1,-1]))

    def test_zip_identity_paths_and_bound(self):
        a=zipfile.ZipInfo('release_v0/meta/meta.csv');a.file_size=20
        validate_members([a],20)
        for members,bound in [([a],19),([a,a],40),([zipfile.ZipInfo('../other')],20)]:
            with self.assertRaises(ValueError):validate_members(members,bound)

    def test_resource_lane_not_scientific_dependency(self):
        plan=dict(task_key='test',commit='a'*40,runtime_root='/scratch/test',release='/scratch/release',
                  mode='mask-sensitivity',deployment_afterok='123',resource_afterany=['456'],
                  resources=dict(partition='gpu_cuda',qos='short',cpus=4,memory='16G',time='00:20:00',gpu='l40s:1'))
        with TemporaryDirectory() as d:
            with patch('hpc.workstreams.Transport') as transport:
                submit(plan,d,execute=False);transport.assert_not_called()
            script=(Path(d)/'test.sbatch').read_text()
            self.assertIn('--dependency=afterok:123,afterany:456',script)
            self.assertIn('--kill-on-invalid-dep=yes',script)


if __name__=='__main__':unittest.main()
