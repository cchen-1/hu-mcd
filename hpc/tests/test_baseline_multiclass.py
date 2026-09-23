"""New class routing/target-gradient and dependency isolation tests; no cluster calls."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from hpc import baseline_multiclass as b,ace_reference as a
from hpc.tests.test_ace_reference import images,tiny_model

class MulticlassTests(unittest.TestCase):
    def test_protocol_only_changes_class_identity(self):
        for name in ['airliner','zebra','ox','Siamese_cat']:
            p=a.protocol_for(name)
            diff={k for k in p if p[k]!=a.PROTOCOL[k]}
            self.assertEqual(diff,{'class_name','discovery_rule'})
        self.assertEqual(a.protocol_for('golden_retriever'),a.PROTOCOL)

    def test_airliner_gradient_uses_404_not_golden_207(self):
        from utils import utils_general
        import torch
        with tempfile.TemporaryDirectory() as tmp,patch.object(utils_general,'DEVICE','cpu'):
            root=Path(tmp);rows=images(tmp,2)*25;model=tiny_model();out=root/'out';out.mkdir()
            actual=a.gradient_cache(model,rows,out,lambda **kw:None,target=404)
            with np.load(out/'gradients.npz') as z:logits=z['logits']
            W=model.fc.weight.detach().numpy();expected=[];wrong=[]
            for start in range(0,50,8):
                y=logits[start:start+8];probs=torch.softmax(torch.from_numpy(y),dim=1).numpy()
                expected.append((probs@W-W[404])/len(y));wrong.append((probs@W-W[207])/len(y))
            np.testing.assert_allclose(actual,np.concatenate(expected),atol=1e-7,rtol=1e-5)
            self.assertFalse(np.allclose(actual,np.concatenate(wrong)))

    def test_dependency_rejects_wrong_class_commit_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'outputs/workstreams/123';root.mkdir(parents=True)
            dep=dict(job_id='123',task_key='MCD-airliner-features',prefix='feature',manifest='mcd_manifest.json',stage='features')
            cfg=dict(runtime_root=tmp,execution_commit='a'*40,class_name='airliner',dataset_manifest_sha256='b'*64,resnet_checkpoint_sha256='c'*64,precision={'tf32':False},stage_dependencies=[dep])
            launch=dict(status='PASS',commit='a'*40,task_key=dep['task_key'])
            manifest=dict(status='PASS',stage='features');(root/'mcd_manifest.json').write_text(json.dumps(manifest))
            index=dict(status='PASS',job_id='123',commit='a'*40,files={'mcd_manifest.json':dict(sha256=b.sha256(root/'mcd_manifest.json'),bytes=(root/'mcd_manifest.json').stat().st_size)})
            def write(old=cfg,record=launch):
                for name,obj in [('launch_manifest.json',record),('artifacts.json',index),('actual_config.json',old)]:
                    (root/name).write_text(json.dumps(obj))
            write();resolved,bindings=b.resolve(cfg);self.assertEqual(resolved['feature_manifest_sha256'],index['files']['mcd_manifest.json']['sha256'])
            for old,record in [(dict(cfg,class_name='zebra'),launch),(cfg,dict(launch,commit='d'*40)),(cfg,dict(launch,status='FAILED'))]:
                write(old,record)
                with self.assertRaises(ValueError):b.resolve(cfg)

if __name__=='__main__':unittest.main()
