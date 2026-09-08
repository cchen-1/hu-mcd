import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('tracking', Path(__file__).resolve().parents[2] / 'utils/run_tracking.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

class RunTrackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {'source_dir': 'data', 'output_dir': 'outputs/smoke', 'cache_root': 'cache/smoke',
                       'segmentation': {'checkpoint': 'models/sam.pth'}}

    def test_stage_progress_failure_and_resolved_paths(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with t.RunTracker(self.config, self.root/'config.json', self.root, '123') as tracker:
                    tracker.stage_times['segmentation'] = 1.5
                    image = self.root / 'image.jpg'
                    image.write_bytes(b'input')
                    tracker.record_inputs('training', [SimpleNamespace(filename=str(image))])
                    raise RuntimeError('injected')
        output = self.root / 'outputs/runs/123'
        progress = json.loads((output/'progress.json').read_text())
        self.assertEqual(progress['status'], 'FAILED')
        self.assertEqual(progress['last_completed_stage'], 'segmentation')
        manifest = json.loads((output/'run_manifest.json').read_text())
        self.assertEqual(manifest['input_files']['training'][0]['sha256'], t.sha256(image))
        self.assertEqual(manifest['resolved_config_sha256'], t.sha256(output/'resolved_config.json'))
        self.assertEqual(self.config['output_dir'], 'outputs/smoke')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}), self.assertRaises(FileExistsError):
            t.RunTracker(self.config, self.root/'config.json', self.root, '123')

    def test_completed_artifact_index_matches_bytes(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}):
            with t.RunTracker(self.config, self.root/'config.json', self.root, '123') as tracker:
                for relative in ('summary.json', 'metrics_report.md', 'training_prototypes/overview.png', 'validation_prototypes/overview.png'):
                    path=tracker.output/relative
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(b'final')
        index=json.loads((tracker.output/'artifacts.json').read_text())
        self.assertEqual(index['files']['summary.json']['sha256'], t.sha256(tracker.output/'summary.json'))
        self.assertEqual(json.loads((tracker.output/'progress.json').read_text())['status'], 'PASS')
        self.assertEqual(list(tracker.output.glob('.*')), [])

    def test_invalid_or_mismatched_run_ids_rejected_before_writes(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}):
            for run_id in ('../evil', '124', None):
                with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                    t.RunTracker(self.config, self.root/'config.json', self.root, run_id)
        self.assertFalse((self.root/'outputs').exists())

    def test_producer_to_collector_live_and_final_snapshots(self):
        import shutil
        collector_spec = importlib.util.spec_from_file_location('collector_integration', Path(__file__).resolve().parents[1]/'collect_results.py')
        c = importlib.util.module_from_spec(collector_spec)
        collector_spec.loader.exec_module(c)
        job_state = ['RUNNING']
        root = self.root
        class LocalTransport:
            def check(self):
                pass
            def query(self, command):
                if command.startswith('sacct '):
                    return f'JobID|JobName|State\n123|humcd-p3-smoke|{job_state[0]}\n'
                return ''
            def download(self, remote, local):
                source = Path(remote)
                if not source.is_file():
                    return False
                local.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source,local)
                return True
        (root/'logs').mkdir()
        for suffix in ('out','err'):
            (root/f'logs/phase3-smoke-123.{suffix}').write_text('log')
        args = c.arguments(['123','--run-layout','--remote-root',str(root),'--output-root',str(root/'collected')])
        cache = {}
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}):
            with t.RunTracker(self.config, root/'config.json', root, '123') as tracker:
                training = tracker.output/'training_prototypes/overview.png'
                training.parent.mkdir()
                training.write_bytes(b'finalized training sheet')
                tracker.stage_times['training_assignment_and_prototypes'] = 2.5
                _, live = c.collect(args, LocalTransport(), cache)
                self.assertEqual(live['collection_status'],'partial')
                self.assertEqual(live['progress']['last_completed_stage'],'training_assignment_and_prototypes')
                for name in ('metrics_report.md','validation_prototypes/overview.png'):
                    path = tracker.output/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('final')
                t.atomic_json(tracker.output/'summary.json', {'status':'PASS', **tracker.identity,
                    'prototype_quality_proxy':{'validation':{'learned_concept_assignment_rate':0.5}}})
        job_state[0] = 'COMPLETED'
        _, final = c.collect(args,LocalTransport(),cache)
        self.assertEqual(final['collection_status'],'complete')
        self.assertEqual(final['progress']['status'],'PASS')
        self.assertTrue(any(f['local_path']=='results/training_prototypes/overview.png' and f['status']=='reused' for f in final['files']))
        (tracker.output/'resolved_config.json').write_text('{}')
        _, invalid = c.collect(args,LocalTransport())
        self.assertEqual(invalid['collection_status'],'partial')
        self.assertTrue(any('config does not match' in error for error in invalid['errors']))

if __name__ == '__main__':
    unittest.main()
