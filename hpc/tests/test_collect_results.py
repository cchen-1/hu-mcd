import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("collector", Path(__file__).resolve().parents[1] / "collect_results.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
ACCOUNTING = "JobID|JobName|User|State|Elapsed|ReqMem|ExitCode\n28005092|humcd-p3-smoke|user|COMPLETED|00:00:42|8G|0:0\n"
SUMMARY = {"status": "PASS", "prototype_quality_proxy": {"validation": {"learned_concept_assignment_rate": 0}}}


class FakeTransport:
    def __init__(self, missing=None, invalid=False, disconnected=False):
        self.missing, self.invalid, self.disconnected = missing, invalid, disconnected
        self.commands = []

    def check(self):
        if self.disconnected:
            raise c.CollectorError("no master")

    def query(self, command):
        self.commands.append(command)
        return ACCOUNTING if command.startswith("sacct ") else "999|RUNNING|node\n"

    def download(self, remote, local):
        if self.missing and remote.endswith(self.missing):
            return False
        if remote.endswith("progress.json"):
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(SUMMARY) if remote.endswith("summary.json") else "fixture"
        if self.invalid and remote.endswith("summary.json"):
            data = "{unfinished"
        local.write_text(data)
        return True


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = c.arguments(["28005092", "--output-root", self.tmp.name])

    def test_snapshots_preserve_previous_downloads_and_report_science_warning(self):
        transport = FakeTransport()
        first, manifest = c.collect(self.args, transport)
        (first / "results/summary.json").write_text("keep original")
        second, _ = c.collect(self.args, transport)
        self.assertNotEqual(first, second)
        self.assertEqual((first / "results/summary.json").read_text(), "keep original")
        self.assertEqual(manifest["collection_status"], "complete")
        self.assertTrue(any("Zero validation" in w for w in manifest["warnings"]))
        self.assertIn("unverified", manifest["artifact_attribution"])
        self.assertEqual((second / "squeue.txt").read_text(), "")
        self.assertTrue(all(cmd.startswith(("sacct ", "squeue ")) for cmd in transport.commands))
        self.assertEqual(sum(f["status"] == "downloaded" for f in manifest["files"]), 6)

    def test_missing_master_does_not_create_snapshot_or_query(self):
        transport = FakeTransport(disconnected=True)
        with self.assertRaises(c.CollectorError):
            c.collect(self.args, transport)
        self.assertEqual(transport.commands, [])
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_missing_required_file_is_partial(self):
        _, manifest = c.collect(self.args, FakeTransport(missing="summary.json"))
        self.assertEqual(manifest["collection_status"], "partial")

    def test_incomplete_json_preserved_but_flagged(self):
        folder, manifest = c.collect(self.args, FakeTransport(invalid=True))
        self.assertEqual(manifest["collection_status"], "partial")
        self.assertEqual((folder / "results/summary.json").read_text(), "{unfinished")
        self.assertTrue(manifest["errors"])

    def test_lost_connection_keeps_error_record(self):
        transport = FakeTransport()
        with patch.object(transport, "download", side_effect=c.CollectorError("connection lost")):
            folder, manifest = c.collect(self.args, transport)
        self.assertEqual(manifest["collection_status"], "partial")
        self.assertTrue((folder / "manifest.json").exists())

    def test_wrong_job_profile_rejected(self):
        with self.assertRaises(c.CollectorError):
            c.parse_accounting(ACCOUNTING, "28005092", c.PROFILES["gpu-probe"])

    def test_injection_and_parent_paths_rejected(self):
        for opts in (["1;id"], ["1", "--remote-root", "/scratch/../home"],
                     ["1", "--output-subdir", "../models"], ["1", "--interval", "1"], ["1", "--output-root", "/tmp/bad\npath"]):
            with self.subTest(opts=opts), self.assertRaises(SystemExit), patch("sys.stderr"):
                c.arguments(opts)
        with self.assertRaises(c.CollectorError):
            c.Transport("-oProxyCommand=anything")

    def test_transport_options_disable_fresh_connections(self):
        transport = c.Transport("bunya")
        with patch.object(c, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as call:
            transport.query("squeue --me")
            argv = call.call_args.args[0]
            for setting in ("ProxyCommand=/bin/false", "BatchMode=yes", "ControlMaster=no"):
                self.assertIn(setting, argv)
            transport.download("/scratch/result.json", Path(self.tmp.name) / "folder with spaces/result.json")
            argv, batch = call.call_args.args
            self.assertEqual(argv[0], "sftp")
            self.assertIn("ProxyCommand=/bin/false", argv)
            self.assertTrue(batch.startswith('get "/scratch/result.json" "'))

    def test_missing_sftp_file_does_not_hide_connection_errors(self):
        transport = c.Transport("bunya")
        dest = Path(self.tmp.name) / "result"
        with patch.object(c, "run", return_value=subprocess.CompletedProcess([], 1, "", "No such file")):
            self.assertFalse(transport.download("/scratch/result", dest))
        with patch.object(c, "run", return_value=subprocess.CompletedProcess([], 1, "", 'File "/scratch/result" not found.')):
            self.assertFalse(transport.download("/scratch/result", dest))
        with patch.object(c, "run", return_value=subprocess.CompletedProcess([], 255, "", "Connection closed")):
            with self.assertRaises(c.CollectorError):
                transport.download("/scratch/result", dest)

    def test_watch_stops_on_terminal_state(self):
        with patch.object(c, "Transport", return_value=FakeTransport()), patch.object(c.time, "sleep") as sleep:
            self.assertEqual(c.main(["28005092", "--output-root", self.tmp.name, "--watch"]), 0)
            sleep.assert_not_called()

class LiveCollectorTests(unittest.TestCase):
    setUp = CollectorTests.setUp
    def test_watch_retries_incomplete_json_until_terminal(self):
        class Live(FakeTransport):
            polls = 0
            def query(self, command):
                if command.startswith('sacct '):
                    self.polls += 1
                    self.invalid = self.polls == 1
                    return ACCOUNTING.replace('COMPLETED', 'RUNNING') if self.polls == 1 else ACCOUNTING
                return ''
        transport = Live()
        with patch.object(c, 'Transport', return_value=transport), patch.object(c.time, 'sleep') as sleep:
            self.assertEqual(c.main(['28005092', '--output-root', self.tmp.name, '--watch', '--max-polls', '3']), 0)
            self.assertEqual(transport.polls, 2)
            sleep.assert_called_once_with(60)

    def test_watch_stops_on_connection_loss_even_if_job_running(self):
        transport = FakeTransport()
        with patch.object(transport, 'query', return_value=ACCOUNTING.replace('COMPLETED', 'RUNNING')), \
             patch.object(transport, 'download', side_effect=c.CollectorError('lost master')), \
             patch.object(c, 'Transport', return_value=transport), patch.object(c.time, 'sleep') as sleep:
            self.assertEqual(c.main(['28005092', '--output-root', self.tmp.name, '--watch']), 2)
            sleep.assert_not_called()

    def test_file_limit_is_enforced_in_actual_child_process(self):
        path = Path(self.tmp.name) / 'oversized'
        result = c.run([__import__('sys').executable, '-c',
                        'import sys; open(sys.argv[1], "wb").write(b"x" * 8192)', str(path)], max_file_bytes=1024)
        self.assertNotEqual(result.returncode, 0)
        self.assertLessEqual(path.stat().st_size, 1024)

    def test_run_layout_checks_identity_checksums_and_reuses_only_verified_bytes(self):
        import hashlib
        class Modern(FakeTransport):
            calls = []
            wrong_job = False
            def download(self, remote, local):
                self.calls.append(remote)
                identity = {'run_id': '28005092', 'slurm_job_id': '1' if self.wrong_job else '28005092'}
                summary = {**SUMMARY, **identity}
                payloads = {name: (json.dumps(summary) if name == 'summary.json' else 'fixture') for name in c.PROFILES['smoke']['files']}
                if remote.endswith('run_manifest.json'):
                    data = json.dumps({**identity, 'resolved_config_sha256': hashlib.sha256(b'{}').hexdigest()})
                elif remote.endswith('resolved_config.json'):
                    data = '{}'
                elif remote.endswith('artifacts.json'):
                    data = json.dumps({**identity, 'files': {name: {'bytes': len(value), 'sha256': hashlib.sha256(value.encode()).hexdigest()} for name,value in payloads.items()}})
                elif remote.endswith('progress.json'):
                    data = json.dumps({**identity, 'status': 'PASS'})
                elif remote.endswith('summary.json'):
                    data = json.dumps(summary)
                else:
                    return super().download(remote, local)
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_text(data)
                return True
        self.args = c.arguments(['28005092', '--output-root', self.tmp.name, '--run-layout'])
        transport, cache = Modern(), {}
        first, manifest = c.collect(self.args, transport, cache)
        self.assertEqual(manifest['collection_status'], 'complete')
        second, manifest = c.collect(self.args, transport, cache)
        self.assertEqual(sum(e['status'] == 'reused' for e in manifest['files']), 4)
        self.assertEqual((first / 'results/training_prototypes/overview.png').read_text(), 'fixture')
        (second / 'results/training_prototypes/overview.png').write_text('tampered')
        third, manifest = c.collect(self.args, transport, cache)
        self.assertEqual(sum(e['status'] == 'reused' for e in manifest['files']), 3)
        self.assertEqual((third / 'results/training_prototypes/overview.png').read_text(), 'fixture')
        transport.wrong_job = True
        _, manifest = c.collect(self.args, transport, cache)
        self.assertEqual(manifest['collection_status'], 'partial')
        self.assertFalse(any(e['status'] == 'reused' for e in manifest['files']))

    def test_transfer_limit_and_layout_arguments(self):
        for options in (['--max-file-mib', '0'], ['--max-file-mib', '257'],
                        ['--run-layout', '--profile', 'gpu-probe'], ['--run-layout', '--output-subdir', 'other']):
            with self.subTest(options=options), self.assertRaises(SystemExit), patch('sys.stderr'):
                c.arguments(['1', *options])


if __name__ == "__main__":
    unittest.main()
