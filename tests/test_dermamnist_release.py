"""Critical data-preparation gates; tiny fixtures, no network or HPC execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from hpc.dermamnist_release import inventory,fetch_once,publish_archive,verify_archive,RELEASE


class ReleaseTests(unittest.TestCase):
    def test_reuse_only_never_falls_back_to_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with patch('hpc.dermamnist_release.subprocess.run') as call:
                with self.assertRaisesRegex(RuntimeError,'network download disabled'):
                    fetch_once('dermamnist_corrected_224.npz',p/'file.npz',p,60,allow_download=False)
                call.assert_not_called()
            self.assertEqual(list(p.iterdir()),[])

    def test_inventory_requires_full_hash_not_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);root=p/'source';root.mkdir();out=p/'out';out.mkdir()
            (root/'dermamnist_corrected_224.npz').write_bytes(b'bad')
            (root/'dermamnist_corrected_224.renamed.npz').write_bytes(b'yes')
            with patch('hpc.dermamnist_release.FILES',{'dermamnist_corrected_224.npz':(3,hashlib.md5(b'yes').hexdigest())}):
                rows=inventory([root],out)
            self.assertEqual(sum(r.get('exact_match',False) for r in rows),1)
            self.assertTrue(any('renamed' in r['path'] and r['exact_match'] for r in rows))

    def test_inventory_access_failure_is_not_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            def denied(*args,**kwargs):
                kwargs['onerror'](PermissionError('collection denied'))
                return iter([])
            with patch('hpc.dermamnist_release.os.walk',side_effect=denied):
                with self.assertRaisesRegex(RuntimeError,'Incomplete'):inventory([p],p)
            self.assertTrue(json.loads((p/'existing_file_inventory.json').read_text())['errors'])

    def test_download_failure_never_retries_or_promotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);target=p/'file.npz'
            with patch('hpc.dermamnist_release.subprocess.run',return_value=subprocess.CompletedProcess([],28,'','time cap')) as call:
                with self.assertRaisesRegex(RuntimeError,'Download failed'):fetch_once('dermamnist_corrected_224.npz',target,p,60)
                self.assertEqual(call.call_count,1);args=call.call_args.args[0]
                self.assertEqual(args[args.index('--retry')+1],'0')
            self.assertFalse(target.exists())

    def test_archive_checks_copy_and_reuses_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);source=p/'scratch';(source/'raw').mkdir(parents=True);parent=p/'rdm';parent.mkdir()
            name='dermamnist_corrected_224.npz';(source/'raw'/name).write_bytes(b'yes')
            tiny={name:(3,hashlib.md5(b'yes').hexdigest())}
            with patch('hpc.dermamnist_release.FILES',tiny):
                final=parent/RELEASE;publish_archive(source,final,'1',p);verify_archive(final)
                r=publish_archive(source,final,'2',p);self.assertEqual(r['status'],'REUSED_VERIFIED_ARCHIVE')
                (final/'raw'/name).write_bytes(b'bad')
                with self.assertRaisesRegex(ValueError,'differs'):publish_archive(source,final,'3',p)
                self.assertEqual((final/'raw'/name).read_bytes(),b'bad')


if __name__=='__main__':unittest.main()
