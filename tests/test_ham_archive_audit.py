"""Synthetic archive/label checks; no real dataset or remote execution."""
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from hpc.audit_ham_original import safe_members, read_ground_truth


class HamArchiveChecks(unittest.TestCase):
    def test_traversal_rejected(self):
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w') as z:z.writestr('../outside.jpg',b'x')
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as z:
            with self.assertRaisesRegex(ValueError,'Unsafe'):safe_members(z)

    def test_official_column_order_maps_to_existing_head_order(self):
        text='image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\nISIC_A,1,0,0,0,0,0,0\nISIC_B,0,0,0,1,0,0,0\n'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'labels.zip'
            with zipfile.ZipFile(path,'w') as z:z.writestr('folder/labels.csv',text)
            self.assertEqual(read_ground_truth(path),{'ISIC_A':'mel','ISIC_B':'akiec'})

    def test_ambiguous_label_rejected(self):
        text='image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\nISIC_A,1,1,0,0,0,0,0\n'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'labels.zip'
            with zipfile.ZipFile(path,'w') as z:z.writestr('labels.csv',text)
            with self.assertRaisesRegex(ValueError,'Non-one-hot'):read_ground_truth(path)
