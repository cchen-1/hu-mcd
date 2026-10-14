"""Reassemble a candidate from the older archive; accept only the release MD5.

This preserves every pixel and uses the author's current CSV order and splits.
It never overwrites or promotes the candidate automatically.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import socket
import numpy as np
from prepare_dermamnist import hashes, LABELS, EXPECTED


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('legacy_root', type=Path)
    p.add_argument('candidate', type=Path)
    a = p.parse_args()
    if any(str(x.resolve()).startswith('/scratch/') for x in [a.legacy_root, a.candidate]):
        if not os.environ.get('SLURM_JOB_ID') or not re.fullmatch(r'bun[0-9]{3}',socket.gethostname().split('.')[0]):
            raise RuntimeError('Remote processing requires Slurm')
    root = a.legacy_root
    audit = json.loads((root/'manifests/dataset_audit.json').read_text())
    if audit.get('integrity_status') != 'PASS':
        raise ValueError('Legacy integrity audit required')
    for name, expected in [('dermamnist_corrected_224.npz','947968c463c25c5d79b946d4311afffb'),('legacy_metadata.csv','504579018fc1d910c32f74f3de9ffbc0'),('current_metadata.csv',EXPECTED['DermaMNIST-C.csv'])]:
        if hashes(root/'raw'/name)['md5'] != expected:
            raise ValueError('Input changed: '+name)
    if a.candidate.exists():
        raise FileExistsError(a.candidate)
    old = list(csv.DictReader((root/'raw/legacy_metadata.csv').open()))
    new = list(csv.DictReader((root/'raw/current_metadata.csv').open()))
    if len({r['image_id'] for r in old}) != len(old) or {r['image_id'] for r in old} != {r['image_id'] for r in new}:
        raise ValueError('Image identity mismatch')
    data = np.load(root/'raw/dermamnist_corrected_224.npz',allow_pickle=False)
    arrays = {s:data[s+'_images'] for s in ['train','val','test']}
    mapping = {r['image_id']:(s,i) for s in arrays for i,r in enumerate([r for r in old if r['split']==s])}
    result = {}
    for s in arrays:
        rows = [r for r in new if r['split']==s]
        result[s+'_images'] = np.stack([arrays[mapping[r['image_id']][0]][mapping[r['image_id']][1]] for r in rows])
        result[s+'_labels'] = np.asarray([LABELS[r['dx']] for r in rows],dtype=np.uint8).reshape(-1,1)
    with a.candidate.open('xb') as f:
        np.savez_compressed(f,**result)
    checksum = hashes(a.candidate)
    match = checksum['md5'] == EXPECTED['dermamnist_corrected_224.npz']
    report = dict(status='BYTE_IDENTICAL_RELEASE_RECOVERED' if match else 'CANDIDATE_NOT_APPROVED',candidate=str(a.candidate),hashes=checksum,expected_md5=EXPECTED['dermamnist_corrected_224.npz'],script_sha256=hashes(Path(__file__))['sha256'],numpy=np.__version__,images_changed=False,rule='Pinned author current CSV order and splits; no new sampling',notebook_basis=audit['author_source'])
    a.candidate.with_suffix('.recovery.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
