"""Audit the author's older OneDrive archive; never approve it as the newer release.

No sampling, conversion, inference, or repartitioning. Bunya execution requires
Slurm. Image identity is inferred from the pinned author's CSV-order construction
and matching labels, not independently verified against original JPEGs.
"""
import argparse
import collections
import csv
import datetime
import json
import os
from pathlib import Path
import re
import socket
import hashlib

import numpy as np

from prepare_dermamnist import hashes, LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    host = socket.gethostname().split('.')[0]
    if str(root).startswith('/scratch/') and not (
        os.environ.get('SLURM_JOB_ID') and re.fullmatch(r'bun[0-9]{3}', host)
    ):
        raise RuntimeError('Remote processing requires a Slurm compute node')
    out = root / 'manifests'
    out.mkdir(exist_ok=True)
    report = dict(status='RUNNING', source='author OneDrive older corrected split',
                  host=host, job_id=os.environ.get('SLURM_JOB_ID'),
                  time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  script_sha256=hashes(Path(__file__))['sha256'],
                  helper_sha256=hashes(Path(__file__).with_name('prepare_dermamnist.py'))['sha256'],
                  files={}, splits={}, anomalies=[],
                  mapping_basis='Pinned author CSV-order construction + published legacy checksum + full label-order agreement; original JPEG comparison not performed',
                  patient_independence='Not verified: no patient IDs',
                  near_duplicate_search='No new exhaustive search; checked all author-confirmed duplicate pairs',
                  experiment_approved=False)
    try:
        expected = {'dermamnist_corrected_224.npz': '947968c463c25c5d79b946d4311afffb',
                    'legacy_metadata.csv': '504579018fc1d910c32f74f3de9ffbc0',
                    'current_metadata.csv': '1c54bc9f483e97a9d7dbbde076a45900'}
        for name, md5 in expected.items():
            report['files'][name] = hashes(root / 'raw' / name)
            if report['files'][name]['md5'] != md5:
                raise ValueError('Source checksum mismatch: ' + name)
        duplicates = root / 'provenance' / 'author_confirmed_duplicates.csv'
        report['files'][duplicates.name] = hashes(duplicates)
        if report['files'][duplicates.name]['sha256'] != 'fafe916c6a97c4d6c6de3c7c93eae1f7eb601680692c25ea82c8ef595fca73c7':
            raise ValueError('Author duplicate evidence checksum mismatch')
        source = json.loads((root / 'provenance' / 'author-source.json').read_text())
        if hashes(root / 'provenance' / 'CreateNPZfromCSV.ipynb')['sha256'] != source['notebook_sha256']:
            raise ValueError('Author construction code mismatch')
        report['author_source'] = source
        rows = list(csv.DictReader((root / 'raw' / 'legacy_metadata.csv').open()))
        current = {r['image_id']: r for r in csv.DictReader((root / 'raw' / 'current_metadata.csv').open())}
        legacy = {r['image_id']: r for r in rows}
        if len(legacy) != len(rows) or set(legacy) != set(current):
            raise ValueError('Image ID sets are not identical and unique')
        report['changed_split_images'] = [dict(image_id=r['image_id'], old=r['split'], new=current[r['image_id']]['split']) for r in rows if r['split'] != current[r['image_id']]['split']]
        lesion_splits = collections.defaultdict(set)
        for row in rows:
            lesion_splits[row['lesion_id']].add(row['split'])
        report['cross_split_lesion_ids'] = {k: sorted(v) for k, v in lesion_splits.items() if len(v) > 1}
        pairs = []
        for pair in csv.DictReader(duplicates.open()):
            a, b = pair['from_img'].removesuffix('.jpg'), pair['to_img'].removesuffix('.jpg')
            if legacy[a]['split'] != legacy[b]['split']:
                pairs.append(dict(image_a=a, image_b=b, split_a=legacy[a]['split'], split_b=legacy[b]['split']))
        report['author_confirmed_cross_split_duplicate_pairs'] = pairs
        seen = collections.defaultdict(list)
        with np.load(root / 'raw' / 'dermamnist_corrected_224.npz', allow_pickle=False) as data, (out / 'image_manifest.csv').open('w', newline='') as stream:
            if set(data.files) != {s+'_'+k for s in ('train','val','test') for k in ('images','labels')}:
                raise ValueError('Unexpected NPZ arrays')
            writer = csv.DictWriter(stream, fieldnames=['split','array_row','image_id','lesion_id','class_name','label','pixel_sha256','current_split','channels_identical'])
            writer.writeheader()
            for split in ('train','val','test'):
                metadata = [r for r in rows if r['split'] == split]
                images, labels = data[split+'_images'], data[split+'_labels']
                if images.shape != (len(metadata),224,224,3) or images.dtype != np.uint8:
                    raise ValueError('Image shape/type mismatch: '+split)
                if labels.shape != (len(metadata),1) or labels.dtype != np.uint8 or not np.array_equal(labels[:,0], [LABELS[r['dx']] for r in metadata]):
                    raise ValueError('Label/CSV order mismatch: '+split)
                gray = 0
                for i, row in enumerate(metadata):
                    pixels = images[i]
                    h = hashlib.sha256(pixels.tobytes()).hexdigest()
                    same = bool(np.array_equal(pixels[:,:,0],pixels[:,:,1]) and np.array_equal(pixels[:,:,1],pixels[:,:,2]))
                    gray += same
                    seen[h].append(dict(split=split,array_row=i,image_id=row['image_id']))
                    writer.writerow(dict(split=split,array_row=i,image_id=row['image_id'],lesion_id=row['lesion_id'],class_name=row['dx'],label=int(labels[i,0]),pixel_sha256=h,current_split=current[row['image_id']]['split'],channels_identical=same))
                report['splits'][split] = dict(images=len(metadata),lesions=len({r['lesion_id'] for r in metadata}),classes=dict(collections.Counter(r['dx'] for r in metadata)),channels_identical=gray)
                print(split, json.dumps(report['splits'][split]), flush=True)
                del images, labels
        report['exact_pixel_duplicate_groups'] = [g for g in seen.values() if len(g)>1]
        report['cross_split_exact_pixel_duplicates'] = [g for g in report['exact_pixel_duplicate_groups'] if len({x['split'] for x in g})>1]
        report['image_manifest_sha256'] = hashes(out / 'image_manifest.csv')['sha256']
        report['integrity_status'] = 'PASS'
        report['status'] = 'INTEGRITY_PASS_SCIENTIFIC_USE_BLOCKED'
        report['anomalies'] = [dict(kind='older_source_version',handling='Separate private storage; not substituted for Zenodo release'),dict(kind='author_confirmed_cross_split_duplicates',count=len(pairs),handling='No experiment approved; retain all images and evidence'),dict(kind='within_split_exact_pixel_duplicates',groups=len(report['exact_pixel_duplicate_groups']),handling='Recorded without deleting images')]
    except Exception as exc:
        report['status'] = 'FAILED'
        report['anomalies'].append(dict(kind=type(exc).__name__,message=str(exc)))
        raise
    finally:
        (out / 'dataset_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
