"""Audit the published DermaMNIST-C archive without changing images or splits.

Run on WSL, or inside a Slurm compute allocation on Bunya. Row identity follows
the authors' CreateNPZfromCSV.ipynb: CSV order, filtered separately by split.
This is dataset preparation only; it does not select samples or fit a model.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket

import numpy as np

LABELS = dict(akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6)
EXPECTED = {
    'dermamnist_corrected_224.npz': '84920fb70c83b234c295b6f0d4ae2bc0',
    'DermaMNIST-C.csv': '1c54bc9f483e97a9d7dbbde076a45900',
}


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            md5.update(block)
            sha.update(block)
    return dict(md5=md5.hexdigest(), sha256=sha.hexdigest(), bytes=path.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--metadata-only', action='store_true', help='Audit CSV identities only; never marks the image dataset ready')
    args = parser.parse_args()
    root = args.root.resolve()
    host = socket.gethostname().split('.')[0]
    if str(root).startswith('/scratch/'):
        if not os.environ.get('SLURM_JOB_ID') or not re.fullmatch(r'bun[0-9]{3}', host):
            raise RuntimeError('Remote inspection requires a Slurm compute node')
    output = root / 'manifests'
    output.mkdir(exist_ok=True)
    report = dict(dataset='DermaMNIST-C', release='zenodo:12739457', size=224,
                  status='RUNNING', host=host, job_id=os.environ.get('SLURM_JOB_ID'),
                  time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  script_sha256=hashes(Path(__file__))['sha256'], numpy=np.__version__,
                  files={}, splits={}, anomalies=[], label_map=LABELS,
                  selection='All published images; no new sampling or conversion',
                  mapping_basis='Published file checksums + pinned author CSV-order construction code + row-label agreement; original JPEG pixels not independently compared',
                  patient_separation='Not verified: metadata contains lesion IDs, not patient IDs',
                  near_duplicate_detection='Not performed',
                  exact_pixel_duplicate_check='NOT_RUN')

    def save():
        name = 'metadata_audit.json' if args.metadata_only else 'dataset_audit.json'
        tmp = output / (name + '.tmp')
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(output / name)

    save()
    try:
        for name, expected in EXPECTED.items():
            if args.metadata_only and name.endswith('.npz'):
                continue
            report['files'][name] = hashes(root / 'raw' / name)
            if report['files'][name]['md5'] != expected:
                raise ValueError('Published checksum mismatch: ' + name)
        source = json.loads((root / 'provenance' / 'author-source.json').read_text())
        if hashes(root / 'provenance' / 'CreateNPZfromCSV.ipynb')['sha256'] != source['notebook_sha256']:
            raise ValueError('Author construction-code hash mismatch')
        report['author_source'] = source
        with (root / 'raw' / 'DermaMNIST-C.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len({r['image_id'] for r in rows}) != len(rows):
            raise ValueError('Repeated image IDs in metadata')
        lesion_splits = collections.defaultdict(set)
        for row in rows:
            if row['split'] not in ('train', 'val', 'test') or row['dx'] not in LABELS or not row['lesion_id']:
                raise ValueError('Invalid split, class or lesion ID')
            lesion_splits[row['lesion_id']].add(row['split'])
        report['cross_split_lesion_ids'] = {k: sorted(v) for k, v in lesion_splits.items() if len(v) > 1}
        if args.metadata_only:
            if report['cross_split_lesion_ids']:
                raise ValueError('Cross-split lesion IDs in metadata')
            fields = ['split', 'array_row_expected', 'csv_row', 'image_id', 'lesion_id', 'class_name', 'label']
            with (output / 'metadata_index.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for split in ('train', 'val', 'test'):
                    metadata = [(i, r) for i, r in enumerate(rows) if r['split'] == split]
                    report['splits'][split] = dict(metadata_images=len(metadata), classes=dict(collections.Counter(r['dx'] for _, r in metadata)), lesions=len({r['lesion_id'] for _, r in metadata}))
                    for i, (csv_row, row) in enumerate(metadata):
                        writer.writerow(dict(split=split, array_row_expected=i, csv_row=csv_row, image_id=row['image_id'], lesion_id=row['lesion_id'], class_name=row['dx'], label=LABELS[row['dx']]))
            report['status'] = 'METADATA_ONLY_IMAGES_NOT_VERIFIED'
            report['metadata_index_sha256'] = hashes(output / 'metadata_index.csv')['sha256']
            print(report['status'], json.dumps(report['splits']), flush=True)
            return
        seen_pixels = collections.defaultdict(list)
        manifest_tmp = output / 'image_manifest.csv.tmp'
        with np.load(root / 'raw' / 'dermamnist_corrected_224.npz', allow_pickle=False) as data, manifest_tmp.open('w', newline='') as stream:
            expected_keys = {s + '_' + k for s in ('train', 'val', 'test') for k in ('images', 'labels')}
            if set(data.files) != expected_keys:
                raise ValueError('Unexpected NPZ arrays')
            writer = csv.DictWriter(stream, fieldnames=['split', 'array_row', 'csv_row', 'image_id', 'lesion_id', 'class_name', 'label', 'pixel_sha256', 'channels_identical'])
            writer.writeheader()
            for split in ('train', 'val', 'test'):
                metadata = [(i, r) for i, r in enumerate(rows) if r['split'] == split]
                images, labels = data[split + '_images'], data[split + '_labels']
                if images.shape != (len(metadata), 224, 224, 3) or images.dtype != np.uint8:
                    raise ValueError('Unexpected image shape or dtype: ' + split)
                if labels.shape != (len(metadata), 1) or labels.dtype != np.uint8:
                    raise ValueError('Unexpected label shape or dtype: ' + split)
                if not np.array_equal(labels[:, 0], [LABELS[r['dx']] for _, r in metadata]):
                    raise ValueError('CSV/NPZ label-order mismatch: ' + split)
                gray = 0
                for i, (csv_row, row) in enumerate(metadata):
                    pixels = images[i]
                    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                    same = bool(np.array_equal(pixels[:, :, 0], pixels[:, :, 1]) and np.array_equal(pixels[:, :, 1], pixels[:, :, 2]))
                    gray += int(same)
                    seen_pixels[digest].append(dict(split=split, array_row=i, image_id=row['image_id']))
                    writer.writerow(dict(split=split, array_row=i, csv_row=csv_row, image_id=row['image_id'], lesion_id=row['lesion_id'], class_name=row['dx'], label=int(labels[i, 0]), pixel_sha256=digest, channels_identical=same))
                report['splits'][split] = dict(images=len(metadata), shape=list(images.shape), dtype=str(images.dtype), classes=dict(collections.Counter(r['dx'] for _, r in metadata)), lesions=len({r['lesion_id'] for _, r in metadata}), channels_identical=gray)
                if gray:
                    report['anomalies'].append(dict(kind='identical_rgb_channels', split=split, count=gray, handling='Retained without modification'))
                print(split, json.dumps(report['splits'][split]), flush=True)
                del images, labels
                save()
        report['exact_pixel_duplicate_groups'] = [v for v in seen_pixels.values() if len(v) > 1]
        report['exact_pixel_duplicate_check'] = 'COMPLETE'
        report['cross_split_pixel_duplicate_groups'] = [v for v in report['exact_pixel_duplicate_groups'] if len({r['split'] for r in v}) > 1]
        if report['exact_pixel_duplicate_groups']:
            report['anomalies'].append(dict(kind='exact_pixel_duplicates', groups=len(report['exact_pixel_duplicate_groups']), handling='Recorded; no images removed'))
        manifest_tmp.replace(output / 'image_manifest.csv')
        report['image_manifest_sha256'] = hashes(output / 'image_manifest.csv')['sha256']
        if report['cross_split_lesion_ids'] or report['cross_split_pixel_duplicate_groups']:
            raise ValueError('Cross-split overlap found; do not use for experiments until reviewed')
        report['status'] = 'PASS'
    except Exception as exc:
        report['status'] = 'FAILED'
        report['anomalies'].append(dict(kind=type(exc).__name__, message=str(exc), handling='Dataset not approved for experiments'))
        raise
    finally:
        save()
    print('DATASET_AUDIT_PASS', flush=True)


if __name__ == '__main__':
    main()
