"""Proposed H: exact/canonical and perceptual candidate retrieval, never exclusion.

All comparisons are source image -> derm image. D4 transforms are retrieval-only,
not classifier preprocessing. Distances/nearest neighbours are not identity labels.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.fft import dctn

from hpc.medical_protocol import execution_gate, bound_inputs, manifest_rows
from utils.run_tracking import atomic_json, sha256, utc_now

POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def pixel_key(a):
    a = np.asarray(a)
    return hashlib.sha256(str((a.shape, str(a.dtype))).encode() + a.tobytes()).hexdigest()


def orientations(a):
    for reflected in (False, True):
        base = np.fliplr(a) if reflected else a
        for k in range(4):
            yield ('mirror_' if reflected else '') + 'rot' + str(k * 90), np.rot90(base, k)


def hashes(a):
    im = Image.fromarray(a).convert('L')
    small = np.asarray(im.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = dctn(small, type=2, norm=None)[:8, :8]
    # Explicit pHash convention: median of all 64 low-frequency coefficients, DC included.
    p = np.packbits((low > np.median(low)).ravel(), bitorder='big').tobytes()
    grad = np.asarray(im.resize((9, 8), Image.Resampling.LANCZOS))
    d = np.packbits((grad[:, 1:] > grad[:, :-1]).ravel(), bitorder='big').tobytes()
    return np.frombuffer(p, dtype=np.uint8).copy(), np.frombuffer(d, dtype=np.uint8).copy()


def distances(source, queries):
    """At most 8x10015x8 bytes per hash; no all-image pair tensor."""
    return POPCOUNT[np.bitwise_xor(queries[:, None, :], source[None, :, :])].sum(axis=2)


def candidate_indices(dp, dd, exact, threshold=6):
    pm, dm = dp.min(axis=0), dd.min(axis=0)
    chosen = set(np.flatnonzero((pm <= threshold) | (dm <= threshold))) | set(exact)
    # Source order is deterministic; nearest ties choose the first source image ID.
    chosen.add(int(pm.argmin()))
    chosen.add(int(dm.argmin()))
    return sorted(chosen), pm, dm


def run(config, output):
    execution_gate(config, 'medical-overlap')
    p = config['protocol']; out = Path(output); started = time.monotonic()
    bound_inputs(p, out)
    rows = manifest_rows(p['inputs']['source_manifest']['path'])
    inventory = json.loads(Path(p['inputs']['derm_inventory']['path']).read_text())
    derm = sorted([r for r in inventory if r['role'] == 'derm'], key=lambda r: int(r['row_index']))
    if len(derm) != 1011 or [int(r['row_index']) for r in derm] != list(range(1011)):
        raise ValueError('Accepted derm inventory rows differ')
    # Stable lexical image-ID ordering makes tied nearest-neighbour selection explicit.
    entries = sorted([r for rr in rows.values() for r in rr], key=lambda r: r['image_id'])
    native = {}; source_p = []; source_d = []
    def budget():
        if time.monotonic() - started > p['work_seconds_ceiling']:
            raise TimeoutError('H processing ceiling reached; incomplete, no retry')
        if sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) > p['output_bytes_ceiling']:
            raise RuntimeError('H output ceiling reached; incomplete candidates retained')
    with np.load(p['inputs']['source_npz']['path'], allow_pickle=False) as z:
        # Only hold one original source split at a time, then restore stable order.
        by_id = {}
        for split, rr in rows.items():
            arr = z[split + '_images']; labels = z[split + '_labels'].reshape(-1)
            if arr.shape != (len(rr), 224, 224, 3) or arr.dtype != np.uint8:
                raise ValueError('Source array geometry/dtype changed')
            if not np.array_equal(labels, [int(r['label']) for r in rr]):
                raise ValueError('Source manifest/array labels differ')
            for row, image in zip(rr, arr):
                ph, dh = hashes(image)
                by_id[row['image_id']] = (pixel_key(image), ph, dh)
            budget()
        for i, row in enumerate(entries):
            key, ph, dh = by_id[row['image_id']]
            native.setdefault(key, []).append(i); source_p.append(ph); source_d.append(dh)
    source_p, source_d = np.array(source_p), np.array(source_d)
    np.savez_compressed(out / 'source_hashes.npz', phash=source_p, dhash=source_d,
                        image_id=np.array([r['image_id'] for r in entries]))
    atomic_json(out / 'source_index.json', entries)
    fields = ['derm_row_index','derm_case_num','derm_split','derm_path','source_image_id',
              'source_split','source_array_row','source_lesion_id','source_label',
              'native_decoded_equal','canonical_224_equal','phash_distance','dhash_distance',
              'phash_orientation','dhash_orientation','threshold_candidate','nearest_phash',
              'nearest_dhash','status']
    total = 0; previews = []; derm_hashes = []
    with gzip.open(out / 'all_candidates.csv.gz', 'wt', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for drow in derm:
            rel = Path(drow['relative_path'])
            if rel.is_absolute() or '..' in rel.parts or drow['status'] != 'PASS':
                raise ValueError('Unsafe or unaccepted derm entry')
            path = Path(p['derm_root']) / 'images' / rel
            if sha256(path) != drow['sha256']:
                raise ValueError('Derm file differs from accepted inventory: ' + str(rel))
            with Image.open(path) as im:
                if im.mode != 'RGB':
                    raise ValueError('Unexpected image mode; no implicit conversion')
                image = np.asarray(im).copy()
                canonical = np.asarray(im.resize((224,224), Image.Resampling.BICUBIC))
            native_match = native.get(pixel_key(image), [])
            canonical_match = native.get(pixel_key(canonical), [])
            pairs = [(name, hashes(a)) for name,a in orientations(image)]
            pp = np.array([v[0] for _,v in pairs]); dd = np.array([v[1] for _,v in pairs])
            dp, dh = distances(source_p, pp), distances(source_d, dd)
            chosen, pm, dm = candidate_indices(dp, dh, native_match + canonical_match, p['distance_threshold'])
            derm_hashes.append(dict(row_index=drow['row_index'], phash=pp.tolist(), dhash=dd.tolist(),
                                    native_key=pixel_key(image), canonical_key=pixel_key(canonical)))
            for idx in chosen:
                sr = entries[idx]
                rec = dict(derm_row_index=drow['row_index'], derm_case_num=drow['case_num'],
                           derm_split=drow['split'], derm_path=str(path), source_image_id=sr['image_id'],
                           source_split=sr['split'], source_array_row=sr['array_row'],
                           source_lesion_id=sr['lesion_id'], source_label=sr['label'],
                           native_decoded_equal=idx in native_match, canonical_224_equal=idx in canonical_match,
                           phash_distance=int(pm[idx]), dhash_distance=int(dm[idx]),
                           phash_orientation=pairs[int(dp[:,idx].argmin())][0],
                           dhash_orientation=pairs[int(dh[:,idx].argmin())][0],
                           threshold_candidate=bool(min(pm[idx], dm[idx]) <= p['distance_threshold']),
                           nearest_phash=idx == int(pm.argmin()), nearest_dhash=idx == int(dm.argmin()),
                           status='CANDIDATE_UNREVIEWED_NOT_IDENTITY')
                writer.writerow(rec); total += 1
                if len(previews) < p['first_review_queue_limit']:
                    previews.append(rec)
            f.flush(); budget()
            atomic_json(out / 'medical_progress.json', dict(stage='H_candidate_retrieval',
                        derm_complete=len(derm_hashes), derm_total=len(derm), candidates=total, utc=utc_now()))
    atomic_json(out / 'derm_hashes.json', derm_hashes)
    with (out / 'first_review_queue.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(previews)
    budget()
    return dict(status='COMPLETED_PENDING_ACCEPTANCE_AND_CANDIDATE_REVIEW', candidates=total,
                source_images=len(entries), derm_images=len(derm), pair_count=len(entries)*len(derm),
                transformed_pairs_per_hash=8*len(entries)*len(derm),
                no_match_does_not_establish_independence=True, exclusions=0,
                first_review_rule='First 200 pairs in derm metadata row/source image-ID order; all pairs retained',
                raw_encoded_byte_comparison='NOT_APPLICABLE: source archive stores decoded image arrays',
                elapsed_seconds=time.monotonic()-started)
