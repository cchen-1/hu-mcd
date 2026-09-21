"""Slurm-only RDM ZIP -> verified scratch copy -> unmodified release audit.

No model, crop, diagnostic grouping, sign binarization, exclusions or sampling.
Metadata split indices are zero-based row indices, as the pinned official loader.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import stat
import zipfile

import numpy as np
from PIL import Image
from utils.run_tracking import atomic_json, sha256, utc_now

SIGNS=['pigment_network','streaks','pigmentation','regression_structures',
       'dots_and_globules','blue_whitish_veil','vascular_structures']


def validate_members(infos, byte_limit):
    names=set()
    for info in infos:
        name=PurePosixPath(info.filename)
        if name.is_absolute() or '..' in name.parts or '\\' in info.filename or info.filename in names:
            raise ValueError('Unsafe/duplicate archive member: '+info.filename)
        mode=info.external_attr>>16
        if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
            raise ValueError('Unsupported archive member type: '+info.filename)
        names.add(info.filename)
    if sum(x.file_size for x in infos)>byte_limit:
        raise ValueError('Expanded ZIP exceeds approved storage bound')


def audit(root, out):
    with (root/'meta/meta.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    splits={};indices={}
    for split in ('train','valid','test'):
        with (root/'meta'/f'{split}_indexes.csv').open(newline='') as f:
            ids=[int(r['indexes']) for r in csv.DictReader(f)]
        if len(set(ids))!=len(ids) or any(i<0 or i>=len(rows) for i in ids):
            raise ValueError('Invalid official split indices')
        splits[split]=ids
        for i in ids:
            if i in indices:raise ValueError('Official case rows overlap splits')
            indices[i]=split
    if set(indices)!=set(range(len(rows))):raise ValueError('Official split coverage incomplete')
    if len({r['case_num'] for r in rows})!=len(rows):raise ValueError('Duplicate case_num')
    records=[];issues=[];filegroups=defaultdict(list);pixelgroups=defaultdict(list)
    for i,row in enumerate(rows):
        for role in ('derm','clinic'):
            relative=PurePosixPath(row[role])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe metadata image path')
            p=root/'images'/relative
            rec=dict(row_index=i,case_num=row['case_num'],case_id=row['case_id'],split=indices[i],role=role,
                     relative_path=str(relative),diagnosis=row['diagnosis'],status='PENDING')
            try:
                rec['sha256']=sha256(p);rec['bytes']=p.stat().st_size
                with Image.open(p) as im:
                    im.load();a=np.asarray(im);rec.update(mode=im.mode,width=im.width,height=im.height,
                        pixel_sha256=hashlib.sha256(str((im.mode,a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest(),status='PASS')
                filegroups[rec['sha256']].append(len(records));pixelgroups[rec['pixel_sha256']].append(len(records))
                if rec['mode']!='RGB':issues.append(dict(kind='non_RGB',record=rec.copy(),action='Preserve; no conversion authorized'))
            except Exception as exc:
                rec.update(status='FAILED',error=str(exc));issues.append(dict(kind='image_read_failure',record=rec.copy()))
            records.append(rec)
        if i%100==0:atomic_json(out/'derm7pt_progress.json',dict(stage='image_audit',cases_complete=i+1,total_cases=len(rows),utc=utc_now()))
    atomic_json(out/'image_inventory.json',records)
    with (out/'cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['row_index','official_split']+list(rows[0]));w.writeheader()
        w.writerows(dict(row_index=i,official_split=indices[i],**r) for i,r in enumerate(rows))
    counts=[]
    # Raw diagnostic subtypes and categorical signs only. No operational melanoma
    # grouping (particularly metastasis) is silently fixed by this audit.
    for split,ids in [('all',list(range(len(rows)))),*splits.items()]:
        for diagnosis in sorted({rows[i]['diagnosis'] for i in ids}):
            selected=[rows[i] for i in ids if rows[i]['diagnosis']==diagnosis]
            counts.append(dict(split=split,diagnosis=diagnosis,cases=len(selected),
                signs={s:dict(Counter(r[s] for r in selected)) for s in SIGNS}))
    duplicates={kind:[[records[i] for i in ids] for ids in groups.values() if len(ids)>1]
                for kind,groups in [('file',filegroups),('decoded_pixels',pixelgroups)]}
    crossings=[g for g in duplicates['decoded_pixels'] if len({x['split'] for x in g})>1]
    atomic_json(out/'categorical_counts.json',counts);atomic_json(out/'duplicates.json',duplicates)
    report=dict(status='AUDITED_PENDING_ACCEPTANCE',case_count=len(rows),splits={k:len(v) for k,v in splits.items()},
        images=len(records),image_failures=sum(r['status']!='PASS' for r in records),issues=issues,
        exact_cross_split_pixel_groups=len(crossings),case_id_missing=sum(not r['case_id'] for r in rows),
        patient_identity='Not supplied as an independently verified patient identifier; no patient independence claim',
        cross_dataset_identity='NOT_CHECKED; requires accepted source images; no independence claim',
        diagnosis_and_sign_protocol='UNDECIDED; raw subtypes/categories preserved',
        preprocessing='Original decoded pixels; no border crop, conversion, exclusion, or model input construction',
        index_basis='Official metadata row indices (zero-based); original case_num retained')
    if report['image_failures'] or crossings:report['status']='AUDITED_WITH_BLOCKING_ISSUES'
    atomic_json(out/'derm7pt_audit.json',report)
    return report


def run(config,out):
    if not os.environ.get('SLURM_JOB_ID') or not re.fullmatch(r'bun[0-9]{3}',socket.gethostname().split('.')[0]):
        raise RuntimeError('Remote file processing requires a Bunya Slurm compute allocation')
    if config['authorization']!='APPROVED':raise ValueError('Missing staging authorization')
    source=Path(config['rdm_zip']);root=Path(config['scratch_parent'])/os.environ['SLURM_JOB_ID']
    root.mkdir(parents=True,exist_ok=False);raw=root/'raw';raw.mkdir()
    atomic_json(out/'derm7pt_progress.json',dict(stage='RDM_identity',rdm=str(source),scratch=str(root),utc=utc_now()))
    if source.stat().st_size!=config['zip_bytes'] or sha256(source)!=config['zip_sha256']:
        raise ValueError('RDM ZIP differs from local user-supplied archive; no extraction')
    dest=raw/'release_v0.zip';shutil.copyfile(source,dest)
    if sha256(dest)!=config['zip_sha256']:raise ValueError('Scratch ZIP copy hash mismatch')
    with zipfile.ZipFile(dest) as z:
        infos=z.infolist();validate_members(infos,config['max_expanded_bytes'])
        atomic_json(out/'zip_inventory.json',[dict(name=x.filename,bytes=x.file_size,crc=x.CRC) for x in infos])
        atomic_json(out/'derm7pt_progress.json',dict(stage='scratch_extract',utc=utc_now()))
        z.extractall(root/'extracted')
    release=root/'extracted/release_v0'
    report=audit(release,out)
    report.update(scratch_root=str(root),release_root=str(release),rdm_zip=str(source),zip_sha256=config['zip_sha256'],
                  recovery='Original ZIP remains unmodified on RDM; audit manifest provides scratch recovery identity')
    atomic_json(out/'derm7pt_audit.json',report)
    atomic_json(out/'metadata_sha256.json',{str(p.relative_to(release)):sha256(p) for p in sorted((release/'meta').glob('*.csv'))})
    # Images stay in scratch; local collection retrieves only audit records.
    # The immutable RDM ZIP plus inventory/hash manifests reconstruct all inputs.
    from hpc.workstream_runtime import event
    for issue in report['issues']:
        event(out,'derm7pt_audit','MED26-003-input-anomaly',issue,'Requires review before medical experiments','Preserve original input; no silent replacement',status='OPEN')
    if report['exact_cross_split_pixel_groups']:
        event(out,'derm7pt_audit','MED26-003-cross-split-duplicates',report['exact_cross_split_pixel_groups'],
            'Potential split leakage; medical experiments blocked','Retain groups for review',status='OPEN')
    return report
