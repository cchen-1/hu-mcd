"""Slurm-only official HAM archive extraction and corrected-ID audit.

No training, predictions, sampling, label correction, or automatic retry.
The upload plan pins archives, accepted manifest and this executable's hash.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import time
import zipfile


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)


def safe_members(archive):
    infos=archive.infolist();seen=set();total=0
    for info in infos:
        p=PurePosixPath(info.filename)
        if (p.is_absolute() or '..' in p.parts or '\\' in info.filename
                or info.filename in seen or (info.external_attr>>16)&0o170000==0o120000):
            raise ValueError('Unsafe or duplicate ZIP member: '+info.filename)
        seen.add(info.filename);total+=info.file_size
        if total>12*1024**3:raise ValueError('Uncompressed archive exceeds12GiB budget')
    return infos


def read_ground_truth(path):
    labels=['akiec','bcc','bkl','df','mel','nv','vasc']
    with zipfile.ZipFile(path) as z:
        infos=safe_members(z)
        names=[i.filename for i in infos if i.filename.lower().endswith('.csv')]
        if len(names)!=1:raise ValueError('Expected one official ground truth CSV')
        rows=list(csv.DictReader(io.StringIO(z.read(names[0]).decode('utf-8-sig'))))
    result={}
    for row in rows:
        image_id=row['image']
        if image_id in result:raise ValueError('Duplicate ground truth image ID')
        values=[float(row[c.upper()]) for c in labels]
        if any(v not in (0.,1.) for v in values) or sum(values)!=1:
            raise ValueError('Non-one-hot official diagnosis')
        result[image_id]=labels[values.index(1.)]
    return result


def audit(plan_path):
    if not (os.environ.get('SLURM_JOB_ID','').isdigit() and
            re.fullmatch(r'bun[0-9]{3}',socket.gethostname().split('.')[0])):
        raise RuntimeError('Remote processing requires a Slurm compute node')
    import numpy as np
    from PIL import Image
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_text())
    if sha256(__file__)!=plan['worker_sha256']:raise ValueError('Worker hash mismatch')
    root=Path(plan['root']);raw=root/'raw';out=root/'audit'/os.environ['SLURM_JOB_ID']
    out.mkdir(exist_ok=False);start=time.monotonic()
    report=dict(status='RUNNING',job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),
                plan_sha256=sha256(plan_path),worker_sha256=sha256(__file__),files=[],warnings=[])
    write_json(out/'plan.json',plan)
    def stage(name,**extra):
        write_json(out/'progress.json',dict(stage=name,elapsed_seconds=time.monotonic()-start,**extra))
        print(name,extra,flush=True)
        if time.monotonic()-start>1100:raise TimeoutError('Audit work budget reached; no extension')
    try:
        stage('VERIFY_TRANSFER')
        for entry in plan['files']:
            path=raw/entry['file']
            if path.stat().st_size!=entry['bytes'] or sha256(path)!=entry['sha256']:
                raise ValueError('Transferred file identity mismatch: '+entry['file'])
            report['files'].append(entry)
        m=plan['accepted_manifest'];mp=Path(m['path'])
        if sha256(mp)!=m['sha256']:raise ValueError('Accepted source manifest changed')
        with mp.open() as f:rows=list(csv.DictReader(f))
        by_id={r['image_id']:r for r in rows}
        if len(rows)!=10015 or len(by_id)!=10015:raise ValueError('Corrected source identity count')
        if dict(collections.Counter(r['split'] for r in rows))!={'train':8215,'val':573,'test':1227}:
            raise ValueError('Corrected source split counts')
        gt=read_ground_truth(raw/'ISIC2018_Task3_Training_GroundTruth.zip')
        if set(gt)!=set(by_id):raise ValueError('Official label IDs differ from corrected manifest')
        disagreements=[i for i in gt if gt[i]!=by_id[i]['class_name']]
        write_json(out/'label_disagreements.json',disagreements)
        if disagreements:raise ValueError('Official/corrected diagnosis mismatch; no silent relabel')
        with (raw/'ISIC2018_Task3_Training_LesionGroupings.csv').open(encoding='utf-8-sig') as f:
            group_rows=list(csv.DictReader(f))
        write_json(out/'official_grouping_schema.json',dict(columns=list(group_rows[0]) if group_rows else [],rows=len(group_rows)))
        # Preserve the official grouping fields verbatim; join by explicit image ID.
        if not group_rows or 'image' not in group_rows[0]:raise ValueError('Unknown official lesion grouping schema')
        groups={r['image']:r for r in group_rows}
        if len(groups)!=len(group_rows) or set(groups)!=set(by_id):raise ValueError('Official grouping identity mismatch')
        write_json(out/'official_groups_by_image.json',groups)
        target=root/'prepared'/('job-'+os.environ['SLURM_JOB_ID']);target.mkdir(exist_ok=False)
        stage('EXTRACT_CRC_AND_IDENTITIES')
        paths={};extras=[]
        with zipfile.ZipFile(raw/'ISIC2018_Task3_Training_Input.zip') as z:
            infos=safe_members(z)
            for info in infos:
                if info.is_dir():continue
                member=PurePosixPath(info.filename)
                if member.suffix.lower()!='.jpg':
                    # Read ancillary members too, so CRC is checked; save provenance text.
                    data=z.read(info);dest=target/'ancillary'/member
                    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
                    extras.append(info.filename);continue
                image_id=member.stem
                if image_id not in by_id or image_id in paths:raise ValueError('Unexpected/duplicate image member')
                dest=target/'images'/member.name;dest.parent.mkdir(exist_ok=True)
                with z.open(info) as src,dest.open('xb') as dst:
                    for b in iter(lambda:src.read(1024**2),b''):dst.write(b)
                paths[image_id]=dest
                if len(paths)%1000==0:stage('EXTRACT',images=len(paths))
        if set(paths)!=set(by_id):raise ValueError('Missing archive images')
        report['ancillary_members']=extras
        audit_rows=[];mismatch=[];modes=collections.Counter();sizes=collections.Counter()
        fields=list(rows[0])+['raw_path','raw_file_sha256','raw_pixel_sha256','width','height',
                              'mode','bicubic224_sha256','matches_accepted224']
        with (out/'raw_image_manifest.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for j,r in enumerate(rows):
                path=paths[r['image_id']]
                with Image.open(path) as im:
                    im.load();modes[im.mode]+=1;sizes[str(im.size)]+=1
                    if im.mode!='RGB':raise ValueError('Unexpected native mode; no conversion authorized')
                    a=np.asarray(im);small=np.asarray(im.resize((224,224),Image.Resampling.BICUBIC))
                    digest=hashlib.sha256(small.tobytes()).hexdigest();match=digest==r['pixel_sha256']
                    if not match:mismatch.append(r['image_id'])
                    writer.writerow(dict(r,raw_path=str(path),raw_file_sha256=sha256(path),
                        raw_pixel_sha256=hashlib.sha256(a.tobytes()).hexdigest(),width=im.width,height=im.height,
                        mode=im.mode,bicubic224_sha256=digest,matches_accepted224=match))
                if (j+1)%1000==0:stage('DECODE_AND_MAP',images=j+1,mismatches=len(mismatch))
        write_json(out/'accepted224_pixel_mismatches.json',mismatch)
        report.update(images=len(rows),split_counts=dict(collections.Counter(r['split'] for r in rows)),
                      modes=dict(modes),sizes=dict(sizes),accepted224_pixel_matches=len(rows)-len(mismatch),
                      accepted224_pixel_mismatches=len(mismatch),images_root=str(target/'images'),
                      raw_manifest=str(out/'raw_image_manifest.csv'),raw_manifest_sha256=sha256(out/'raw_image_manifest.csv'))
        # A mismatch is not automatically a bad image, but blocks silent equivalence claims/training.
        report['status']='PASS_DATA_AUDIT' if not mismatch else 'REVIEW_REQUIRED_PIXEL_DIFFERENCES'
        report['elapsed_seconds']=time.monotonic()-start
        write_json(out/'acceptance.json',report);stage(report['status'])
        if mismatch:raise SystemExit(4)
    except Exception as exc:
        report.update(status='FAILED_NO_RETRY',error=repr(exc),elapsed_seconds=time.monotonic()-start)
        write_json(out/'acceptance.json',report);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True)
    audit(parser.parse_args().plan)
