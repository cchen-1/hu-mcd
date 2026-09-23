"""Materialize approved raw-HAM classifier inputs and Derm7pt R101 fit inputs.

CPU-only Slurm task; no weights, predictions, SAM, fitting or label changes.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time
import warnings

from hpc.audit_ham_original import sha256, write_json


def run(plan_path):
    if not (os.environ.get('SLURM_JOB_ID','').isdigit() and
            re.fullmatch(r'bun[0-9]{3}',socket.gethostname().split('.')[0])):
        raise RuntimeError('Slurm compute node required')
    plan_path=Path(plan_path);p=json.loads(plan_path.read_text())
    if p.get('sam_input_policy') != {
        'short_side_cap':300, 'preserve_aspect_ratio':True,
        'extra_square_resize':False, 'use_classifier_cache':False,
    }:
        raise ValueError('Only released pre-SAM geometry is authorized')
    root=Path(__file__).resolve().parents[1]
    for name,digest in p['source_sha256'].items():
        if sha256(root/name)!=digest:raise ValueError('Code mismatch: '+name)
    out=Path(p['output_root'])/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    write_json(out/'actual_plan.json',p);started=time.monotonic()
    def warning(message,category,filename,lineno,file=None,line=None):
        with (out/'warnings.jsonl').open('a') as f:
            f.write(json.dumps(dict(message=str(message),category=category.__name__,filename=filename,line=lineno))+'\n')
    warnings.showwarning=warning;warnings.simplefilter('always')
    status=dict(status='RUNNING',job_id=os.environ['SLURM_JOB_ID'],execution_commit=p['execution_commit'],
                plan_sha256=sha256(plan_path),model_execution=False)
    def progress(stage,**details):
        write_json(out/'progress.json',dict(stage=stage,elapsed_seconds=time.monotonic()-started,**details))
        print(stage,details,flush=True)
        if time.monotonic()-started>1100:raise TimeoutError('Preparation budget exceeded; no extension')
        if sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>4*1024**3:
            raise RuntimeError('Preparation output exceeds4GiB')
    try:
        for name,entry in p['inputs'].items():
            if sha256(entry['path'])!=entry['sha256']:raise ValueError('Input binding: '+name)
        import numpy as np
        import torch
        import torchvision
        import PIL
        from PIL import Image
        import classes
        from hpc.ham_input_preparation import (released_classifier_pixels, released_full_image_input,
                                               save_released_presam_input)
        torch.set_num_threads(2)
        write_json(out/'environment.json',dict(torch=torch.__version__,torchvision=torchvision.__version__,
                   numpy=np.__version__,pillow=PIL.__version__,host=socket.gethostname()))
        a=json.loads(Path(p['inputs']['ham_audit']['path']).read_text())
        if a['status']!='PASS_DATA_AUDIT':raise ValueError('Raw HAM audit not accepted')
        with Path(p['inputs']['ham_manifest']['path']).open() as f:rows=list(csv.DictReader(f))
        if len(rows)!=10015 or len({r['image_id'] for r in rows})!=10015:raise ValueError('Source identities')
        counts=dict(train=8215,val=573,test=1227)
        cfg=dict(input_size=(3,224,224),mean=(.485,.456,.406),std=(.229,.224,.225))
        arrays={};records=[];adapter=[];different=0
        for split,count in counts.items():
            selected=[r for r in rows if r['split']==split]
            if len(selected)!=count or [int(r['array_row']) for r in selected]!=list(range(count)):
                raise ValueError('Source split/order changed')
            images=np.empty((count,224,224,3),np.uint8)
            labels=np.empty((count,1),np.uint8)
            for i,r in enumerate(selected):
                path=Path(r['raw_path'])
                if sha256(path)!=r['raw_file_sha256']:raise ValueError('Raw image changed: '+r['image_id'])
                pixels,identity=released_classifier_pixels(path,cfg)
                if identity['pre_sam_size_wh']!=[400,300]:raise ValueError('Unexpected audited HAM geometry')
                images[i]=pixels;labels[i,0]=int(r['label'])
                ph=hashlib.sha256(pixels.tobytes()).hexdigest();changed=ph!=r['pixel_sha256'];different+=int(changed)
                records.append(dict(r,pixel_sha256=ph,prior_direct224_sha256=r['pixel_sha256'],
                                    changed_from_direct224=changed,pre_sam_width=400,pre_sam_height=300))
                # Check the actual cache arithmetic against the released Dataset path,
                # on first8 rows of each split, without a model or GPU.
                if i<8:
                    actual,_=released_full_image_input(path,cfg)
                    x=torch.from_numpy(pixels.copy()).permute(2,0,1).float()/255
                    x=(x-torch.tensor(cfg['mean'])[:,None,None])/torch.tensor(cfg['std'])[:,None,None]
                    eq=torch.equal(actual,x);adapter.append(dict(image_id=r['image_id'],split=split,exact_equal=eq))
                    if not eq:raise ValueError('Cached classifier input differs from released Dataset')
                if (i+1)%1000==0:progress('HAM_INPUTS',split=split,images=i+1)
            arrays[split+'_images']=images;arrays[split+'_labels']=labels
        cache=out/'ham_released_geometry_224.npz'
        np.savez(cache,**arrays);del arrays
        write_json(out/'classifier_cache_role.json',dict(
            file=cache.name,use='CLASSIFIER_TRAINING_ONLY',allowed_as_sam_input=False,
            underlying_source='original HAM JPEG',
            transform='released short-side300 load then released classifier224 resize'))
        with (out/'classifier_image_manifest.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        write_json(out/'cache_tensor_checks.json',adapter)
        progress('HAM_CACHE_COMPLETE',different_from_direct224=different)
        frozen=json.loads(Path(p['inputs']['r101']['path']).read_text());cohort=frozen['rows']
        if len(cohort)!=101 or len({r['image_id'] for r in cohort})!=101:raise ValueError('Frozen R101 identity/count')
        dermdir=out/'derm7pt_r101_presam';dermdir.mkdir();fit=[]
        for i,r in enumerate(cohort):
            source=Path(r['source_path']);inv=r['inventory']
            if sha256(source)!=inv['sha256']:raise ValueError('Derm7pt original file changed')
            path=dermdir/(r['image_id']+'.png')
            geometry=save_released_presam_input(source,path)
            fit.append(dict(r,fit_order=i,input_path=str(path),input_sha256=sha256(path),
                            original_size_wh=geometry['original_size_wh'],input_size_wh=geometry['input_size_wh'],
                            role='DISCOVERY_FIT_NO_HELD_OUT',preprocessing=geometry['preprocessing'],
                            extra_square_resize_before_sam=False,uses_classifier_cache=False))
        write_json(out/'derm7pt_fit_manifest.json',dict(count=101,held_out=0,rows=fit,
                   no_label_or_sign_guided_tuning=True,source_cohort_sha256=p['inputs']['r101']['sha256']))
        sizes=dict(collections.Counter(str(tuple(r['input_size_wh'])) for r in fit))
        if sizes!=p['expected_derm_presam_sizes']:
            raise ValueError('Derm7pt pre-SAM geometry differs from frozen original-size expectations')
        progress('R101_INPUTS_COMPLETE',images=101)
        status.update(status='PREPARED_PENDING_ACCEPTANCE',ham_images=10015,ham_split_counts=counts,
                      ham_changed_from_direct224=different,derm_fit_images=101,derm_held_out=0,
                      tensor_checks=len(adapter),source_masks_reused=False,derm_presam_sizes=sizes,
                      elapsed_seconds=time.monotonic()-started)
    except Exception as exc:
        status.update(status='FAILED_NO_RETRY',error=repr(exc));raise
    finally:
        write_json(out/'status.json',status)
        files={str(f.relative_to(out)):dict(bytes=f.stat().st_size,sha256=sha256(f))
               for f in out.rglob('*') if f.is_file() and f.name!='artifacts.json'}
        write_json(out/'artifacts.json',dict(job_id=os.environ['SLURM_JOB_ID'],execution_commit=p['execution_commit'],
                                           status=status['status'],files=files))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True)
    run(parser.parse_args().plan)
