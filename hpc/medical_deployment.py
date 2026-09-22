"""CPU-only release/readability checks inside the approved shared deployment job.

No dataset decoding, model loading or inference. H and M readiness are independent.
Execution workers bind complete input/weight hashes before their actual workload.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

from hpc.medical_protocol import validate_config
from utils.run_tracking import atomic_json, sha256, utc_now


def readiness(root, mode):
    config_path = root/'configs/medical'/(mode+'.approved.json')
    config = json.loads(config_path.read_text())
    protocol = validate_config(config, mode, approved=True)
    report = dict(status='PASS', protocol_sha256=config['protocol_sha256'],
                  config_sha256=sha256(config_path), inputs={}, errors=[])
    for key, entry in protocol['inputs'].items():
        path = Path(entry['path'])
        try:
            with path.open('rb') as f:
                if not f.read(1): raise ValueError('Empty required input')
            report['inputs'][key] = dict(path=str(path), bytes=path.stat().st_size,
                                         readable=True, full_hash='Worker gate, not repeated here')
        except (OSError, ValueError) as exc:
            report['errors'].append(dict(input=key, error=str(exc)))
    if mode == 'medical-overlap':
        try:
            # One role-linked readability check; no image decoding or whole-data scan.
            inv=json.loads(Path(protocol['inputs']['derm_inventory']['path']).read_text())
            row=next(r for r in inv if r['role']=='derm')
            rel=Path(row['relative_path'])
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe accepted inventory path')
            with (Path(protocol['derm_root'])/'images'/rel).open('rb') as f:
                if not f.read(1):raise ValueError('Empty derm image')
            report['derm_root_readable']=True
        except (OSError, ValueError, StopIteration) as exc:
            report['errors'].append(dict(input='derm_root',error=str(exc)))
    if mode == 'medical-classifier':
        try:
            if importlib.metadata.version('timm')!='0.6.13':raise ValueError('Unpinned timm')
            dist=importlib.metadata.distribution('timm')
            report['installed_model_sources']={}
            for name in ('resnet.py','sal_layers.py'):
                path=Path(dist.locate_file('timm/models/'+name))
                actual=sha256(path); expected=sha256(root/'input_masking'/name)
                report['installed_model_sources'][name]=dict(actual=actual,expected=expected)
                if actual!=expected:raise ValueError('Installed model source differs: '+name)
        except (OSError, ValueError, importlib.metadata.PackageNotFoundError) as exc:
            report['errors'].append(dict(input='installed_model_code',error=str(exc)))
    if report['errors']:report['status']='BLOCKED'
    return report


def main():
    if not (os.environ.get('SLURM_JOB_ID','').isdigit() and
            re.fullmatch(r'bun[0-9]{3}',socket.gethostname().split('.')[0])):
        raise RuntimeError('Deployment checks require Slurm compute node')
    expected,worker_hash=sys.argv[1:]
    root=Path(__file__).resolve().parents[1]
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if actual!=expected or subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('Wrong/dirty deployed release')
    if sha256(root/'hpc/workstream_runtime.py')!=worker_hash:raise RuntimeError('Launcher source hash mismatch')
    for name in ('medical_protocol','medical_overlap','medical_classifier','medical_sign_analysis','medical_deployment','workstream_runtime'):
        p=root/'hpc'/(name+'.py');compile(p.read_text(),str(p),'exec')
    versions={name:importlib.metadata.version(name) for name in
              ('numpy','Pillow','scipy','torch','torchvision','timm','scikit-learn','scikit-image')}
    tasks={mode:readiness(root,mode) for mode in ('medical-overlap','medical-classifier')}
    record=dict(status='RELEASE_VERIFIED',commit=actual,worker_sha256=worker_hash,
                job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),utc=utc_now(),
                versions=versions,tasks=tasks,scope='Code/environment metadata/readability only; no medical results')
    path=root.parent/'medical-readiness.json';atomic_json(path,record)
    print(json.dumps(record,sort_keys=True))


if __name__=='__main__':main()
