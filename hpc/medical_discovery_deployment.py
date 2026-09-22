"""Readiness only, inside the approved1CPU/1GiB/5min deployment allocation."""
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from hpc.medical_discovery import validate
from utils.run_tracking import atomic_json,sha256,utc_now


def main():
    if not (os.environ.get('SLURM_JOB_ID','').isdigit() and re.fullmatch('bun[0-9]{3}',socket.gethostname().split('.')[0])):
        raise RuntimeError('Slurm compute allocation required')
    commit,worker=sys.argv[1:];root=Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=commit or subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise ValueError('Wrong/dirty release')
    if sha256(root/'hpc/workstream_runtime.py')!=worker:raise ValueError('Worker identity mismatch')
    for name in ['medical_discovery','medical_discovery_deployment','workstream_runtime','workstreams']:
        f=root/'hpc'/(name+'.py');compile(f.read_text(),str(f),'exec')
    records={}
    for mode in ['medical-discovery-inputs','medical-discovery']:
        path=root/'configs/medical'/(mode+'.approved.json');c=json.loads(path.read_text());p=validate(c)
        inputs={}
        for name,e in p['inputs'].items():
            f=Path(e['path'])
            with f.open('rb') as h:
                if not h.read(1):raise ValueError('Empty input '+name)
            inputs[name]=dict(path=str(f),bytes=f.stat().st_size,readable=True)
        for name,expected in p['released_science_files'].items():
            if sha256(root/name)!=expected:raise ValueError('Released HU scientific source changed: '+name)
        records[mode]=dict(config_sha256=sha256(path),protocol_sha256=c['protocol_sha256'],inputs=inputs)
    if importlib.metadata.version('timm')!='0.6.13':raise ValueError('Unpinned timm')
    dist=importlib.metadata.distribution('timm')
    for name in ['resnet.py','sal_layers.py']:
        if sha256(Path(dist.locate_file('timm/models/'+name)))!=sha256(root/'input_masking'/name):raise ValueError('Installed masking source changed')
    record=dict(status='PASS_DEPLOYMENT_READINESS_ONLY',commit=commit,worker_sha256=worker,job_id=os.environ['SLURM_JOB_ID'],utc=utc_now(),tasks=records,not_gpu_adapter_acceptance=True)
    atomic_json(root.parent/'medical-discovery-readiness.json',record);print(json.dumps(record))
if __name__=='__main__':main()
