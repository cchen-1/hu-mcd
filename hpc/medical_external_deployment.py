"""E224 deployment checks; compute allocation only, no GPU/model execution."""
import importlib.metadata, json, os, re, socket, subprocess, sys
from pathlib import Path
from hpc.medical_external import validate
from utils.run_tracking import atomic_json, sha256, utc_now

def main():
    if not(os.environ.get('SLURM_JOB_ID','').isdigit() and re.fullmatch('bun[0-9]{3}',socket.gethostname().split('.')[0])):raise RuntimeError('Slurm compute allocation required')
    commit,worker=sys.argv[1:];root=Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=commit or subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise ValueError('Wrong/dirty release')
    if sha256(root/'hpc/workstream_runtime.py')!=worker:raise ValueError('Worker source hash differs')
    cfg=json.loads((root/'configs/medical/medical-external.approved.json').read_text());p=validate(cfg)
    # Full hashes of compact protocol, basis, receipt and metadata; weight readability
    # here, actual complete weight hashes in the GPU worker before inference.
    inputs={}
    for name,e in p['inputs'].items():
        f=Path(e['path'])
        if name in ['sam','classifier']:
            with f.open('rb') as h:
                if not h.read(1):raise ValueError('Empty weight '+name)
        elif sha256(f)!=e['sha256']:raise ValueError('Dependency mismatch '+name)
        inputs[name]=dict(path=str(f),bytes=f.stat().st_size)
    for name,expected in p['science_sha256'].items():
        if sha256(root/name)!=expected:raise ValueError('Frozen source mismatch '+name)
    if importlib.metadata.version('timm')!='0.6.13':raise ValueError('Unpinned timm')
    for name in ['medical_external','medical_external_deployment','medical_sign_analysis','workstream_runtime','workstreams']:
        f=root/'hpc'/(name+'.py');compile(f.read_text(),str(f),'exec')
    atomic_json(root.parent/'medical-external-readiness.json',dict(status='PASS_DEPLOYMENT_READINESS_ONLY',commit=commit,worker_sha256=worker,job_id=os.environ['SLURM_JOB_ID'],utc=utc_now(),protocol_sha256=cfg['protocol_sha256'],inputs=inputs))
if __name__=='__main__':main()
