#!/usr/bin/env python3
"""Local, locked submission receipts and Slurm/SFTP collection for independent workstreams."""
import argparse
import base64
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
from datetime import datetime, timezone
from hpc.collect_results import Transport, SSH_OPTIONS
from hpc.submit_release import absolute_path


def submit(plan, state, execute=False):
    state=Path(state);state.mkdir(parents=True,exist_ok=True)
    key=plan['task_key']
    if not re.fullmatch('[A-Za-z0-9_-]+',key) or not re.fullmatch('[a-f0-9]{40}',plan['commit']):raise ValueError('Explicit task key and commit required')
    r=plan['resources'];root=absolute_path(plan['runtime_root']);release=absolute_path(plan['release'])
    for name in ['partition','qos']:
        if not re.fullmatch('[A-Za-z0-9_-]+',r[name]):raise ValueError('Unsafe resource option')
    if not re.fullmatch('[1-9][0-9]*[MG]',r['memory']) or not re.fullmatch('[0-9]{2}:[0-5][0-9]:[0-5][0-9]',r['time']) or type(r['cpus']) is not int or r['cpus']<1:raise ValueError('Invalid resources')
    if r.get('gpu') and not re.fullmatch('[A-Za-z0-9_]+:1',r['gpu']):raise ValueError('Require one explicit GPU type')
    if plan['mode'] in ('inventory','publish','baseline-readiness') and r.get('gpu'):raise ValueError('CPU preparation must not request GPU')
    jobname='humcd-reference' if plan['mode']=='discover' else 'humcd-'+key
    log='reference' if plan['mode']=='discover' else 'workstream'
    lines=['#!/usr/bin/env bash','#SBATCH --job-name='+jobname,'#SBATCH --account=a_ai_collab','#SBATCH --partition='+r['partition'],'#SBATCH --qos='+r['qos'],'#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task='+str(r['cpus']),'#SBATCH --mem='+r['memory'],'#SBATCH --time='+r['time'],'#SBATCH --output='+root+'/logs/'+log+'-%j.out','#SBATCH --error='+root+'/logs/'+log+'-%j.err']
    if plan.get('resource_afterany'):
        if not all(str(x).isdigit() for x in plan['resource_afterany']):raise ValueError('Numeric resource dependency IDs required')
        lines.append('#SBATCH --dependency=afterany:'+':'.join(str(x) for x in plan['resource_afterany']))
        lines.append('#SBATCH --kill-on-invalid-dep=yes')
    if r.get('gpu'):lines.append('#SBATCH --gres=gpu:'+r['gpu'])
    lines+=['set -euo pipefail','[[ -n "${SLURM_JOB_ID:-}" && "$(hostname -s)" =~ ^bun[0-9]{3}$ ]]','release='+shlex.quote(release),'source "$release/hpc/lib.sh"','require_compute_node','load_humcd_environment','export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg','export TORCH_HOME='+shlex.quote(root+'/models/torch'),'export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1','export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" LOKY_MAX_CPU_COUNT="$SLURM_CPUS_PER_TASK"','cd "$release"','python -m hpc.workstream_runtime '+shlex.quote(base64.b64encode(json.dumps(plan,sort_keys=True).encode()).decode())]
    script='\n'.join(lines)+'\n';subprocess.run(['bash','-n'],input=script,text=True,check=True)
    receipt=state/(key+'.submission.json')
    with (state/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if receipt.exists():raise FileExistsError('Task already has a submission receipt, including uncertain attempts: '+str(receipt))
        (state/(key+'.sbatch')).write_text(script);(state/(key+'.plan.json')).write_text(json.dumps(plan,indent=2)+'\n')
        if not execute:return {'status':'PLANNED','script_sha256':hashlib.sha256(script.encode()).hexdigest()}
        t=Transport('bunya');t.check()
        prior=t.query('squeue -u uqcche38 -h -o "%i|%j"')
        if any(line.split('|')[-1]==jobname for line in prior.splitlines()) and plan['mode']!='discover':raise ValueError('Matching task already in queue')
        record={'task_key':key,'status':'SUBMISSION_UNCERTAIN','commit':plan['commit'],'script_sha256':hashlib.sha256(script.encode()).hexdigest(),'time_utc':datetime.now(timezone.utc).isoformat()}
        receipt.write_text(json.dumps(record,indent=2)+'\n')
        result=subprocess.run(['ssh',*SSH_OPTIONS,'bunya','sbatch --parsable --chdir=/scratch/user/uqcche38'],input=script,text=True,capture_output=True,timeout=45)
        record.update(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
        if result.returncode==0 and re.fullmatch(r'[1-9][0-9]*\s*',result.stdout):record.update(status='SUBMITTED',job_id=result.stdout.strip())
        receipt.write_text(json.dumps(record,indent=2)+'\n')
        if record['status']!='SUBMITTED':raise RuntimeError('Uncertain submission: inspect queue/accounting, never automatically retry')
        return record


def collect(job, commit, destination, root='/scratch/user/uqcche38/hu-mcd'):
    if not str(job).isdigit():raise ValueError('Numeric job required')
    target=Path(destination)/str(job)/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');target.mkdir(parents=True,exist_ok=False)
    t=Transport('bunya',256*1024**2);t.check();remote=root+'/outputs/workstreams/'+str(job)
    accounting=t.query('sacct -j '+str(job)+' --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS -P');(target/'sacct.psv').write_text(accounting)
    result={'job_id':str(job),'commit':commit,'status':'PARTIAL','files':[],'errors':[]}
    if t.download(remote+'/artifacts.json',target/'artifacts.json'):
        index=json.loads((target/'artifacts.json').read_text())
        if index['job_id']!=str(job) or index['commit']!=commit:raise ValueError('Artifact identity mismatch')
        for name,entry in index['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe output path')
            path=target/name
            if not t.download(remote+'/'+name,path) or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:result['errors'].append('Missing/hash mismatch: '+name)
            else:result['files'].append(name)
        result['status']='COMPLETE' if not result['errors'] else 'INTEGRITY_ERROR';result['worker_status']=index['status']
    else:
        for name in ['launch_manifest.json','evaluation_progress.json','mcd_progress.json','mcd_manifest.json','ace_inputs_manifest.json','ace_inputs/progress.json','ace_progress.json','ace_manifest.json','inputs/input_audit_progress.json','inventory.json','anomalies.jsonl']:
            t.download(remote+'/'+name,target/name)
    launch=target/'launch_manifest.json'
    mode=json.loads(launch.read_text()).get('mode') if launch.exists() else None
    log='reference' if mode=='discover' else 'workstream'
    for ext in ['out','err']:t.download(root+'/logs/'+log+'-'+str(job)+'.'+ext,target/('job.'+ext))
    (target/'collection.json').write_text(json.dumps(result,indent=2)+'\n');return {'folder':str(target),**result}


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('submit');s.add_argument('plan',type=Path);s.add_argument('--state',required=True);s.add_argument('--execute',action='store_true')
    c=sub.add_parser('collect');c.add_argument('job');c.add_argument('--commit',required=True);c.add_argument('--destination',required=True)
    a=p.parse_args();print(json.dumps(submit(json.loads(a.plan.read_text()),a.state,a.execute) if a.cmd=='submit' else collect(a.job,a.commit,a.destination),indent=2))
if __name__=='__main__':main()
