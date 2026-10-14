"""Collect the authorized E224 batch: Slurm accounting and named SFTP only.

No remote processing, experiment submission, or automatic transfer retries.
Already verified local bytes can be reused on an explicit subsequent invocation.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from hpc.collect_results import Transport, SSH_OPTIONS, run

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'artifacts/bunya/medical-external-results-20260923'
COMMIT = 'dcc61934973aa9054a4e689ccf2334652447c575'

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    t = Transport('bunya', 256 * 1024**2)
    t.check()
    (DEST/'sacct.psv').write_text(t.query('sacct -j 28796207,28796209 --units=M -P --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,AllocCPUS,ReqMem,MaxRSS,TotalCPU,AllocTRES%100'))
    local = DEST/'collection'; local.mkdir(exist_ok=True)
    remote = '/scratch/user/uqcche38/hu-mcd/outputs/workstreams/28796209/'
    assert t.download(remote+'artifacts.json', local/'artifacts.json')
    a=json.loads((local/'artifacts.json').read_text())
    assert a['commit']==COMMIT and a['job_id']=='28796209'
    receipt=dict(job_id='28796209',commit=COMMIT,worker_status=a['status'],collected_at_utc=datetime.now(timezone.utc).isoformat(),status='PARTIAL',files=[],errors=[])
    def save(): (DEST/'collection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    try:
        for ext in ['out','err']:
            assert t.download('/scratch/user/uqcche38/hu-mcd/logs/workstream-28796209.'+ext,local/('job.'+ext))
            assert t.download('/scratch/user/uqcche38/humcd-git-prepare-28796207.'+ext,DEST/('deployment.'+ext))
        for name in ['medical-external-readiness.json','provenance.txt']:
            assert t.download('/scratch/user/uqcche38/hu-mcd-git/releases/'+COMMIT+'/'+name,DEST/name)
        names=sorted(a['files'],key=lambda n:(n.startswith(('cache/','inputs/')),n))
        pending=[]
        for name in names:
            assert not Path(name).is_absolute() and '..' not in Path(name).parts and not any(c in name for c in '\n\r"\\')
            f=local/name; f.parent.mkdir(parents=True,exist_ok=True); entry=a['files'][name]
            assert entry['bytes']<=256*1024**2
            if f.exists() and f.stat().st_size==entry['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==entry['sha256']:receipt['files'].append(name)
            else:pending.append(name)
        for start in range(0,len(pending),40):
            batch=pending[start:start+40]
            command=''.join(f'get "{remote+n}" "{local/n}"\n' for n in batch)
            p=run(['sftp',*SSH_OPTIONS,'-b','-','bunya'],command,max_file_bytes=256*1024**2)
            if p.returncode:raise RuntimeError(p.stderr)
            for n in batch:
                f=local/n;e=a['files'][n]
                assert f.stat().st_size==e['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==e['sha256'],n
                receipt['files'].append(n)
            save();print(len(receipt['files']), '/',len(names),flush=True)
        receipt['status']='COMPLETE';receipt['bytes']=sum(e['bytes'] for e in a['files'].values());save()
    except Exception as exc:
        receipt['errors'].append(repr(exc));save();raise

if __name__=='__main__':main()
