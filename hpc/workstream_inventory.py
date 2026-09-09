"""Read bounded project inventories and audit new class inputs inside Slurm only."""
import json
from pathlib import Path
from utils.run_tracking import sha256, atomic_json
from hpc.multiclass_inputs import audit


def run(config, output):
    root=Path(config['runtime_root']); report={'status':'RUNNING','runs':[],'launches':[],'data':[],'cache':[],'errors':[]}
    for kind,base,pattern in [('runs',root/'outputs/runs','*/summary.json'),('launches',root/'launches','*/launch_manifest.json')]:
        for path in sorted(base.glob(pattern)):
            try:
                obj=json.loads(path.read_text());report[kind].append({'path':str(path),'sha256':sha256(path),'status':obj.get('status'),'commit':obj.get('git_commit',obj.get('actual_commit')),'class_name':obj.get('class_name',obj.get('plan',{}).get('config',{}).get('class_name')),'job_id':obj.get('slurm_job_id',obj.get('job_id')),'mode':obj.get('mode')})
            except Exception as exc:report['errors'].append({'path':str(path),'error':str(exc)})
    for kind,base in [('data',root/'data'),('cache',root/'cache')]:
        if base.exists():
            report[kind]=[{'name':p.name,'directory':p.is_dir(),'manifest':str(p/'dataset_manifest.json') if (p/'dataset_manifest.json').is_file() else None} for p in sorted(base.iterdir())]
    for name in ['random','random_acts','acts','acts_mcd','segms_val','self_repr_matrices_mcd','self_repr_matrices_humcd']:
        for base in (root, root/'data',root/'cache'):
            p=base/name
            if p.exists():report.setdefault('baseline_candidates',[]).append({'path':str(p),'entries':[x.name for x in sorted(p.iterdir())][:30] if p.is_dir() else []})
    atomic_json(output/'inventory.json',report)
    report['input_audit']=audit(config,output/'inputs')
    report['status']='PASS' if not report['errors'] else 'COMPLETED_WITH_ISSUES'
    atomic_json(output/'inventory.json',report)
    return report
