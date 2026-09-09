"""Commit-bound Slurm worker for coordinated discovery, evaluation and data preparation."""
import base64
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import traceback
import warnings

from utils.run_tracking import atomic_json, sha256, utc_now

MODES={'inventory':'hpc.workstream_inventory','publish':'hpc.multiclass_inputs','evaluate':'hpc.evaluate_reference'}


def event(output, stage, kind, evidence, impact, handling, status='OPEN', severity='warning'):
    record={'time_utc':utc_now(),'job_id':os.environ.get('SLURM_JOB_ID'),'stage':stage,'kind':kind,'evidence':evidence,
            'possible_impact':impact,'handling':handling,'status':status,'severity':severity}
    with (Path(output)/'anomalies.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    return record


def verify_inputs(config):
    path=Path(config['dataset_manifest'])
    if sha256(path)!=config['dataset_manifest_sha256']:raise ValueError('Input manifest hash mismatch')
    data=json.loads(path.read_text());mapping=json.loads(Path('imagenet1k_class_info.json').read_text());name=config['class_name']
    if data['synset']!=mapping[name]['wnid'] or data['class_name']!=name or data['seed']!=43 or data['source_dir']!=config['source_dir']:
        raise ValueError('Input class, seed or root mismatch')
    if not data.get('ready_for_reference') or data.get('validation_issues'):raise ValueError('Unresolved input anomaly')
    seen=set();paths=set()
    for split,rel,n in [('training',name,400),('validation','val_imgs/'+name+'_val',50)]:
        entries=data[split];folder=Path(data['source_dir'])/rel
        if len(entries)!=n or sorted(p.name for p in folder.iterdir())!=sorted(e['prepared_name'] for e in entries):
            raise ValueError('Input count/directory contents mismatch')
        for entry in entries:
            p=folder/entry['prepared_name']
            if p.name!=entry['prepared_name'] or str(p)!=entry['input_path'] or sha256(p)!=entry['input_sha256'] or sha256(entry['source'])!=entry['sha256']:
                raise ValueError('Actual/original input identity mismatch')
            if entry['input_sha256'] in seen or str(p.resolve()) in paths:raise ValueError('Duplicate input within/across splits')
            seen.add(entry['input_sha256']);paths.add(str(p.resolve()))
    return {'images':450,'dataset_manifest_sha256':sha256(path),'duplicates':0}


def verify_discovery_reuse(config, release):
    """Reuse the completed Golden run's checks; independently verify new class identities."""
    for key in ('reference_summary','reference_launch','reference_config'):
        if sha256(config[key])!=config[key+'_sha256']:raise ValueError('Changed reusable reference evidence: '+key)
    summary=json.loads(Path(config['reference_summary']).read_text());launch=json.loads(Path(config['reference_launch']).read_text());old=json.loads(Path(config['reference_config']).read_text())
    if summary['status']!='PASS' or str(summary['slurm_job_id'])!='28208840' or launch['status']!='EXECUTION_COMPLETED':
        raise ValueError('Completed reference evidence required')
    if summary['git_commit']!=launch['actual_commit'] or summary['resolved_config_sha256']!=sha256(config['reference_config']):
        raise ValueError('Reference evidence cross-identity mismatch')
    from hpc.prerequisites import verify_completed
    prior=verify_completed(launch['plan']['prerequisite_report'],launch['plan']['prerequisite_report_sha256'],release,old)
    keys=('model_name','layer_name','max_shortest_side','seed','train_images','validation_images','segmentation','clustering','precision','sam_checkpoint_sha256','resnet_checkpoint_sha256','batch_size','save_scientific_records')
    if any(config[k]!=old[k] for k in keys):raise ValueError('Scientific settings differ from completed reference')
    return {'source_job':'28208840','scope':'shared code/weights/precision/batch and algebra checks; new inputs checked independently','original_prerequisites':prior}


def main():
    plan=json.loads(base64.b64decode(sys.argv[1],validate=True));job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdigit() or not re.fullmatch('bun[0-9]{3}',socket.gethostname().split('.')[0]):raise RuntimeError('Slurm compute allocation required')
    release=Path(plan['release']);actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=release,text=True).strip()
    if actual!=plan['commit'] or subprocess.check_output(['git','status','--porcelain'],cwd=release,text=True).strip():raise RuntimeError('Wrong/dirty immutable release')
    if sha256(release/'hpc/workstream_runtime.py')!=plan['worker_sha256']:raise RuntimeError('Worker source mismatch')
    output=Path(plan['runtime_root'])/'outputs/workstreams'/job;output.mkdir(parents=True,exist_ok=False)
    config=plan['config'];record={'status':'RUNNING','job_id':job,'commit':actual,'host':socket.gethostname(),'mode':plan['mode'],'task_key':plan['task_key'],'plan':plan,'started_at_utc':utc_now(),'stage':'initialization'}
    atomic_json(output/'launch_manifest.json',record);atomic_json(output/'actual_config.json',config)
    (output/'anomalies.jsonl').touch()
    def warning(message, category, filename, lineno, file=None, line=None):
        event(output,record['stage'],'python_warning',{'message':str(message),'category':category.__name__,'filename':filename,'line':lineno},'UNASSESSED','Recorded; task continues unless an explicit correctness assertion fails')
        print(f'WARNING {category.__name__}: {message}',file=sys.stderr,flush=True)
    warnings.showwarning=warning;warnings.simplefilter('always')
    try:
        record['versions']={x:importlib.metadata.version(x) for x in ['numpy','scipy','torch','torchvision','timm','scikit-learn','scikit-image']}
        if plan['mode'] in ('discover','evaluate'):
            import torch
            if record['versions']['timm']!='0.6.13':raise ValueError('Unpinned timm version')
            installed=Path(importlib.import_module('timm').__file__).parent/'models'
            for name in ['resnet.py','sal_layers.py']:
                if sha256(installed/name)!=sha256(release/'input_masking'/name):raise ValueError('Installed mask code differs from release')
            record['inputs']=verify_inputs(config)
            for p,key in [(config['resnet_checkpoint'],'resnet_checkpoint_sha256'),(config['segmentation']['checkpoint'],'sam_checkpoint_sha256')]:
                if sha256(p)!=config[key]:raise ValueError('Weight identity mismatch: '+key)
            if Path(config['resnet_checkpoint']).resolve()!=Path(os.environ['TORCH_HOME'],'hub/checkpoints/resnet50_a1_0-14fe96d1.pth').resolve():raise ValueError('Classifier cache path mismatch')
            torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
            observed={'cudnn_allow_tf32':bool(torch.backends.cudnn.allow_tf32),'matmul_allow_tf32':bool(torch.backends.cuda.matmul.allow_tf32),'cudnn_benchmark':bool(torch.backends.cudnn.benchmark),'cudnn_deterministic':bool(torch.backends.cudnn.deterministic),'deterministic_algorithms':bool(torch.are_deterministic_algorithms_enabled()),'float32_matmul_precision':torch.get_float32_matmul_precision()}
            if observed!=config['precision']:raise ValueError('Precision differs from fixed reference')
            record['precision']=observed
        record['stage']=plan['mode'];atomic_json(output/'launch_manifest.json',record)
        if plan['mode']=='discover':
            record['reused_checks']=verify_discovery_reuse(config,release);atomic_json(output/'launch_manifest.json',record)
            launchdir=Path(plan['runtime_root'])/'launches'/job;launchdir.mkdir(exist_ok=False)
            public_launch={'schema_version':1,'job_id':job,'run_id':job,'slurm_job_id':job,'actual_commit':actual,'mode':'reference','host':socket.gethostname(),'status':'RUNNING','plan':plan}
            atomic_json(launchdir/'launch_manifest.json',public_launch)
            # Stream and preserve every child warning/diagnostic; do not allocate a second GPU.
            with (output/'discovery-process.log').open('w') as childlog:
                proc=subprocess.Popen([sys.executable,str(release/'run_smoke.py'),'--config',str(output/'actual_config.json'),'--run-id',job],cwd=release,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                for line in proc.stdout:
                    print(line,end='',flush=True);childlog.write(line);childlog.flush()
                    if 'warning' in line.lower():event(output,'discover','child_warning',line.rstrip(),'UNASSESSED','Preserved with complete child log; inspect in context')
                code=proc.wait()
            public_launch['status']='EXECUTION_COMPLETED' if code==0 else 'FAILED'
            atomic_json(launchdir/'launch_manifest.json',public_launch)
            if code:raise RuntimeError('Discovery process failed: '+str(code))
            result=json.loads((Path(plan['runtime_root'])/'outputs/runs'/job/'summary.json').read_text())
            record['discovery_output']=str(Path(plan['runtime_root'])/'outputs/runs'/job)
            # Class-format compatible launch record for the established reference collector.
            atomic_json(launchdir/'launch_manifest.json',{'schema_version':1,'job_id':job,'run_id':job,'slurm_job_id':job,'actual_commit':actual,'mode':'reference','host':socket.gethostname(),'status':'EXECUTION_COMPLETED','plan':plan})
        else:
            module=importlib.import_module(MODES[plan['mode']]);function=module.publish if plan['mode']=='publish' else module.run
            result=function(config,output)
        atomic_json(output/'result.json',result);record.update(status='PASS',finished_at_utc=utc_now())
    except BaseException as exc:
        record.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),finished_at_utc=utc_now())
        event(output,record['stage'],'task_failure',record['traceback'],'Affected task may lack valid/complete results','Pause affected task; retain files; no automatic resubmission',severity='error')
        raise
    finally:
        atomic_json(output/'launch_manifest.json',record)
        files={str(p.relative_to(output)):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(output.rglob('*')) if p.is_file() and p.name!='artifacts.json'}
        atomic_json(output/'artifacts.json',{'job_id':job,'commit':actual,'task_key':plan['task_key'],'status':record['status'],'files':files})

if __name__=='__main__':main()
