#!/usr/bin/env python3
"""Plan/submit exactly once; no wait, result collection, retry or remote processing.

Two independent method lanes, at most two GPUs. afterany serializes resources;
afterok plus compute-node manifest gates enforce scientific dependencies.
"""
import argparse,copy,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hpc.workstreams import submit
from hpc.mcd_reference import PROTOCOL,digest
from hpc.ace_reference import protocol_for
from hpc.evaluate_reference import random_signature

R=Path(__file__).resolve().parents[1]
W=R/'artifacts/bunya/evaluation-baselines-20260910'
O=R/'artifacts/bunya/nine-class-baselines-20260923'
REMOTE='/scratch/user/uqcche38/hu-mcd'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
write=lambda p,d:p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
CAPS={
 ('MCD','features'):dict(cpus=4,memory='16G',time='00:30:00',partition='gpu_cuda',qos='short',gpu='l40s:1'),
 ('MCD','fit'):dict(cpus=8,memory='64G',time='04:00:00',partition='general',qos='normal'),
 ('MCD','evaluation'):dict(cpus=4,memory='16G',time='00:30:00',partition='gpu_cuda',qos='short',gpu='l40s:1'),
 ('ACE','inputs'):dict(cpus=4,memory='8G',time='00:15:00',partition='general',qos='normal'),
 ('ACE','features'):dict(cpus=4,memory='16G',time='01:00:00',partition='gpu_cuda',qos='short',gpu='l40s:1'),
 ('ACE','cav'):dict(cpus=4,memory='16G',time='01:00:00',partition='general',qos='normal'),
 ('ACE','evaluation'):dict(cpus=4,memory='16G',time='01:00:00',partition='gpu_cuda',qos='short',gpu='l40s:1')}

def configurations(commit):
    golden=load(W/'MCD-golden-features.plan.json')['config']
    bindings=load(W/'analysis/analysis.json')['bindings'];configs={};audit=[]
    for binding in bindings:
        name=binding['class_name']
        if name=='golden_retriever':continue
        col=load(Path(binding['collection']));acc=load(Path(binding['acceptance']));folder=Path(col['folder'])
        assert col['status']=='COMPLETE' and col['worker_status']=='PASS' and acc['status']=='PASS'
        assert str(acc['job_id'])==str(col['job_id'])==str(binding['job_id']) and acc['class_name']==name
        original=load(folder/'actual_config.json');src=load(folder/'evaluation_sources.json')
        proof=load(folder/'artifacts.json');signature=load(folder/'random_signature.json')
        evidence=R/'artifacts/bunya/ten-class-review/evidence'/original['source_job']
        dataset=load(evidence/'inputs/dataset_manifest.json')
        assert sha(evidence/'inputs/dataset_manifest.json')==original['dataset_manifest_sha256']
        cfg={k:copy.deepcopy(v) for k,v in original.items() if k.startswith('source_') or k in (
            'class_name','model_name','dataset_manifest','dataset_manifest_sha256','resnet_checkpoint',
            'resnet_checkpoint_sha256','precision','batch_size','max_shortest_side','train_images','validation_images','seed','random_seeds')}
        cfg.update(multiclass_reference=True,execution_commit=commit,runtime_root=REMOTE,
            golden_run_dir=original['source_run_dir'],golden_files_sha256={n:original['source_files_sha256'][n] for n in ('summary.json','run_manifest.json','resolved_config.json','scientific/discovery.json')},
            golden_launch_manifest=golden['golden_launch_manifest'],golden_launch_sha256=golden['golden_launch_sha256'],
            licensed_root='/scratch/licenseddata/imagenet/imagenet-1k',source_alias_explanation='golden_run_dir/files are legacy key names: bind THIS class; golden_launch pins shared original environment only')
        expected=random_signature(cfg,signature['identity']['model_default_cfg'],dataset['validation'],signature['identity']['implementation'])
        assert expected==signature and src['inputs']==dataset['validation']
        files={k:v for k,v in proof['files'].items() if k in ['evaluation_sources.json','evaluation_config.json','evaluation_curves.json'] or k.startswith('evaluation_data/rdm_')}
        for k,item in files.items():assert sha(folder/k)==item['sha256']
        contract=dict(status='PASS',source_job=str(col['job_id']),source_commit=col['commit'],signature=signature,files=files,
            local_acceptance_sha256=sha(Path(binding['acceptance'])),class_name=name)
        write(O/(name+'-random-contract.json'),contract)
        cfg['random_reuse']=dict(directory=REMOTE+'/outputs/workstreams/'+str(col['job_id']),source_job=str(col['job_id']),contract=contract,contract_sha256=digest(contract))
        for method in ['MCD','ACE']:
            c=copy.deepcopy(cfg);c['baseline_method']=method;c['layer_name']='layer4' if method=='MCD' else 'global_pool'
            if method=='MCD':c['protocol']=PROTOCOL
            else:c['ace_protocol_sha256']=digest(protocol_for(name))
            configs[(name,method)]=c
        audit.append(dict(class_name=name,discovery_job=original['source_job'],random_job=str(col['job_id']),dataset_sha256=cfg['dataset_manifest_sha256'],weights_sha256=cfg['resnet_checkpoint_sha256'],random_signature=signature['sha256'],target_index=load(R/'imagenet1k_class_info.json')[name]['class_index']))
    assert len(audit)==9
    write(O/'input-reuse-audit.json',dict(status='PASS',scope='Local accepted evidence and contracts; actual remote files rechecked inside Slurm',classes=audit))
    return configs

def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--deployment-job');p.add_argument('--execute',action='store_true');a=p.parse_args()
    O.mkdir(exist_ok=True);configs=configurations(a.commit)
    if a.execute:
        assert a.deployment_job and a.deployment_job.isdigit()
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==a.commit
        # Do not silently repeat a partial batch. User/coordinator must inspect receipts.
        assert not list((O/'submissions').glob('*.submission.json')),'Existing attempts: refuse automatic resubmission'
    entries=[];lane={};counter=90000000
    for name in sorted({k[0] for k in configs}):
        for method in ['MCD','ACE']:
            stages=['features','fit','evaluation'] if method=='MCD' else ['inputs','features','cav','evaluation']
            prior={};failed=False
            for stage in stages:
                key=f'BASE26-{method}-{name}-{stage}'
                cfg=copy.deepcopy(configs[(name,method)]);caps=CAPS[(method,stage)]
                cfg.update(stage=stage,device='cuda:0' if 'gpu' in caps else 'cpu',torch_num_threads=caps['cpus'],stage_dependencies=[])
                needed=[]
                if method=='ACE' and stage!='inputs':needed.append(('inputs','ace_input','ace_inputs_manifest.json'))
                if stage in ('fit','cav','evaluation'):needed.append(('features','feature',method.lower()+'_manifest.json'))
                if stage=='evaluation':needed.append(('fit' if method=='MCD' else 'cav','fit' if method=='MCD' else 'cav',method.lower()+'_manifest.json'))
                for step,prefix,manifest in needed:
                    cfg['stage_dependencies'].append(dict(job_id=prior[step]['job_id'],task_key=prior[step]['task_key'],stage=step,prefix=prefix,manifest=manifest))
                plan=dict(task_key=key,mode='baseline-stage',commit=a.commit,release='/scratch/user/uqcche38/hu-mcd-git/releases/'+a.commit+'/code',runtime_root=REMOTE,worker_sha256=sha(R/'hpc/workstream_runtime.py'),resources=caps,config=cfg)
                if a.deployment_job:plan['deployment_afterok']=a.deployment_job
                if needed:plan['prerequisite_afterok']=[prior[step]['job_id'] for step,_,_ in needed]
                # Each method has its own afterany resource lane. An unrelated
                # class failure cannot block the other class scientifically.
                if stage==stages[0] and method in lane:plan['resource_afterany']=[lane[method]]
                write(O/(key+'.plan.json'),plan)
                try:
                    receipt=submit(plan,O/('submissions' if a.execute else 'dry-run'),execute=a.execute)
                except Exception as exc:
                    failed=True;entries.append(dict(task_key=key,class_name=name,method=method,stage=stage,status='SUBMISSION_BLOCKED',error=str(exc)))
                    write(O/'batch.json',dict(status='PARTIAL_SUBMISSION',commit=a.commit,entries=entries,no_wait=True))
                    # Stop this class chain only; never retry this attempt.
                    break
                if not a.execute:counter+=1;receipt['job_id']=str(counter)
                prior[stage]=dict(job_id=receipt['job_id'],task_key=key)
                entries.append(dict(**receipt,class_name=name,method=method,stage=stage,resources=caps,plan=str(O/(key+'.plan.json'))))
                write(O/('batch.json' if a.execute else 'batch-plan.json'),dict(status='SUBMITTING' if a.execute else 'PLANNED',commit=a.commit,entries=entries,no_wait=True))
            if prior:lane[method]=list(prior.values())[-1]['job_id']
    result=dict(status='SUBMITTED' if a.execute and all(e['status']=='SUBMITTED' for e in entries) else 'PLANNED' if not a.execute else 'PARTIAL_SUBMISSION',commit=a.commit,entries=entries,no_wait=True,collection_policy='Wait for user notice; no automatic collection or retry',gpu_concurrency_upper_bound=2,gpu_allocation_hours_upper_bound=27)
    write(O/('batch.json' if a.execute else 'batch-plan.json'),result)
    print(json.dumps(dict(status=result['status'],tasks=len(entries),commit=a.commit)))
if __name__=='__main__':main()
