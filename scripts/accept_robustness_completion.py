"""Accept fixed-slot completion from cached arrays; no inference or remote work."""
import sys, json, csv, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import accept_next_phase_batch as a
from hpc.mask_sensitivity import ObliqueScores, identity_check
R=a.ROOT; O=R/'artifacts/bunya/robustness-completion-20260924'; OLD=a.OUT
J=a.J; batch=J(O/'batch.json'); commit=batch['commit']; a.OUT=O
# Reuse the established array-level acceptance implementation for ONLY the four
# new collections. Its original six-class outputs remain immutable.
def reader(p):
    if Path(p)==R/'artifacts/bunya/next-phase-20260921/submitted-batch.json':return batch
    return J(p)
a.J=reader
new=a.robustness()
extra=[]; warnings=[]
a.ck('commit='+commit in (O/'deployment-provenance.txt').read_text(),'Deployment exact commit')
for name in ['deployment.out','deployment.err']:
    warnings.append(dict(job=batch['deployment']['job_id'],file=str((O/name).relative_to(R)),text=(O/name).read_text()))
account=list(csv.DictReader((O/'final-sacct.psv').open(),delimiter='|'))
for j in [batch['deployment']]+batch['jobs']:
    row=next(r for r in account if r['JobID']==j['job_id'])
    a.ck(row['State']=='COMPLETED' and row['ExitCode']=='0:0','Final Slurm success '+j['job_id'])
for j in batch['jobs']:
    p,idx=a.collection(j['job_id']);cfg=J(p/'actual_config.json'); cp=cfg['completion']
    a.ck(idx['status']=='PASS' and idx['commit']==commit,'Worker PASS and frozen commit')
    old=list((OLD/'collections'/cp['previous_job']).glob('*/artifacts.json'))
    a.ck(len(old)==1 and a.sha(old[0])==cp['artifacts_sha256'],'Original failed manifest binding')
    q=old[0].parent; oldcfg=J(q/'actual_config.json')
    for key in ['class_name','payload','payload_sha256']:a.ck(cfg[key]==oldcfg[key],'Unchanged '+key)
    for name in ['identity_gate.json','identity0_science.npz','identity0_masks.npz','erosion1_science.npz','erosion1_masks.npz','fixed_discovery.npz','model_identity.json','region_ledger.json','input_manifest.json','frozen_protocol.json','baseline_strata.json']:
        a.ck(a.sha(p/name)==a.sha(q/name),'Byte-identical preserved artifact '+j['class_name']+' '+name)
    reuse=J(p/'reuse_provenance.json'); a.ck(reuse['previous_job']==cp['previous_job'] and reuse['previous_manifest_sha256']==cp['artifacts_sha256'],'Reuse provenance')
    base=a.loadz(p/'identity0_science.npz');bs=a.loadz(p/'fixed_discovery.npz');scorer=ObliqueScores([bs[k] for k in sorted(bs) if k.startswith('concept_basis_')]+[bs['complement_basis']])
    entry=J(p/'fixed_slot_entry_gate.json');a.ck(entry['status']=='PASS' and entry['batch']==cp['failed_batch'],'Known failed batch entry gate')
    controls=[p/'fixed_slot_entry_control.npz']+sorted(p.glob('*_fixed_unchanged.npz')); controlled=set(); stats=[]
    for f in controls:
        z=a.loadz(f);rr=z['rows']; a.ck(np.array_equal(rr,np.arange(int(rr[0]),min(int(rr[0])+8,len(base['valid'])))) and int(rr[0])%8==0,'Full original batch slots')
        gate=identity_check(z['features'],base['features'][rr],scorer,base['scores'][rr],base['assignments'][rr]); np.testing.assert_allclose(z['logits'],base['logits'][rr],rtol=1e-4,atol=1e-4)
        controlled.add(int(rr[0])//8);stats.append(dict(file=f.name,**gate))
    slots=J(p/'fixed_slot_batches.json');expected=[]
    for c in ['erosion2','dilation1','dilation2']:
        z=a.loadz(p/(c+'_science.npz'));empty=z['empty']; valid=z['valid']
        a.ck(np.array_equal(z['zero'],~empty & np.all(z['features']==0,axis=1)),c+' zero flag replay')
        a.ck(np.isnan(z['logits'][empty]).all() and np.isfinite(z['logits'][~empty]).all() and np.isnan(z['scores'][~valid]).all(),c+' filler outputs discarded/invalid scores missing')
        a.ck(np.array_equal(z['baseline_valid'],base['valid']) and np.array_equal(z['baseline_assignments'],base['assignments']),c+' fixed baseline denominator and assignments')
        for i in range(0,len(empty),8):
            rr=list(range(i,min(i+8,len(empty))));fill=[r for r in rr if empty[r]]
            if fill:
                expected.append((c,i//8));s=next(s for s in slots if s['condition']==c and s['batch']==i//8)
                a.ck(s['original_rows']==rr and s['filler_rows']==fill and s['active_rows']==[r for r in rr if not empty[r]] and i//8 in controlled,c+' exact active/filler partition with unchanged control')
    a.ck(sorted(expected)==sorted((s['condition'],s['batch']) for s in slots),'All and only empty batches recorded')
    a.ck(len(J(p/'condition_counts.json'))==5 and J(p/'sensitivity_progress.json')['stage']=='COMPLETE_PENDING_ACCEPTANCE','Five conditions complete')
    extra.append(dict(class_name=j['class_name'],job=j['job_id'],previous_job=cp['previous_job'],status='ACCEPTED',entry_gate=entry,controls=stats,fixed_slot_batches=slots,collection=str(p.relative_to(R))))
    for f in [p/'anomalies.jsonl',p/'job.err',p/'job.out']:
        if f.exists():
            text=f.read_text()
            if f.name=='anomalies.jsonl' or any(w in text.lower() for w in ['warning','error','exception']):warnings.append(dict(job=j['job_id'],file=str(f.relative_to(R)),text=text))
# Preserve source-specific four-class acceptance before producing merged tables.
a.dump(O/'four-class-acceptance.json',new)
for filename in ['robustness-conditions.csv','robustness-per-image.csv','robustness-transitions.csv','robustness-strata.csv']:
    old=list(csv.DictReader((OLD/filename).open())); current=list(csv.DictReader((O/filename).open()))
    (O/('new-'+filename)).write_text((O/filename).read_text())
    added=[r for r in current if r['condition'] in ['erosion2','dilation1','dilation2']]
    merged=old+added
    if filename=='robustness-conditions.csv':a.ck(len(merged)==50 and len({(r['class_name'],r['condition']) for r in merged})==50,'Unique complete ten-class five-condition grid')
    a.table(O/filename,merged)
rows=list(csv.DictReader((O/'robustness-conditions.csv').open()));macro=[]
for c in ['identity0','erosion1','erosion2','dilation1','dilation2']:
    rr=[r for r in rows if r['condition']==c];a.ck(len(rr)==10,'Ten accepted classes '+c)
    macro.append(dict(condition=c,contributing_classes=10,equal_class_mean=float(np.mean([float(r['image_mean_retention']) for r in rr])),raw_regions=sum(int(r['raw_regions']) for r in rr),baseline_valid=sum(int(r['baseline_valid']) for r in rr),empty=sum(int(r['empty']) for r in rr),zero=sum(int(r['zero']) for r in rr),valid=sum(int(r['valid']) for r in rr),retained=sum(int(r['retained']) for r in rr)))
a.table(O/'robustness-macro.csv',macro)
result=dict(status='ACCEPTED',generated_at_utc=datetime.now(timezone.utc).isoformat(),commit=commit,checks_passed=len(a.checks),checks=a.checks,completed_classes=extra,macro=macro,region_condition_records=sum(int(r['raw_regions']) for r in rows),old_accepted_rows_preserved=True,missing_conditions=0,precision_and_tolerances_unchanged=True,new_inference_during_acceptance=False,script_sha256=a.sha(__file__),limitations=['Assignment stability, not segmentation accuracy or semantic quality','Original reduced-batch arithmetic mechanism remains unproven','Fixed-slot correction applies only where an intervention produces empty masks; fillers are discarded'])
a.dump(O/'warnings-and-events.json',warnings);a.dump(O/'acceptance.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['checks','completed_classes']},indent=2))
