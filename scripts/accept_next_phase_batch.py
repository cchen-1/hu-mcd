"""Local acceptance of immutable ROB26 and medical D1 artifacts; no inference/refit.

Recomputes metrics from saved arrays and verifies mask/feature/member identities.
Run using humcd-upstream Python. All remote outputs must already be collected.
"""
import csv, hashlib, json, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.linalg import norm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hpc.mask_sensitivity import ObliqueScores
from hpc.final_mask_intervention import intervene

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/bunya/batch-review-20260922'
J=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(p,rows):
    if not rows:return
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
checks=[]
def ck(ok,label):
    if not ok:raise ValueError(label)
    checks.append(label)
def loadz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def collection(job):
    pp=list((OUT/'collections'/str(job)).glob('*/collection.json'))
    ck(len(pp)==1,'Unambiguous collection '+str(job));p=pp[0].parent
    receipt=J(p/'collection.json');index=J(p/'artifacts.json')
    ck(receipt['status']=='COMPLETE' and not receipt['errors'],str(job)+' complete transfer')
    for n,e in index['files'].items():
        ck((p/n).stat().st_size==e['bytes'] and sha(p/n)==e['sha256'],str(job)+' artifact '+n)
    launch=J(p/'launch_manifest.json');config=J(p/'actual_config.json')
    ck(index['commit']==launch['commit']==launch['plan']['commit'],str(job)+' commit binding')
    if 'execution_commit' in config:ck(config['execution_commit']==index['commit'],str(job)+' explicit config commit')
    ck(config==launch['plan']['config'],str(job)+' actual config matches plan')
    return p,index

def robustness():
    jobs=[j for j in J(ROOT/'artifacts/bunya/next-phase-20260921/submitted-batch.json')['jobs'] if j.get('task_key','').startswith('ROB26-')]
    rows=[];perimage=[];controls=[];statuses=[];transitions=[];strata=[]
    for j in jobs:
        p,idx=collection(j['job_id']);ledger=J(p/'region_ledger.json');n=len(ledger);name=ledger[0]['class_name']
        ck(idx['commit']==j['commit'],'ROB commit '+name)
        identity=loadz(p/'identity0_science.npz');bs=loadz(p/'fixed_discovery.npz')
        bases=[bs[k] for k in sorted(bs) if k.startswith('concept_basis_')]+[bs['complement_basis']]
        scorer=ObliqueScores(bases);oc=len(bases)-1
        masks0=loadz(p/'identity0_masks.npz');m0=np.unpackbits(masks0['packed'],axis=2,count=int(masks0['shape'][2])).astype(bool)
        ck(J(p/'identity_gate.json')['status']=='PASS',name+' original-batch identity gate')
        ck([r['global_row'] if 'global_row' in r else r['raw_feature_row'] for r in ledger]==list(range(n)),name+' region row order')
        for i,r in enumerate(ledger):
            ck(hashlib.sha256(m0[i].tobytes()).hexdigest()==r['effective_mask_sha256'],name+' mask identity '+str(i))
        image_ids=[Path(r['source']).name for r in J(p/'input_manifest.json')['validation']]
        # The original image manifest uses prepared_name in some versions.
        ids=list(dict.fromkeys(r['image_id'] for r in ledger))
        ck(len(image_ids)==50 and set(ids).issubset(image_ids),name+' fixed50 image identity')
        group=np.array([r['image_id'] for r in ledger]);base=identity['valid'];assign0=identity['assignments']
        completed=J(p/'condition_counts.json');names={r['condition'] for r in completed}
        statuses.append(dict(class_name=name,job=j['job_id'],state=idx['status'],stage=J(p/'sensitivity_progress.json')['stage'],completed_conditions=','.join(r['condition'] for r in completed),acceptance='ACCEPTED' if idx['status']=='PASS' else 'FAILED_PARTIAL_ONLY',commit=idx['commit']))
        for count in completed:
            condition=count['condition'];z=loadz(p/(condition+'_science.npz'));mz=loadz(p/(condition+'_masks.npz'))
            m=np.unpackbits(mz['packed'],axis=2,count=int(mz['shape'][2])).astype(bool)
            operation=condition[:-1];radius=int(condition[-1])
            ck(np.array_equal(m,np.stack([intervene(a,operation,radius) for a in m0])),name+condition+' geometry replay')
            ck(np.array_equal(z['area'],m.sum((1,2))) and np.array_equal(z['empty'],~m.any((1,2))),name+condition+' geometry counts')
            valid=z['valid'];ck(np.array_equal(valid,~(z['empty']|z['zero'])),name+condition+' validity')
            ck(np.isfinite(z['features'][~z['empty']]).all() and np.isnan(z['features'][z['empty']]).all(),name+condition+' explicit missing features')
            sc,err=scorer(z['features'][valid]);np.testing.assert_allclose(sc,z['scores'][valid],rtol=1e-5,atol=1e-5)
            ck(np.array_equal(sc.argmax(1),z['assignments'][valid]) and np.all(z['assignments'][~valid]==-1),name+condition+' assignment replay')
            retained=base&valid&(assign0==z['assignments']);both=base&valid
            ck(int(retained.sum())==count['retained'] and int(valid.sum())==count['valid'],name+condition+' saved counts')
            rates=[];learnrates=[];ocrates=[]
            for im in image_ids:
                g=group==im;den=int((g&base).sum());num=int((g&retained).sum())
                lr=g&base&(assign0!=oc);cr=g&base&(assign0==oc)
                value=num/den if den else None
                if den:rates.append(value)
                if lr.any():learnrates.append(float(retained[lr].mean()))
                if cr.any():ocrates.append(float(retained[cr].mean()))
                perimage.append(dict(class_name=name,condition=condition,image_id=im,baseline_valid=den,retained=num,retention=value,empty=int((g&z['empty']).sum()),zero=int((g&z['zero']).sum()),condition_valid=int((g&valid).sum())))
            rng=np.random.RandomState(20260921);boot=np.mean(np.array(rates)[rng.randint(0,len(rates),(2000,len(rates)))],axis=1)
            rows.append(dict(class_name=name,job=j['job_id'],condition=condition,scope='COMPLETE_CLASS' if idx['status']=='PASS' else 'PARTIAL_CLASS_COMPLETED_CONDITION',images=50,evaluable_images=len(rates),raw_regions=n,baseline_valid=int(base.sum()),valid=int(valid.sum()),empty=int(z['empty'].sum()),zero=int(z['zero'].sum()),retained=int(retained.sum()),image_mean_retention=float(np.mean(rates)),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),region_weighted_retention=float(retained.sum()/base.sum()),conditional_agreement=float(retained.sum()/both.sum()),conditional_denominator=int(both.sum()),learned_image_mean=float(np.mean(learnrates)) if learnrates else None,complement_image_mean=float(np.mean(ocrates)) if ocrates else None,invalid_to_valid=int((~base&valid).sum()),max_reconstruction_error=float(err.max())))
            pair=Counter(zip(assign0[base].tolist(),z['assignments'][base].tolist()))
            for (a,b),v in sorted(pair.items()):transitions.append(dict(class_name=name,condition=condition,from_assignment=a,to_assignment=b,count=v,complement_index=oc,invalid_index=-1))
            cuts=J(p/'baseline_strata.json')['cuts']
            for dimension,vals in [('area',identity['area']),('margin',identity['margin'])]:
                bins=np.searchsorted(cuts[dimension],vals,side='right')
                for q in range(4):
                    selected=base&(bins==q);k=int(selected.sum())
                    strata.append(dict(class_name=name,condition=condition,dimension=dimension,quartile=q,count=k,region_retention=float(retained[selected].mean()) if k else None))
        for f in p.glob('*_batch*_unchanged.npz'):
            z=loadz(f);rr=z['rows'];f0=identity['features'][rr];delta=z['features']-f0;valid=np.any(f0!=0,axis=1)
            sc,_=scorer(z['features'][valid]);old=identity['scores'][rr][valid]
            controls.append(dict(class_name=name,job=j['job_id'],evidence=str(f.relative_to(ROOT)),rows=len(rr),max_abs_feature=float(abs(delta).max()),max_relative_feature=float((norm(delta[valid],axis=1)/norm(f0[valid],axis=1)).max()),max_abs_score=float(abs(sc-old).max()),changed_assignments=int(np.sum(sc.argmax(1)!=identity['assignments'][rr][valid])),zero_identity_equal=bool(np.array_equal(np.all(z['features']==0,axis=1),np.all(f0==0,axis=1))),feature_elementwise_gate=bool(np.allclose(z['features'],f0,rtol=1e-4,atol=1e-4))))
    for name,data in [('robustness-class-status.csv',statuses),('robustness-conditions.csv',rows),('robustness-per-image.csv',perimage),('robustness-reduced-batch-controls.csv',controls),('robustness-transitions.csv',transitions),('robustness-strata.csv',strata)]:table(OUT/name,data)
    macro=[]
    for c in ['identity0','erosion1','erosion2','dilation1','dilation2']:
        rr=[r for r in rows if r['condition']==c]
        macro.append(dict(condition=c,contributing_classes=len(rr),planned_classes=10,equal_class_mean=float(np.mean([r['image_mean_retention'] for r in rr])),status='TEN_CLASS' if len(rr)==10 else 'INCOMPLETE_NOT_TEN_CLASS',class_names=[r['class_name'] for r in rr]))
    return dict(statuses=statuses,macro=macro,reduced_batch_controls=controls,rows=rows)

def medical():
    from utils import utils_general
    from PIL import Image
    p,index=collection('28792059');result=J(p/'result.json');cfg=J(p/'actual_config.json');concepts=J(p/'concept_index.json')
    ck(index['status']=='PASS' and index['commit']=='5623abe8a843deaea7ab838a8c76c41afbad565e','D1 execution PASS')
    d0=ROOT/'artifacts/bunya/medical-discovery-20260922/completed-prerequisite-evidence'
    inputs=J(d0/'dataset_manifest.json');ck(sha(d0/'dataset_manifest.json')==J(p/'D0_binding.json')['manifest_sha256'],'D0 accepted manifest binding')
    ck(len({r['lesion_id'] for r in inputs['training']})==400 and len(inputs['held_out'])==70,'400 distinct lesions / all70 heldout')
    ck(not {r['lesion_id'] for r in inputs['training']} & {r['lesion_id'] for r in inputs['held_out']},'D0 no train/test lesion overlap')
    bs=loadz(p/'scientific/discovery.npz');bases=[bs[k] for k in sorted(bs) if k.startswith('concept_basis_')]+[bs['complement_basis']];scorer=ObliqueScores(bases)
    ck(sum(len(b) for b in bases)==2048 and len(bases)==result['concepts']+1,'Complete medical basis and distinct complement')
    importance=norm(bases[0]@bs['fc_weight'][4])/norm(bs['fc_weight'][4]);ck(abs(float(importance)-result['completeness'])<1e-5,'Completeness independently replayed')
    ck(all(r['pass_check'] for r in J(p/'adapter_checks.json')['checks']),'D1 input/ordinary/masked adapter gates')
    data={};summaries=[];perimage=[];coverage_rows=[]
    for role in ['training','held_out']:
        raw=J(p/(role+'_raw_segments.json'));assign=J(p/(role+'_assignments.json'));z=loadz(p/(role+'_raw_features.npz'));s=loadz(p/'scientific'/(role+'.npz'));full=loadz(p/(role+'_full_images.npz'))
        ck(len(raw)==len(z['features']) and list(z['segment_id'])==[r['segment_id'] for r in raw],role+' raw row identity')
        keep=np.any(z['features']!=0,axis=1);ck(np.isfinite(z['features']).all() and np.isfinite(z['logits']).all(),role+' finite raw feature/logits')
        ck([r['zero_feature'] for r in raw]==(~keep).tolist(),role+' zero flags')
        ck(np.array_equal(z['features'][keep],s['features']) and [r['segment_id'] for r in assign]==list(z['segment_id'][keep]),role+' retained mapping')
        ck(np.array_equal(s['assignments'],[r['assignment'] for r in assign]),role+' assignment mapping')
        for r in assign:ck(r['concept_id']==(concepts[r['assignment']]['concept_id'] if r['assignment']<len(concepts) else 'HUMCD-MEL-COMPLEMENT'),role+' stable ID '+r['segment_id'])
        np.testing.assert_allclose(z['features']@bs['fc_weight'].T+bs['fc_bias'],z['logits'],rtol=1e-4,atol=1e-4)
        score,err=scorer(s['features']);np.testing.assert_allclose(score,s['concept_activations'],rtol=1e-5,atol=1e-5)
        ck(np.array_equal(score.argmax(1),s['assignments']),role+' full score/assignment replay')
        ck(J(p/'scientific'/(role+'_checks.json'))['status']=='PASS',role+' saved reconstruction checks')
        ck(list(full['image_id'])==[r['image_id'] for r in inputs[role]],role+' full-image ordering')
        np.testing.assert_allclose(full['features']@bs['fc_weight'].T+bs['fc_bias'],full['logits'],rtol=1e-4,atol=1e-4)
        _,ferr=scorer(full['features']);group=defaultdict(list);aidx={r['segment_id']:r for r in assign}
        for r in raw:group[r['image_id']].append(r)
        for im in inputs[role]:
            iid=im['image_id'];rr=group[iid];folder=p/'cache'/role
            masks=loadz(folder/(iid+'_effective_masks.npz'))['masks'];original=np.load(folder/(iid+'.png_sam.npy'),allow_pickle=False)
            proposals=loadz(folder/(iid+'_sam_raw.npz'))['masks'];meta=J(folder/(iid+'_sam_metadata.json'))
            ck(len(proposals)==len(meta) and len(masks)==len(original)==len(rr),role+iid+' proposal/raw/effective count')
            for r,m,old in zip(rr,masks,original):
                ck(hashlib.sha256(m.tobytes()).hexdigest()==r['effective_mask_sha256'] and hashlib.sha256(old.tobytes()).hexdigest()==r['raw_mask_sha256'],role+iid+' mask hash '+str(r['raw_index']))
                ck(int(m.sum())==r['effective_pixels'] and int(old.sum())==r['raw_pixels'],role+iid+' mask area')
                effective=utils_general.shrink_mask(old) if old.mean()>.25 else old
                ck(np.array_equal(effective.astype(bool),m),role+iid+' raw-to-effective conditional erosion replay')
            assigned=[aidx[r['segment_id']]['assignment'] if r['segment_id'] in aidx else -1 for r in rr]
            perimage.append(dict(role=role,image_id=iid,lesion_id=im['lesion_id'],sam_proposals=len(proposals),raw_regions=len(rr),retained_regions=sum(a>=0 for a in assigned),zero_features=sum(r['zero_feature'] for r in rr),full_prediction=int(full['logits'][im['order']].argmax())))
            for ci in range(len(concepts)+1):
                sel=np.array(assigned)==ci;v=float(masks[sel].any(0).mean()) if len(masks) else 0.
                coverage_rows.append(dict(role=role,image_id=iid,concept_id=concepts[ci]['concept_id'] if ci<len(concepts) else 'HUMCD-MEL-COMPLEMENT',coverage=v,assigned_regions=int(sel.sum()),status='MEASURED' if sel.any() else 'NO_VALID_ASSIGNED_MEMBER'))
        examples=J(p/(role+'_example_index.json'))
        rng=np.random.RandomState(4301)
        for ex in examples:
            for key in ['top_prototypes','random_final_members']:
                ck(len(ex[key])==len(set(ex[key])) and all(aidx[sid]['concept_id']==ex['concept_id'] for sid in ex[key]),role+' example membership '+ex['concept_id']+key)
            ci=next((i for i,c in enumerate(concepts) if c['concept_id']==ex['concept_id']),len(concepts))
            members=np.flatnonzero(s['assignments']==ci);chosen=rng.choice(members,min(10,len(members)),replace=False) if len(members) else []
            ck([assign[i]['segment_id'] for i in chosen]==ex['random_final_members'],role+' frozen random examples '+ex['concept_id'])
            ordered=members[np.argsort(s['concept_activations'][members,ci])[::-1]];top=[];seen=set()
            for i in ordered:
                if assign[i]['image_id'] not in seen:top.append(assign[i]['segment_id']);seen.add(assign[i]['image_id'])
                if len(top)==10:break
            ck(top==ex['top_prototypes'],role+' original top prototype order '+ex['concept_id'])
        counts=Counter(r['concept_id'] for r in assign)
        summaries.append(dict(role=role,images=len(inputs[role]),lesions=len({r['lesion_id'] for r in inputs[role]}),raw_regions=len(raw),retained_regions=len(assign),zero_features=int((~keep).sum()),zero_fraction=float((~keep).mean()),learned_assignments=sum(v for k,v in counts.items() if k!='HUMCD-MEL-COMPLEMENT'),complement_assignments=counts['HUMCD-MEL-COMPLEMENT'],max_segment_reconstruction_error=float(err.max()),max_full_image_reconstruction_error=float(ferr.max()),MEL_predictions=int(np.sum(full['logits'].argmax(1)==4))))
        data[role]=dict(raw=raw,assign=assign,scientific=s)
    initial=loadz(p/'initial_clusters.npz');train=data['training'];ids=[r['segment_id'] for r in train['assign']]
    ck(list(initial['segment_id'])==ids and not initial['outlier_mask'].any() and np.isfinite(initial['row_l1']).all(),'SSC labels/row order/q1 finite')
    counts=Counter(initial['labels'].tolist());filters=J(p/'cluster_filters.json')
    for f in filters:ck(f['size']==counts[f['cluster_id']] and f['retained']==(f['size']>=50),'Cluster filtering '+str(f['cluster_id']))
    for c in concepts:
        expected={ids[i] for i in np.flatnonzero(initial['labels']==c['cluster_id'])}
        ck(set(c['initial_sorted_segments'])==set(c['pca_fitting_segments'])==expected and len(c['pca_fitting_segments'])==len(expected),'Initial/PCA membership '+c['concept_id'])
    per_image_counts=[r['retained_regions'] for r in perimage if r['role']=='training' and r['retained_regions']>0]
    ck(int(np.mean(per_image_counts)+np.std(per_image_counts))==result['initial_clusters'],'Released initial K heuristic from retained image counts')
    cross=Counter(zip(initial['labels'].tolist(),train['scientific']['assignments'].tolist()))
    xt=[dict(initial_cluster=a,final_assignment=b,count=v,final_concept=concepts[b]['concept_id'] if b<len(concepts) else 'HUMCD-MEL-COMPLEMENT') for (a,b),v in sorted(cross.items())]
    table(OUT/'medical-split-summary.csv',summaries);table(OUT/'medical-per-image.csv',perimage);table(OUT/'medical-image-coverage.csv',coverage_rows);table(OUT/'medical-cluster-to-assignment.csv',xt)
    table(OUT/'medical-concept-index.csv',[{k:v for k,v in c.items() if not isinstance(v,list)} for c in concepts])
    return dict(status='ENGINEERING_ACCEPTED_RESEARCH_LIMITS_DISCLOSED',collection=str(p.relative_to(ROOT)),result=result,splits=summaries,concepts=[{k:v for k,v in c.items() if not isinstance(v,list)} for c in concepts],cluster_to_assignment=xt,empty_or_zero_images=[r for r in perimage if r['retained_regions']==0],scope='No clinical validation, patient independence or semantic interpretability conclusion')

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--line',choices=['A','B','both'],default='both');args=parser.parse_args()
    results={}
    if args.line in ['A','both']:results['robustness']=robustness()
    if args.line in ['B','both']:results['medical']=medical()
    results.update(generated_at=datetime.now(timezone.utc).isoformat(),checks_passed=len(checks),script_sha256=sha(__file__),checks=checks)
    dump(OUT/('acceptance-'+args.line+'.json'),results)
    print(json.dumps({k:v for k,v in results.items() if k not in ['checks','robustness','medical']},indent=2))
