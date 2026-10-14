"""Local cached acceptance of E224: no predictions, fitting, or remote work."""
import csv, hashlib, io, json, random, sys, zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hpc.mask_sensitivity import ObliqueScores
from hpc.medical_external import cohort, validate
from utils import utils_general

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/bunya/medical-external-results-20260923'
P=OUT/'collection'
PREP=ROOT/'artifacts/bunya/medical-external-20260922'
D1=ROOT/'artifacts/bunya/batch-review-20260922/collections/28792059/20260922T063256.936418Z'
DATA=ROOT/'artifacts/bunya/next-phase-20260921/derm7pt-resumption/collections/28734478/20260921T051458.316863Z'
COMMIT='dcc61934973aa9054a4e689ccf2334652447c575'
J=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def z(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
checks=[]
def ck(ok,label):
    if not ok:raise ValueError(label)
    checks.append(label)

def main():
    a=J(P/'artifacts.json');cfg=J(P/'actual_config.json');launch=J(P/'launch_manifest.json');result=J(P/'result.json')
    ck(J(OUT/'collection.json')['status']=='COMPLETE','Complete transfer')
    for name,e in a['files'].items():ck(sha(P/name)==e['sha256'] and (P/name).stat().st_size==e['bytes'],'Artifact '+name)
    ck(a['status']=='PASS' and a['commit']==cfg['execution_commit']==launch['commit']==COMMIT,'Execution and full commit binding')
    ck(cfg==launch['plan']['config'],'Resolved configuration equals submitted configuration')
    protocol=validate(cfg);ready=J(OUT/'medical-external-readiness.json')
    ck(ready['commit']==COMMIT and ready['status']=='PASS_DEPLOYMENT_READINESS_ONLY' and ready['protocol_sha256']==cfg['protocol_sha256'],'Deployment identity')
    jobs=list(csv.DictReader((OUT/'sacct.psv').open(),delimiter='|'))
    for job in ['28796207','28796209']:
        row=next(r for r in jobs if r['JobID']==job);ck(row['State']=='COMPLETED' and row['ExitCode']=='0:0','Terminal accounting '+job)
    for name,e in J(P/'input_binding.json').items():ck(e['sha256']==e['expected_sha256']==protocol['inputs'][name]['sha256'],'Runtime input binding '+name)
    local_inputs={'D1_artifacts':D1/'artifacts.json','D1_config':D1/'actual_config.json','basis':D1/'scientific/discovery.npz','concept_index':D1/'concept_index.json','cases':DATA/'cases.csv','inventory':DATA/'image_inventory.json','cohort':PREP/'cohort.json','acceptance':PREP/'acceptance.json'}
    for k,f in local_inputs.items():ck(sha(f)==protocol['inputs'][k]['sha256'],'Accepted local dependency '+k)
    ck(sha(P/'concept_index.json')==sha(D1/'concept_index.json'),'Stable concept identity unchanged')
    ck(J(D1/'actual_config.json')['protocol']['precision']==protocol['precision'],'Same source precision')
    for n,v in protocol['science_sha256'].items():ck(sha(ROOT/n)==v,'Frozen science '+n)
    bs=z(P/'frozen_discovery.npz');old=z(D1/'scientific/discovery.npz')
    ck(bs.keys()==old.keys() and all(np.array_equal(bs[k],old[k]) for k in bs),'All frozen basis/FC arrays identical to accepted D1')
    bases=[bs[k] for k in sorted(bs) if k.startswith('concept_basis_')]+[bs['complement_basis']];scorer=ObliqueScores(bases)
    ids=['HUMCD-MEL-C001','HUMCD-MEL-COMPLEMENT']
    ck([len(b) for b in bases]==[32,2016],'One learned32-D concept plus complement')
    rows=cohort(list(csv.DictReader((DATA/'cases.csv').open())),J(DATA/'image_inventory.json'),protocol['derm_root'])
    ck(rows==J(PREP/'cohort.json')['rows'],'All101 frozen records, labels, ordering')
    actual=J(P/'actual_input_mapping.json');mapping=J(P/'segment_mapping.json');s=z(P/'segments.npz');f=z(P/'full_images.npz');cv=z(P/'coverage_matrix.npz')
    ck(len(actual)==101 and [r['image_id'] for r in rows]==f['image_id'].tolist()==cv['image_id'].tolist(),'101 full image/coverage identities')
    ck(s['segment_id'].tolist()==[r['segment_id'] for r in mapping] and len(set(s['segment_id']))==len(mapping)==296,'296 unique ordered segment IDs')
    ck(np.isfinite(s['features']).all() and np.isfinite(s['logits']).all() and np.isfinite(f['features']).all() and np.isfinite(f['logits']).all(),'Finite full-image and raw-region features/logits')
    valid=np.any(s['features']!=0,axis=1)
    ck(np.array_equal(s['valid'],valid) and np.isnan(s['scores'][~valid]).all() and np.all(s['assignments'][~valid]==-1),'Exact-zero invalidity and no arbitrary C0 assignment')
    scores,err=scorer(s['features'][valid]);_,ferr=scorer(f['features'])
    np.testing.assert_allclose(scores,s['scores'][valid],atol=1e-5,rtol=1e-5)
    ck(np.array_equal(scores.argmax(1),s['assignments'][valid]),'All valid final assignments replayed')
    fcerrors={}
    for name,t in [('full',f),('segments',s)]:
        pred=t['features']@bs['fc_weight'].T+bs['fc_bias'];np.testing.assert_allclose(pred,t['logits'],rtol=1e-4,atol=1e-4);fcerrors[name]=float(abs(pred-t['logits']).max())
    ck(J(P/'external_progress.json')['stage']=='COMPLETE_PENDING_ACCEPTANCE','Final stage reached')
    unions=J(P/'coverage.json');perimage=[];perregion=[];offset=0;replayed=[]
    # Released composition assigns random integer region IDs, then sorts them.
    # Replay the worker's Python RNG and the complete frozen image order.
    random.seed(43)
    with zipfile.ZipFile('/mnt/c/Users/uqcche38/Downloads/release_v0.zip') as archive:
        for i,(r,act) in enumerate(zip(rows,actual)):
            iid=r['image_id'];ck(all(act[k]==r[k] for k in r) and act['order']==i,'Actual mapping '+iid)
            source=archive.read('release_v0/images/'+r['inventory']['relative_path']);ck(hashlib.sha256(source).hexdigest()==r['inventory']['sha256'],'Original source bytes '+iid)
            with Image.open(io.BytesIO(source)) as im:
                original=np.array(im);expected=np.array(im.resize((224,224),Image.Resampling.BICUBIC));size=im.size
                ck(im.mode=='RGB' and list(im.size)==[r['inventory']['width'],r['inventory']['height']],'Original geometry '+iid)
            inp=P/'inputs'/(iid+'.png');pixels=np.array(Image.open(inp))
            ck(np.array_equal(pixels,expected) and sha(inp)==act['input_sha256'] and hashlib.sha256(pixels.tobytes()).hexdigest()==act['input_pixel_sha256'],'Approved E224 pixels '+iid)
            raw=np.load(P/'cache'/(iid+'.png_sam.npy'),allow_pickle=False);m=z(P/'cache'/(iid+'_effective_masks.npz'))['masks'];props=z(P/'cache'/(iid+'_sam_raw.npz'))['masks'];meta=J(P/'cache'/(iid+'_sam_metadata.json'))
            ck(len(props)==len(meta) and raw.shape==m.shape and m.dtype==bool,'SAM/raw/effective geometry '+iid)
            ordered=[props[j] for j in sorted(range(len(meta)),key=lambda j:meta[j]['predicted_iou'],reverse=True)]
            replay=np.array(utils_general.get_unique_concept_masks(ordered,.01),dtype=np.float32)
            ck(np.array_equal(raw,replay),'Saved SAM proposals to composed masks '+iid)
            rr=mapping[offset:offset+len(m)];assignment=s['assignments'][offset:offset+len(m)];keep=valid[offset:offset+len(m)]
            for j,(mr,mask,rm) in enumerate(zip(rr,m,raw)):
                k=offset+j;ck(mr['image_id']==iid and mr['raw_index']==j and mr['global_row']==k and mr['segment_id']==f'HUMCD-MEL-External-{iid}-S{j:04d}','Image/region association '+str(k))
                ck(hashlib.sha256(mask.tobytes()).hexdigest()==mr['effective_mask_sha256'] and hashlib.sha256(rm.tobytes()).hexdigest()==mr['raw_mask_sha256'],'Both mask hashes '+str(k))
                ck(int(mask.sum())==mr['effective_pixels'] and int(rm.sum())==mr['raw_pixels'],'Both mask areas '+str(k))
                effective=utils_general.shrink_mask(rm) if rm.mean()>.25 else rm
                ck(np.array_equal(effective.astype(bool),mask),'Conditional erosion '+str(k))
                ck(mr['zero_feature']==bool(~valid[k]) and mr['assignment']==int(s['assignments'][k]) and mr['concept_id']==(ids[s['assignments'][k]] if valid[k] else None),'Mapping validity '+str(k))
                perregion.append({**mr,'assignment_score':float(s['scores'][k,s['assignments'][k]]) if valid[k] else None,'margin':float(abs(np.diff(s['scores'][k]))[0]) if valid[k] else None})
            response=[]
            for ci,cid in enumerate(ids):
                members=keep&(assignment==ci);value=float(m[members].any(axis=0).mean());response.append(value)
                saved=unions[i*2+ci];ck(saved['image_id']==iid and saved['concept_id']==cid and saved['valid_member_count']==int(members.sum()) and saved['value']==value and saved['status']==('MEASURED' if members.any() else 'NO_VALID_ASSIGNED_MEMBER'),'Coverage union '+iid+cid)
            replayed.append(response)
            perimage.append(dict(image_id=iid,case_num=r['record']['case_num'],row_index=r['record']['row_index'],diagnosis=r['record']['diagnosis'],source_width=size[0],source_height=size[1],sam_proposals=len(props),raw_regions=len(m),valid_regions=int(keep.sum()),zero_features=int((~keep).sum()),empty_masks=int((~m.any((1,2))).sum()),learned_assignments=int((assignment==0).sum()),complement_assignments=int((assignment==1).sum()),C001_coverage=response[0],complement_coverage=response[1],prediction=int(f['logits'][i].argmax()),MEL_logit=float(f['logits'][i,4])))
            offset+=len(m)
    ck(offset==296 and np.array_equal(replayed,cv['coverage']) and cv['concept_id'].tolist()==ids,'All coverage rows/matrix and segment totals')
    contrasts=J(P/'descriptive_contrasts.json');ck(len(contrasts)==24 and sum(not c['is_complement'] for c in contrasts)==12,'12 learned +12 separately reported complement comparisons')
    for c in contrasts:
        ci=ids.index(c['concept_id']);labels=np.array([protocol['grouping'][c['sign']][r['record'][c['sign']]] for r in rows]);a=cv['coverage'][labels==c['group'],ci];b=cv['coverage'][labels=='ABS',ci]
        auc=float(((a[:,None]>b[None,:])+.5*(a[:,None]==b[None,:])).mean())
        ck((len(a),len(b))==(c['group_n'],c['reference_n']) and c['missing_n']==0 and c['other_category_n']==101-len(a)-len(b),'Contrast denominators '+c['concept_id']+c['sign']+c['group'])
        ck(abs(auc-c['auc'])<1e-12 and abs(float(a.mean()-b.mean())-c['mean_difference'])<1e-12 and c['p_value'] is None,'Independent pairwise AUC and mean difference '+c['concept_id']+c['sign']+c['group'])
    predictions=Counter(f['logits'].argmax(1).tolist());ck(predictions[4]==result['MEL_predictions']==29 and int(valid.sum())==result['valid_regions']==259 and int((~valid).sum())==result['zero_features']==37,'Worker totals reconciled')
    summaries=[]
    for ci,cid in enumerate(ids):
        v=cv['coverage'][:,ci];summaries.append(dict(concept_id=cid,is_learned=ci==0,assigned_regions=int((s['assignments']==ci).sum()),images_with_members=sum(r['learned_assignments' if ci==0 else 'complement_assignments']>0 for r in perimage),mean_coverage=float(v.mean()),median_coverage=float(np.median(v)),min_coverage=float(v.min()),max_coverage=float(v.max()),zero_coverage_images=int((v==0).sum())))
    table(OUT/'per-image.csv',perimage);table(OUT/'per-region.csv',perregion);table(OUT/'contrasts.csv',contrasts);table(OUT/'concept-summary.csv',summaries);table(OUT/'job-results.csv',jobs)
    accept=dict(status='ENGINEERING_ACCEPTED_RESEARCH_LIMITS_DISCLOSED',generated_at=datetime.now(timezone.utc).isoformat(),job_id='28796209',commit=COMMIT,result=result,collection_sha256=sha(P/'artifacts.json'),script_sha256=sha(__file__),checks_passed=len(checks),checks=checks,fc_max_abs_error=fcerrors,max_region_reconstruction_error=float(err.max()),max_full_reconstruction_error=float(ferr.max()),predictions={str(k):predictions[k] for k in range(7)},concept_summary=summaries,all_invalid_images=[r['image_id'] for r in perimage if r['valid_regions']==0],empty_masks=sum(r['empty_masks'] for r in perimage),shape_warp_images=sum(r['source_width']!=r['source_height'] for r in perimage),no_new_inference=True,scope='Frozen classifier/concept transfer on R101 E224; no clinical certification, causal domain attribution, localization or patient-independence claim')
    dump(OUT/'acceptance.json',accept);print(json.dumps({k:v for k,v in accept.items() if k!='checks'},indent=2))

if __name__=='__main__':main()
