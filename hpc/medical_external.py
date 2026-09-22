"""Approved E224/R101 descriptive transfer of frozen medical HU-MCD concepts.

No fitting, label-dependent sampling, exclusions, classifier selection or retry.
Every invocation requires a Slurm compute node and an approved, hashed protocol.
"""
import csv, gc, hashlib, json, os, random, re, socket, time
from pathlib import Path
import numpy as np
from PIL import Image
from hpc.medical_protocol import digest, bound_inputs
from hpc.medical_sign_analysis import GROUPS, CONTRASTS, coverage, contrast
from utils.run_tracking import atomic_json, sha256, utc_now

CAPS=dict(cpus=8,memory='32G',time='00:30:00',gpu='l40s:1')
MEL=('melanoma','melanoma (in situ)','melanoma (less than 0.76 mm)',
     'melanoma (0.76 to 1.5 mm)','melanoma (more than 1.5 mm)','melanoma metastasis')

def cohort(cases,inventory,root):
    inv={int(r['row_index']):r for r in inventory if r['role']=='derm'}
    selected=[r for r in cases if r['official_split']=='test' and r['diagnosis'] in MEL]
    selected.sort(key=lambda r:int(r['row_index']))
    if len(selected)!=101 or len({r['row_index'] for r in selected})!=101:raise ValueError('R101 cohort identity/count')
    rows=[]
    for r in selected:
        v=inv[int(r['row_index'])]
        if r['derm']!=v['relative_path'] or r['case_num']!=v['case_num'] or v['mode']!='RGB':raise ValueError('Derm record/inventory mismatch')
        rows.append(dict(record=r,inventory=v,source_path=str(Path(root)/'images'/v['relative_path']),image_id='Derm7pt-case'+r['case_num']))
    if len({r['source_path'] for r in rows})!=101:raise ValueError('Unexpected repeated derm paths in R101')
    return rows

def prepare_pixels(image):
    if image.mode!='RGB':raise ValueError('Unexpected non-RGB external input; no silent conversion')
    return np.array(image.resize((224,224),Image.Resampling.BICUBIC),dtype=np.uint8)

def inventory_pixel_digest(image):
    """Exact accepted Derm7pt audit format; distinct from raw-array-only hashes."""
    a=np.array(image)
    return hashlib.sha256(str((image.mode,a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()

def validate(config,approved=True):
    p=config['protocol']
    if config['mode']!='medical-external' or config['resources']!=CAPS or digest(p)!=config['protocol_sha256']:raise ValueError('External protocol/resource identity')
    if approved:
        a=config['authorization']
        if a['status']!='APPROVED' or a['approved_protocol_sha256']!=digest(p) or not a['user_decision']:raise ValueError('E224 execution approval required')
    expected=dict(protocol_id='MED26-E224-R101-S-v1',count=101,preprocessing='PIL_RGB_bicubic_warp224_before_SAM',batch_size=8,target_label=4,
                  fit=False,retrain=False,automatic_retry=False,exclude_overlap_candidates=False,
                  p_values=False,bootstrap=False,work_seconds_ceiling=1680,output_bytes_ceiling=4*1024**3)
    if any(p[k]!=v for k,v in expected.items()):raise ValueError('Unsupported scientific or budget change')
    return p

def run(config,output):
    p=validate(config)
    if not(os.environ.get('SLURM_JOB_ID','').isdigit() and re.fullmatch('bun[0-9]{3}',socket.gethostname().split('.')[0])):raise RuntimeError('Slurm compute allocation required')
    out=Path(output);started=time.monotonic();bound_inputs(p,out)
    def mark(stage,**details):
        atomic_json(out/'external_progress.json',dict(stage=stage,utc=utc_now(),elapsed_seconds=time.monotonic()-started,**details))
        print(stage,details,flush=True)
        if time.monotonic()-started>p['work_seconds_ceiling']:raise TimeoutError('E224 time cap; no retry/extension')
        if sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>p['output_bytes_ceiling']:raise RuntimeError('E224 output cap; no expansion')
    import torch, timm
    from torch.utils.data import DataLoader
    import classes
    from utils import utils_general,utils_mcd,scientific_records
    from hpc.workstream_runtime import event
    if timm.__version__!='0.6.13' or not torch.cuda.is_available():raise RuntimeError('Pinned GPU environment required')
    root=Path(__file__).resolve().parents[1]
    readiness=json.loads((root.parent/'medical-external-readiness.json').read_text())
    if (readiness['status']!='PASS_DEPLOYMENT_READINESS_ONLY' or readiness['commit']!=config['execution_commit'] or
        readiness['protocol_sha256']!=config['protocol_sha256'] or readiness['worker_sha256']!=sha256(root/'hpc/workstream_runtime.py')):
        raise ValueError('Deployment readiness does not match execution')
    for name,value in p['science_sha256'].items():
        if sha256(root/name)!=value:raise ValueError('Frozen science source changed: '+name)
    for name in ['resnet.py','sal_layers.py']:
        if sha256(Path(timm.__file__).parent/'models'/name)!=sha256(root/'input_masking'/name):raise ValueError('Installed masking code mismatch')
    receipt=json.loads(Path(p['inputs']['D1_artifacts']['path']).read_text());dc=json.loads(Path(p['inputs']['D1_config']['path']).read_text())
    acceptance=json.loads(Path(p['inputs']['acceptance']['path']).read_text())
    if receipt['status']!='PASS' or receipt['job_id']!='28792059' or receipt['commit']!=p['discovery_commit'] or acceptance['medical']['status']!='ENGINEERING_ACCEPTED_RESEARCH_LIMITS_DISCLOSED':raise ValueError('Discovery not accepted')
    for key,n in [('basis','scientific/discovery.npz'),('concept_index','concept_index.json'),('D1_config','actual_config.json')]:
        if receipt['files'][n]['sha256']!=p['inputs'][key]['sha256']:raise ValueError('D1 artifact binding '+key)
    if p['inputs']['classifier']!=dc['protocol']['inputs']['classifier'] or p['inputs']['sam']!=dc['protocol']['inputs']['sam'] or p['precision']!=dc['protocol']['precision']:raise ValueError('Medical weights/precision changed')
    with Path(p['inputs']['cases']['path']).open(newline='') as f:cases=list(csv.DictReader(f))
    inventory=json.loads(Path(p['inputs']['inventory']['path']).read_text())
    rows=cohort(cases,inventory,p['derm_root']);payload=json.loads(Path(p['inputs']['cohort']['path']).read_text())
    if rows!=payload['rows'] or payload['grouping']!=p['grouping']:raise ValueError('Frozen R101 or sign grouping mismatch')
    for r in rows:
        for sign in GROUPS:
            if r['record'][sign] not in p['grouping'][sign]:raise ValueError('Unknown sign label; no silent grouping')
    torch.set_num_threads(8);torch.set_float32_matmul_precision(p['precision']['float32_matmul_precision'])
    for k in ['allow_tf32','benchmark','deterministic']:setattr(torch.backends.cudnn,k,p['precision']['cudnn_'+k])
    torch.backends.cuda.matmul.allow_tf32=p['precision']['matmul_allow_tf32'];torch.use_deterministic_algorithms(p['precision']['deterministic_algorithms'])
    random.seed(43);np.random.seed(43);torch.manual_seed(43);utils_general.DEVICE='cuda:0';torch.cuda.reset_peak_memory_stats()
    model=timm.create_model('resnet50',pretrained=False,num_classes=7);model.load_state_dict(torch.load(p['inputs']['classifier']['path'],map_location='cpu'),strict=True);model.to('cuda:0').eval()
    model.default_cfg=dict(model.default_cfg,num_classes=7)
    with np.load(p['inputs']['basis']['path'],allow_pickle=False) as z:
        saved={k:z[k] for k in z.files};bases=[saved[k] for k in sorted(saved) if k.startswith('concept_basis_')]+[saved['complement_basis']]
    if not np.array_equal(saved['fc_weight'],model.fc.weight.detach().cpu().numpy()) or not np.array_equal(saved['fc_bias'],model.fc.bias.detach().cpu().numpy()):raise ValueError('Basis/classifier head identity differs')
    concepts=json.loads(Path(p['inputs']['concept_index']['path']).read_text());ids=[r['concept_id'] for r in concepts]+['HUMCD-MEL-COMPLEMENT']
    if len(ids)!=len(bases) or sum(len(b) for b in bases)!=2048:raise ValueError('Accepted basis geometry changed')
    atomic_json(out/'concept_index.json',concepts);np.savez_compressed(out/'frozen_discovery.npz',**saved)
    cache=out/'cache';cache.mkdir();work=out/'inputs';work.mkdir();images=[];actual=[]
    for order,r in enumerate(rows):
        source=Path(r['source_path'])
        if sha256(source)!=r['inventory']['sha256']:raise ValueError('External source file changed')
        with Image.open(source) as im:
            a=np.array(im)
            if inventory_pixel_digest(im)!=r['inventory']['pixel_sha256']:raise ValueError('External source pixel identity changed')
            pixels=prepare_pixels(im)
        path=work/(r['image_id']+'.png');Image.fromarray(pixels).save(path)
        im=classes.ImageClass(str(path),max_shortest_side=300);im.image_id=r['image_id'];images.append(im)
        actual.append(dict(r,order=order,input_path=str(path),input_sha256=sha256(path),input_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),input_size=[224,224],adjustment=p['preprocessing']))
    atomic_json(out/'actual_input_mapping.json',actual);mark('INPUTS_VERIFIED',images=len(images))
    event(out,'inputs','MED26-E224-GEOMETRY',dict(images=101,method=p['preprocessing']),'Pre-SAM warp changes original aspect ratio and fine structures; controlled size transfer only','Preserve original dimensions/hash and explicit work inputs',status='DISCLOSED')
    event(out,'scope','MED26-E224-OVERLAP',dict(R101_threshold_pairs=214,R101_candidate_cases=20,review='UNREVIEWED_UNLESS_USER_RECORD_UPDATED'),'Negative exact checks do not prove independence; perceptual candidates not confirmed duplicates','No exclusions; association conditional on current cohort, no patient-independent claim',status='DISCLOSED')
    # Same arithmetic/input path as accepted D1; no repeated GPU adapter/SAM probe.
    for im in images:im.segments=[classes.SegmentClass(np.ones((224,224),np.float32),im)]
    full=classes.ConceptDatasetClass(images,model.default_cfg,0,False)
    for i in range(min(8,len(images))):
        expected=(torch.from_numpy(np.array(images[i].img_pil).copy()).permute(2,0,1).float()/255-torch.tensor(model.default_cfg['mean'])[:,None,None])/torch.tensor(model.default_cfg['std'])[:,None,None]
        if not torch.equal(full[i],expected):raise ValueError('E224 input tensor arithmetic differs')
    fa,fl=utils_general.compute_activations(model,'global_pool',DataLoader(full,batch_size=8,shuffle=False,collate_fn=utils_general.custom_collate))
    np.savez_compressed(out/'full_images.npz',features=fa,logits=fl,image_id=np.array([im.image_id for im in images]))
    checks,samples=scientific_records.numerical_checks(fa,fl[:,4],saved['fc_weight'][4],saved['fc_bias'][4],bases);atomic_json(out/'full_checks.json',checks);np.savez_compressed(out/'full_checks.npz',**samples)
    mark('FULL_IMAGE_INFERENCE_COMPLETE')
    generator=utils_general.load_sam_mask_generator(sam_type='vit_h',sam_checkpoint=p['inputs']['sam']['path'],points_per_side=32,min_mask_region_area=256)
    class Observed:
        image_id=None
        def generate(self,a):
            proposals=generator.generate(a);np.savez_compressed(cache/(self.image_id+'_sam_raw.npz'),masks=np.array([r['segmentation'] for r in proposals],dtype=bool))
            atomic_json(cache/(self.image_id+'_sam_metadata.json'),[{k:v for k,v in r.items() if k!='segmentation'} for r in proposals]);return proposals
    observed=Observed()
    for i,im in enumerate(images):
        im.segments=[];observed.image_id=im.image_id;im.load_segments(str(cache),observed)
        for j,s in enumerate(im.segments):s.stable_id=f'HUMCD-MEL-External-{im.image_id}-S{j:04d}'
        mark('SEGMENTATION',completed=i+1,total=101)
    del generator,observed;gc.collect();torch.cuda.empty_cache()
    segments=[s for im in images for s in im.segments];ds=classes.ConceptDatasetClass(images,model.default_cfg,0,True,-1,.25);mapping=[];offset=0;mask_by_image={}
    for im in images:
        effective=[]
        for j,s in enumerate(im.segments):
            m=ds[offset][1].numpy()[0].astype(bool);effective.append(m)
            mapping.append(dict(segment_id=s.stable_id,image_id=im.image_id,global_row=offset,raw_index=j,raw_mask_sha256=hashlib.sha256(s.mask.tobytes()).hexdigest(),effective_mask_sha256=hashlib.sha256(m.tobytes()).hexdigest(),raw_pixels=int(s.mask.sum()),effective_pixels=int(m.sum())))
            offset+=1
        masks=np.array(effective,dtype=bool).reshape(-1,224,224);mask_by_image[im.image_id]=masks;np.savez_compressed(cache/(im.image_id+'_effective_masks.npz'),masks=masks)
    if not segments:raise ValueError('No external regions; evidence kept, no rescue')
    acts,logits=utils_general.compute_activations(model,'global_pool',DataLoader(ds,batch_size=8,shuffle=False,collate_fn=utils_general.custom_collate))
    valid=np.any(acts!=0,axis=1);scores=np.full((len(acts),len(bases)),np.nan);assign=np.full(len(acts),-1,dtype=np.int32)
    checks,_=scientific_records.numerical_checks(acts,logits,saved['fc_weight'],saved['fc_bias'],None)
    # Exact released per-vector projection, no fitting or batch normalization.
    if valid.any():
        scores[valid]=utils_mcd.batch_concept_activations(acts[valid],bases,False)
        if not np.isfinite(scores[valid]).all():raise ValueError('Nonfinite external concept scores')
        assign[valid]=scores[valid].argmax(1)
        decomposition,samples=scientific_records.numerical_checks(acts[valid],logits[valid,4],saved['fc_weight'][4],saved['fc_bias'][4],bases)
        checks['decomposition']=decomposition;np.savez_compressed(out/'segment_checks.npz',**samples)
    atomic_json(out/'segment_checks.json',checks)
    np.savez_compressed(out/'segments.npz',features=acts,logits=logits,scores=scores,assignments=assign,valid=valid,segment_id=np.array([s.stable_id for s in segments]))
    for i,r in enumerate(mapping):r.update(zero_feature=not bool(valid[i]),assignment=int(assign[i]),concept_id=ids[assign[i]] if valid[i] else None)
    atomic_json(out/'segment_mapping.json',mapping);mark('ASSIGNMENTS_COMPLETE',regions=len(acts),zeros=int((~valid).sum()))
    if (~valid).any():event(out,'assignment','MED26-E224-ZERO',dict(regions=len(acts),zero_features=int((~valid).sum())),'No valid concept assignment for exact-zero vectors','Retain raw rows; coverage excludes invalid assignments, no C0 argmax substitution',status='DISCLOSED')
    union=[];offset=0;response=np.empty((101,len(ids)))
    for i,(im,row) in enumerate(zip(images,rows)):
        n=len(im.segments);v=coverage(mask_by_image[im.image_id],assign[offset:offset+n],valid[offset:offset+n],list(range(len(ids))))
        for c,rec in v.items():
            response[i,c]=rec['value'];union.append(dict(image_id=im.image_id,case_num=row['record']['case_num'],row_index=row['record']['row_index'],concept_id=ids[c],**rec))
        offset+=n
    atomic_json(out/'coverage.json',union);analyses=[]
    for c,cid in enumerate(ids):
        for sign,group,ref in CONTRASTS:
            labels=np.array([p['grouping'][sign][r['record'][sign]] for r in rows]);record=contrast(response[:,c],labels,group,ref)
            positive=response[labels==group,c];negative=response[labels==ref,c]
            record.update(concept_id=cid,is_complement=c==len(concepts),sign=sign,group_mean=float(positive.mean()) if len(positive) else None,reference_mean=float(negative.mean()) if len(negative) else None,mean_difference=float(positive.mean()-negative.mean()) if len(positive) and len(negative) else None)
            analyses.append(record)
    atomic_json(out/'descriptive_contrasts.json',analyses)
    np.savez_compressed(out/'coverage_matrix.npz',coverage=response,image_id=np.array([r['image_id'] for r in rows]),concept_id=np.array(ids))
    mark('COMPLETE_PENDING_ACCEPTANCE',images=101,learned_concepts=len(concepts),comparisons=len(analyses))
    return dict(status='COMPLETED_PENDING_ACCEPTANCE',images=101,raw_regions=len(acts),valid_regions=int(valid.sum()),zero_features=int((~valid).sum()),MEL_predictions=int((fl.argmax(1)==4).sum()),learned_concepts=len(concepts),learned_comparisons=12*len(concepts),complement_comparisons=12,elapsed_seconds=time.monotonic()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),no_refit=True,clinical_utility_certified=False,patient_independence_unknown=True)
