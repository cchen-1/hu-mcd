"""Approved medical D0/D1: fixed classifier and released HU discovery operations.

No training, external Derm7pt inference, auto retry or altered outlier quantile.
All remote entry points require Slurm compute allocation and exact config digest.
"""
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import socket
import time

import numpy as np
from hpc.medical_protocol import digest, bound_inputs, manifest_rows
from utils.run_tracking import atomic_json, sha256, utc_now

CAPS={'medical-discovery-inputs':dict(cpus=1,memory='4G',time='00:10:00',gpu=None),
      'medical-discovery':dict(cpus=8,memory='32G',time='02:00:00',gpu='l40s:1')}
CHECKPOINT='07a6f65dbb23eb7c97510bef09c4b624a1a86007b27a64e324151fd1dfd2a655'
SAM='a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e'


def select_inputs(rows):
    """Separate legacy RandomState43: sorted lesion IDs, then sorted image IDs."""
    grouped={}
    for r in rows['train']:
        if int(r['label'])==4:grouped.setdefault(r['lesion_id'],[]).append(r)
    lesions=sorted(grouped);rng=np.random.RandomState(43)
    selected=rng.choice(lesions,size=400,replace=False).tolist()
    train=[dict(sorted(grouped[l],key=lambda r:r['image_id'])[int(rng.choice(len(grouped[l])))]) for l in selected]
    test=[dict(r) for r in rows['test'] if int(r['label'])==4]
    return dict(candidate_lesions=lesions,selected_lesions_in_order=selected,training=train,held_out=test,
                rule='NumPy RandomState(43).choice(sorted lesion IDs,400,replace=False); in sampled order, choice index from each image-ID-sorted lesion; held-out original array order',
                seed_role='discovery sampling only, not classifier training',candidate_images=sum(map(len,grouped.values())))


def validate(config,approved=True):
    mode=config['mode'];p=config['protocol'];a=config['authorization']
    if mode not in CAPS or config['resources']!=CAPS[mode] or digest(p)!=config['protocol_sha256']:
        raise ValueError('Medical discovery protocol/resource identity mismatch')
    if approved and (a['status']!='APPROVED' or a['approved_protocol_sha256']!=digest(p) or not a['user_decision']):
        raise ValueError('Explicit discovery authorization required')
    if p['target_label']!=4 or p['training_images']!=400 or p['held_out_images']!=70 or p['sampling_seed']!=43:
        raise ValueError('Unapproved input scope')
    if p['outlier_quantile']!=1.0 or p['min_cluster_size']!=50 or p['batch_size']!=8:
        raise ValueError('Changed released discovery rule')
    if p['inputs']['classifier']['sha256']!=CHECKPOINT or p['inputs']['sam']['sha256']!=SAM:
        raise ValueError('Unapproved weights')
    if p['external_inference'] or p['classifier_training'] or p['automatic_retry']:
        raise ValueError('Unapproved experiment')
    cap=512*1024**2 if mode=='medical-discovery-inputs' else 15*1024**3
    seconds=540 if mode=='medical-discovery-inputs' else 6900
    if p['output_bytes_ceiling']!=cap or p['work_seconds_ceiling']!=seconds:raise ValueError('Changed output/time budget')
    return p


def run(config,output):
    p=validate(config);out=Path(output)
    if not (os.environ.get('SLURM_JOB_ID','').isdigit() and re.fullmatch('bun[0-9]{3}',socket.gethostname().split('.')[0])):
        raise RuntimeError('Medical discovery requires Slurm compute allocation')
    started=time.monotonic()
    def budget(stage,**kw):
        atomic_json(out/'medical_discovery_progress.json',dict(stage=stage,utc=utc_now(),elapsed_seconds=time.monotonic()-started,**kw))
        if time.monotonic()-started>p['work_seconds_ceiling']:raise TimeoutError('Discovery time budget exhausted; no extension')
        if sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>p['output_bytes_ceiling']:raise RuntimeError('Discovery output/cache budget exhausted')
    # Hash only inputs used by this stage; accepted model/SAM bindings are deferred to D1.
    used={'source_npz','source_manifest'} if config['mode']=='medical-discovery-inputs' else {'classifier','classifier_identity','sam'}
    bound_inputs({'inputs':{k:v for k,v in p['inputs'].items() if k in used}},out)
    return prepare(config,out,budget) if config['mode']=='medical-discovery-inputs' else discover(config,out,budget)


def prepare(config,out,budget):
    from PIL import Image
    p=config['protocol'];rr=manifest_rows(p['inputs']['source_manifest']['path']);selection=select_inputs(rr)
    if digest(selection)!=p['selection_sha256'] or len(selection['candidate_lesions'])!=533 or selection['candidate_images']!=1021:
        raise ValueError('Approved selection differs from accepted metadata')
    if len(selection['held_out'])!=70 or len({r['lesion_id'] for r in selection['held_out']})!=61:
        raise ValueError('Fixed held-out identity changed')
    train=selection['training'];test=selection['held_out']
    if ({r['image_id'] for r in train}&{r['image_id'] for r in test} or
        {r['lesion_id'] for r in train}&{r['lesion_id'] for r in test} or len({r['lesion_id'] for r in train})!=400):
        raise ValueError('Discovery/held-out overlap or lesion multiplicity')
    atomic_json(out/'selection.json',selection)
    actual={}
    with np.load(p['inputs']['source_npz']['path'],allow_pickle=False) as z:
        for role,split,records in [('training','train',train),('held_out','test',test)]:
            arr=z[split+'_images'];labels=z[split+'_labels'].reshape(-1);folder=out/role;folder.mkdir()
            actual[role]=[]
            for order,row in enumerate(records):
                i=int(row['array_row']);a=arr[i]
                if a.shape!=(224,224,3) or a.dtype!=np.uint8 or int(labels[i])!=4:raise ValueError('Selected image geometry/label mismatch')
                pixel=hashlib.sha256(a.tobytes()).hexdigest()
                if pixel!=row['pixel_sha256']:raise ValueError('Selected pixels differ from accepted manifest')
                path=folder/(row['image_id']+'.png');Image.fromarray(a).save(path)
                with Image.open(path) as im:
                    if im.mode!='RGB' or not np.array_equal(np.array(im),a):raise ValueError('Lossless input staging mismatch')
                actual[role].append(dict(row,order=order,input_path=str(path),input_sha256=sha256(path),input_pixel_sha256=pixel))
                budget('D0_staging',role=role,completed=order+1,total=len(records))
            del arr
    manifest=dict(status='PASS_INPUT_IDENTITY_ONLY',job_id=os.environ['SLURM_JOB_ID'],commit=config['execution_commit'],
                  selection_sha256=digest(selection),source_inputs=p['inputs'],training=actual['training'],held_out=actual['held_out'],
                  input_adjustment='Lossless PNG serialization of accepted NPZ RGB pixels; no resizing, augmentation, exclusion or diagnosis change',
                  patient_independence='NOT_ESTABLISHED')
    atomic_json(out/'dataset_manifest.json',manifest)
    for role,rr in actual.items():
        with (out/(role+'_manifest.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    budget('D0_COMPLETE',training=400,held_out=70)
    return dict(status='PASS_INPUT_IDENTITY_ONLY',selection_sha256=digest(selection),dataset_manifest_sha256=sha256(out/'dataset_manifest.json'),training=400,training_lesions=400,held_out=70,held_out_lesions=61)


def discover(config,out,budget,fit_context=None):
    import torch
    import timm
    from torch.utils.data import DataLoader
    from joblib import parallel_backend
    import classes
    from concept_explainer import ConceptExplainer
    from utils import utils_general,utils_mcd,scientific_records
    from hpc.medical_classifier import adapter_checks
    from hpc.workstream_runtime import event
    p=config['protocol'];prefix='HUMCD-MEL'
    if fit_context is None:
        p=config['protocol'];d0=Path(config['prepared_inputs']);source=json.loads((d0/'dataset_manifest.json').read_text())
        receipt=json.loads((d0/'artifacts.json').read_text());launch=json.loads((d0/'launch_manifest.json').read_text())
        if (receipt['status']!='PASS' or launch['status']!='PASS' or source['job_id']!=receipt['job_id'] or source['job_id']!=d0.name or source['commit']!=config['execution_commit'] or
            receipt['commit']!=config['execution_commit'] or source['selection_sha256']!=p['selection_sha256'] or
            sha256(d0/'dataset_manifest.json')!=receipt['files']['dataset_manifest.json']['sha256']):raise ValueError('D0 prerequisite not complete/matching')
        atomic_json(out/'D0_binding.json',dict(job_id=source['job_id'],manifest_sha256=sha256(d0/'dataset_manifest.json'),manifest_path=str(d0/'dataset_manifest.json'),selection_sha256=p['selection_sha256']))
        frozen=json.loads(Path(p['inputs']['classifier_identity']['path']).read_text())
        if frozen['selected_epoch']!=10 or frozen['checkpoint_sha256']!=CHECKPOINT:raise ValueError('Frozen classifier identity differs')
    else:
        source=fit_context;d0=Path(p['inputs']['fit_manifest']['path']).parent
        prefix='HUMCD-DERM7PT-R101-MEL'
        frozen=json.loads(Path(p['inputs']['classifier_identity']['path']).read_text())
        if frozen['checkpoint_sha256']!=p['inputs']['classifier']['sha256'] or frozen['selected_epoch']!=p['selected_epoch']:
            raise ValueError('Frozen HAM classifier identity differs')
        atomic_json(out/'fit_input_binding.json',source)
    checkpoint_hash=p['inputs']['classifier']['sha256']
    if timm.__version__!='0.6.13' or not torch.cuda.is_available():raise RuntimeError('Pinned model runtime and GPU required')
    import timm.models.resnet as ir
    root=Path(__file__).resolve().parents[1]
    for name in ['resnet.py','sal_layers.py']:
        if sha256(Path(ir.__file__).parent/name)!=sha256(root/'input_masking'/name):raise ValueError('Masking implementation changed')
    torch.set_num_threads(8);torch.set_float32_matmul_precision(p['precision']['float32_matmul_precision'])
    for k in ['allow_tf32','benchmark','deterministic']:setattr(torch.backends.cudnn,k,p['precision']['cudnn_'+k])
    torch.backends.cuda.matmul.allow_tf32=p['precision']['matmul_allow_tf32'];torch.use_deterministic_algorithms(p['precision']['deterministic_algorithms'])
    utils_general.DEVICE='cuda:0';torch.cuda.reset_peak_memory_stats();random.seed(43);np.random.seed(43);torch.manual_seed(43)
    model=timm.create_model('resnet50',pretrained=False,num_classes=7)
    model.load_state_dict(torch.load(p['inputs']['classifier']['path'],map_location='cpu'),strict=True);model.to('cuda:0').eval()
    model.default_cfg=dict(model.default_cfg,num_classes=7)
    if list(model.default_cfg['mean'])!=p['mean'] or list(model.default_cfg['std'])!=p['std']:raise ValueError('Classifier/discovery preprocessing mismatch')
    atomic_json(out/'environment.json',dict(torch=torch.__version__,timm=timm.__version__,gpu=torch.cuda.get_device_name(),precision=p['precision'],classifier_sha256=checkpoint_hash,training=False))
    images={}
    for role,count in [('training',p['training_images'])]+([('held_out',p['held_out_images'])] if p['held_out_images'] else []):
        if len(source[role])!=count:raise ValueError('Incorrect fixed input count')
        images[role]=[]
        for row in source[role]:
            if sha256(row['input_path'])!=row['input_sha256']:raise ValueError('Prepared input changed')
            im=classes.ImageClass(row['input_path'],max_shortest_side=300)
            expected_wh=(224,224) if fit_context is None else tuple(row['input_size_wh'])
            if im.img_pil.size!=expected_wh:raise ValueError('Bound pre-SAM geometry changed')
            im.image_id=row['image_id'];images[role].append(im)
    # Test actual released ConceptDataset preprocessing against classifier input arithmetic,
    # on the first same batch used for GPU adapter checks; no SAM probe.
    eight=images['training'][:8]
    for im in eight:im.segments=[classes.SegmentClass(np.ones(im.img_numpy.shape[:2],np.float32),im)]
    ds=classes.ConceptDatasetClass(eight,model.default_cfg,cropping_mode=0,use_masks=False)
    x=torch.stack([ds[i] for i in range(8)])
    if fit_context is None:
        expected=torch.stack([(torch.from_numpy(np.array(im.img_pil).copy()).permute(2,0,1).float()/255-torch.tensor(p['mean'])[:,None,None])/torch.tensor(p['std'])[:,None,None] for im in eight])
    else:
        from hpc.ham_input_preparation import released_full_image_input
        expected=torch.stack([released_full_image_input(row['source_path'],model.default_cfg)[0]
                              for row in source['training'][:8]])
    if not torch.equal(x,expected):raise ValueError('Actual HU input tensors differ from frozen classifier preprocessing')
    adapter_checks(model,x.to('cuda:0'),out,source['training'][:8])
    adapter=json.loads((out/'adapter_checks.json').read_text());adapter['selection']='First8 fixed discovery training rows, not classifier validation; same batch';adapter['input_preprocessing_exact_equal']=True;atomic_json(out/'adapter_checks.json',adapter)
    for im in eight:im.segments=[]
    budget('D1_ADAPTER_PASS')
    class MedicalExplainer(ConceptExplainer):
        def __init__(self):
            self.model=model;self.target_class='melanoma';self.layer_name='global_pool';self.class_imgs=images['training'];self.source_dir=str(d0)
            self.max_shortest_side=300;self.segm_algo='sam';self.max_samples_per_concept=None
        def concept_relevances(self,acts,n_jobs=-1):
            return np.stack([utils_mcd.subspace_projection(self.concept_bases+[self.compl_basis],v) for v in acts])@model.fc.weight.detach().cpu().numpy()[4]
    exp=MedicalExplainer();cache=out/'cache';cache.mkdir();times={};start=time.monotonic()
    raw_records={};full_records={}
    def process(role):
        ims=images[role];stage=time.monotonic();folder=cache/role;folder.mkdir()
        generator=utils_general.load_sam_mask_generator(sam_type='vit_h',sam_checkpoint=p['inputs']['sam']['path'],points_per_side=32,min_mask_region_area=256)
        # Observation-only wrapper preserves all raw masks and original SAM quality fields.
        class ObservedGenerator:
            image_id=None
            def generate(self,a):
                result=generator.generate(a);np.savez_compressed(folder/(self.image_id+'_sam_raw.npz'),masks=np.array([r['segmentation'] for r in result],dtype=bool))
                atomic_json(folder/(self.image_id+'_sam_metadata.json'),[{k:v for k,v in r.items() if k!='segmentation'} for r in result])
                return result
        observed=ObservedGenerator()
        for idx,im in enumerate(ims):
            observed.image_id=im.image_id;im.load_segments(str(folder),observed)
            for j,s in enumerate(im.segments):s.stable_id=f'{prefix}-{im.image_id}-S{j:04d}'
            if not im.segments:event(out,role,'MED26-D-EMPTY_IMAGE',dict(image_id=im.image_id),'No retained segmentation region','Retain image; do not replace',status='RECORDED')
            budget(role+'_segmentation',completed=idx+1,total=len(ims))
        del generator,observed;gc.collect();torch.cuda.empty_cache();times[role+'_segmentation']=time.monotonic()-stage;stage=time.monotonic()
        segments=[s for im in ims for s in im.segments];ds=classes.ConceptDatasetClass(ims,model.default_cfg,0,True,-1,.25)
        if not segments:raise ValueError('No regions in entire split; no replacement')
        # Save effective post-erosion masks as used by __getitem__, together with raw-mask IDs.
        mapping=[];offset=0
        for im in ims:
            effective=[]
            for j,s in enumerate(im.segments):
                mask=ds[offset][1].numpy()[0].astype(bool);effective.append(mask)
                mapping.append(dict(segment_id=s.stable_id,image_id=im.image_id,source_split=role,raw_index=j,global_row=offset,raw_mask_sha256=hashlib.sha256(s.mask.tobytes()).hexdigest(),effective_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),raw_pixels=int(s.mask.sum()),effective_pixels=int(mask.sum()),empty_effective=not bool(mask.any())))
                offset+=1
            np.savez_compressed(folder/(im.image_id+'_effective_masks.npz'),masks=np.array(effective,dtype=bool))
        atomic_json(out/(role+'_raw_segments.json'),mapping)
        loader=DataLoader(ds,batch_size=8,shuffle=False,collate_fn=utils_general.custom_collate)
        acts,logits=utils_general.compute_activations(model,'global_pool',loader)
        np.savez_compressed(out/(role+'_raw_features.npz'),features=acts,logits=logits,segment_id=np.array([s.stable_id for s in segments]))
        scientific_records.numerical_checks(acts,logits,model.fc.weight.detach().cpu().numpy(),model.fc.bias.detach().cpu().numpy(),None)
        keep=np.any(acts!=0,axis=1)
        for i,(s,r) in enumerate(zip(segments,mapping)):
            r['zero_feature']=not bool(keep[i]);s.model_act=acts[i];s.model_pred=logits[i]
        atomic_json(out/(role+'_raw_segments.json'),mapping)
        # Released discovery excludes exact-zero vectors; raw masks/features/IDs remain saved.
        for im in ims:im.segments=[s for s in im.segments if np.any(s.model_act!=0)]
        if not keep.all() or any(r['empty_effective'] for r in mapping):event(out,role,'MED26-D-ZERO_EMPTY',dict(zero_features=int((~keep).sum()),empty_effective=sum(r['empty_effective'] for r in mapping),raw_regions=len(mapping)),'Released zero-feature exclusion changes discovery membership; images remain fixed','Retain all raw arrays and masks; separate zero/empty counts',status='DISCLOSED')
        raw_records[role]=mapping
        # Full-image features/predictions, with the same tensors as the frozen classifier.
        saved=[im.segments for im in ims]
        for im in ims:im.segments=[classes.SegmentClass(np.ones(im.img_numpy.shape[:2],np.float32),im)]
        full=classes.ConceptDatasetClass(ims,model.default_cfg,0,False)
        fa,fl=utils_general.compute_activations(model,'global_pool',DataLoader(full,batch_size=8,shuffle=False,collate_fn=utils_general.custom_collate))
        for im,ss in zip(ims,saved):im.segments=ss
        np.savez_compressed(out/(role+'_full_images.npz'),features=fa,logits=fl,image_id=np.array([im.image_id for im in ims]))
        full_records[role]=(fa,fl);times[role+'_features']=time.monotonic()-stage
        budget(role+'_features_complete',images=len(ims),raw_regions=len(mapping),retained_regions=int(keep.sum()))
    with parallel_backend('loky',inner_max_num_threads=1):
        process('training');stage=time.monotonic()
        exp.create_concepts('sparse_subspace_clustering',n_clusters=None,norm_acts=True,min_size=50,min_coverage=0.0,max_samples=None,outlier_percentile=1.0,folderpath=str(cache/'self_representation'))
        segs=exp.clustering.segms;idx={id(s):i for i,s in enumerate(segs)}
        sparse=exp.clustering.sparse_repr_matrix
        if not np.isfinite(sparse.data).all() or np.any(exp.clustering.outlier_mask):raise ValueError('SSC finite/q1 gate failed')
        np.savez_compressed(out/'initial_clusters.npz',labels=exp.clustering.labels,outlier_mask=exp.clustering.outlier_mask,segment_id=np.array([s.stable_id for s in segs]),row_l1=np.asarray(abs(sparse).sum(axis=1)).ravel())
        concept_index=[]
        for i,c in enumerate(exp.concepts):
            concept_index.append(dict(concept_id=f'{prefix}-C{i+1:03d}',basis_index=i,cluster_id=int(c.label),initial_members=len(c.segments),initial_sorted_segments=[s.stable_id for s in c.segments],pca_fitting_segments=[s.stable_id for s in c.get_segments(mode='diverse',num=None)]))
        atomic_json(out/'cluster_filters.json',[dict(cluster_id=int(c.label),size=len(c.segments),retained=len(c.segments)>=50,filter_reason=None if len(c.segments)>=50 else 'size<50') for c in exp.clustering.clusters])
        atomic_json(out/'concept_index.json',concept_index);times['ssc']=time.monotonic()-stage;budget('D1_SSC_COMPLETE',initial_clusters=len(exp.clustering.clusters),learned_concepts=len(exp.concepts))
        if not exp.concepts:raise ValueError('No concept survived unchanged min_size50; evidence retained, no retune')
        stage=time.monotonic();exp.compute_concept_subspace_bases(None,'ratio',False);scientific_records.save_discovery(exp,out,4)
        for r,b in zip(concept_index,exp.concept_bases):r['subspace_dimension']=len(b)
        weight=model.fc.weight.detach().cpu().numpy()[4];scores,_=exp.concept_quantification(weight)
        for i,r in enumerate(concept_index):r['global_importance']=float(scores[i]);r['global_rank']=int(np.where(np.argsort(-scores[:-1])==i)[0][0]+1)
        atomic_json(out/'concept_index.json',concept_index);times['basis_and_score']=time.monotonic()-stage
        def assign(role):
            ims=images[role];segs=[s for im in ims for s in im.segments]
            if not segs:raise ValueError('No nonzero region in split; assignments unavailable')
            acts=np.stack([s.model_act for s in segs]);sizes=[len(im.segments) for im in ims]
            similarities=exp.concept_activations(acts,sizes,norm_batch=False,n_jobs=1)
            scientific_records.save_split(exp,ims,similarities,out,role,4)
            chosen=similarities.argmax(1);mapping=[dict(segment_id=s.stable_id,image_id=s.org_img.image_id,retained_row=i,assignment=int(chosen[i]),concept_id=concept_index[int(chosen[i])]['concept_id'] if chosen[i]<len(concept_index) else prefix+'-COMPLEMENT') for i,s in enumerate(segs)]
            atomic_json(out/(role+'_assignments.json'),mapping)
            fa,fl=full_records[role];checks,samples=scientific_records.numerical_checks(fa,fl[:,4],weight,model.fc.bias.detach().cpu().numpy()[4],exp.concept_bases+[exp.compl_basis])
            atomic_json(out/(role+'_full_reconstruction.json'),checks);np.savez_compressed(out/(role+'_full_reconstruction.npz'),**samples)
            from run_humcd import get_top_concept_segms
            top_by_concept=get_top_concept_segms(similarities,sizes,10)
            examples=[];rng=np.random.RandomState(4301) # presentation only; does not alter fit or training RNG
            for ci in range(len(concept_index)+1):
                members=np.flatnonzero(chosen==ci)
                random_members=rng.choice(members,size=min(10,len(members)),replace=False) if len(members) else []
                examples.append(dict(concept_id=concept_index[ci]['concept_id'] if ci<len(concept_index) else prefix+'-COMPLEMENT',top_prototypes=[ims[ii].segments[jj].stable_id for ii,jj in top_by_concept[ci]],random_final_members=[segs[int(i)].stable_id for i in random_members],top_rule='Released get_top_concept_segms: assigned members, descending similarity, at most one region per image, up to10 images',random_rule='Independent RandomState4301, concept-index order, no replacement, at most10 assigned members'))
            atomic_json(out/(role+'_example_index.json'),examples);budget(role+'_assignments_complete',images=len(ims),retained_regions=len(segs))
        assign('training')
        if p['held_out_images']:
            process('held_out');assign('held_out')
    completeness=float(utils_mcd.calc_completeness(weight,exp.concept_bases))
    if not np.isfinite(completeness):raise ValueError('Nonfinite completeness')
    budget('D1_COMPLETE_PENDING_ACCEPTANCE',training=p['training_images'],held_out=p['held_out_images'],concepts=len(concept_index))
    return dict(status='COMPLETED_PENDING_ACCEPTANCE',training=p['training_images'],held_out=p['held_out_images'],training_lesions=400 if fit_context is None else None,held_out_lesions=61 if fit_context is None else None,initial_clusters=len(exp.clustering.clusters),concepts=len(concept_index),completeness=completeness,concept_scores=scores.tolist(),outlier_quantile=1.0,outliers=int(exp.clustering.outlier_mask.sum()),stage_seconds=times,elapsed_seconds=time.monotonic()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),classifier_sha256=checkpoint_hash,external_inference=False,clinical_utility_certified=False)
