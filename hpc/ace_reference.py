"""ACE — HU-MCD released-code reference. Slurm worker; no submission/retry.

Three stages: features -> cav -> evaluation. Scientific operators are called from
released classes/concept_explainer/benchmark_methods; instrumentation observes
real RNG choices, splits and estimators, never supplies replacement samples.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback
from types import SimpleNamespace
import warnings

import numpy as np
from hpc.evaluate_reference import checked_file, under, sha256, write_json, save_masks, predict_stream, runtime_precision, random_signature
from hpc.mcd_reference import load, digest, make_model, verify_random_reuse

ROOT=Path(__file__).resolve().parents[1]
NAME='ACE — HU-MCD released-code reference'
PROTOCOL=dict(schema='ace-humcd-released-R-v1',name=NAME,arm='R',seed=43,
    class_name='golden_retriever',model_name='resnet50',layer='global_pool',batch_size=8,
    discovery_images=50,discovery_rule='frozen Golden400 prefix50',gradient_images=50,
    gradient_rule='original Python shuffle of sorted frozen400 after control and CAVs',
    validation_images=50,random_images=2000,random_rule='label-blind RGB-compatible training pool; independent preparation seed43',
    max_shortest_side=300,slic_n_segments=[15,50,80],slic_sigma=1,slic_compactness=50,
    slic_labels='range(segm_idx.max()), intentionally omits maximum label',
    slic_min_fraction=0.001,slic_duplicate_iou=0.5,cropping_mode=1,use_masks=False,
    masking_mode=None,fill_value=0.4588,erosion_threshold=1.0,normalization=False,
    clusters=25,kmeans_random_state=43,min_size=50,min_coverage=0.5,max_samples=50,
    cav_runs=20,cav_negative_rule='first50 from full effective pool; later rounds permute overwritten50',
    cav_algorithm='SGDClassifier(alpha=.01,max_iter=1000,tol=.001), other installed defaults',
    split_test_size=.2,split_random_state=42,gradient_objective='mean CE on logits; dot(normalized CAV,gradient)<0',
    p_test='two-sided scipy.stats.ttest_rel; nominal/code-reference; affects filter and evaluation',
    p_threshold=.01,sort='descending mean TCAV, original np.argsort reverse',
    prediction_cropping_mode=0,prediction_use_masks=True,prediction_masking_mode=1,
    modes=['sdc','ssc'],contributor_rule='count > .75 * 50')
PROTOCOL_SHA256=digest(PROTOCOL)
SCIENCE_FILES=('classes.py','concept_explainer.py','benchmark_methods.py','run_ace.py',
    'utils/utils_general.py','utils/utils_ace.py','input_masking/resnet.py','input_masking/sal_layers.py')
GOLDEN_FILES=('summary.json','run_manifest.json','resolved_config.json','scientific/discovery.json')


def tuples(value):
    return tuple(tuples(v) for v in value) if isinstance(value,list) else value


def rng_snapshot(deferred_cuda=None):
    import torch
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=[n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])],
        torch_cpu=torch.get_rng_state().tolist(),
        torch_cuda=[s.cpu().tolist() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else (deferred_cuda or []),
        python_version=sys.version)


def restore_rng(state):
    import torch
    random.setstate(tuples(state['python']))
    n=state['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(torch.tensor(state['torch_cpu'],dtype=torch.uint8))
    if torch.cuda.is_initialized() and state['torch_cuda']:
        torch.cuda.set_rng_state_all([torch.tensor(s,dtype=torch.uint8) for s in state['torch_cuda']])


def finite_json(value):
    """JSON only: retain NaN/inf exactly in scientific NPZ, annotate missing JSON values."""
    if isinstance(value,np.ndarray):return finite_json(value.tolist())
    if isinstance(value,np.generic):return finite_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {str(k):finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    return value


def array_sha(value):
    a=np.ascontiguousarray(value)
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def original_id(row):return str(row['source'])


def role_overlap(left,right):
    a={original_id(r) for r in left};b={original_id(r) for r in right}
    ah={r['input_sha256'] for r in left};bh={r['input_sha256'] for r in right}
    return dict(left_unique_images=len(a),right_unique_images=len(b),source_intersection=sorted(a&b),
                count=len(a&b),content_hash_intersection=sorted(ah&bh),content_count=len(ah&bh))


def verify_sources(config):
    """Scientific identity only; generic release/scheduler checks belong to parent."""
    from hpc.workstream_runtime import verify_inputs
    root=Path(config['golden_run_dir'])
    saved=[load(checked_file(under(root,n),config['golden_files_sha256'][n])) for n in GOLDEN_FILES]
    summary,golden,old,discovery=saved
    launch=load(checked_file(Path(config['golden_launch_manifest']),config['golden_launch_sha256']))
    if (summary['status']!='PASS' or str(summary['run_id'])!='28208840' or golden['git_dirty']
        or launch['status']!='EXECUTION_COMPLETED' or summary['git_commit']!=golden['git_commit']
        or launch['actual_commit']!=golden['git_commit']
        or any(r['resolved_config_sha256']!=config['golden_files_sha256']['resolved_config.json'] for r in (golden,summary))):
        raise ValueError('Golden provenance mismatch')
    for key in ('class_name','model_name','source_dir','dataset_manifest_sha256','resnet_checkpoint_sha256','precision','batch_size','max_shortest_side'):
        if config[key]!=old[key]:raise ValueError('Changed Golden fixed setting: '+key)
    if config['layer_name']!='global_pool' or config['class_name']!='golden_retriever' or config['batch_size']!=8:
        raise ValueError('Only authorized Golden global_pool batch8 ACE-R is supported')
    if config['ace_protocol_sha256']!=PROTOCOL_SHA256 or config['random_seeds']!={'sdc':43,'ssc':43}:
        raise ValueError('Unapproved ACE/Random protocol')
    dataset=load(checked_file(Path(config['dataset_manifest']),config['dataset_manifest_sha256']))
    for split,n in (('training',400),('validation',50)):
        rows=dataset[split]
        if len(rows)!=n or [r['input_sha256'] for r in rows]!=[r['sha256'] for r in golden['input_files'][split]]:
            raise ValueError('Frozen actual Golden input order changed')
        if [r['prepared_name'] for r in rows]!=sorted(r['prepared_name'] for r in rows):raise ValueError('Unsorted Golden mapping')
    prepared=load(checked_file(Path(config['ace_input_manifest']),config['ace_input_manifest_sha256']))
    prepared_root=Path(config['ace_input_manifest']).parent
    if prepared['schema']!='ace-r-inputs-v1' or prepared['status']!='PASS' or prepared['config']['dataset_manifest_sha256']!=config['dataset_manifest_sha256']:
        raise ValueError('ACE prepared-input provenance mismatch')
    for name,item in prepared['files'].items():checked_file(under(prepared_root,name),item['sha256'],item['bytes'])
    roles={}
    for role,n in (('target400',400),('discovery50',50),('validation50',50),('random2000',2000)):
        item=prepared['roles'][role]
        rows=load(checked_file(under(prepared_root,item['path']),item['sha256']))
        if item['count']!=n or len(rows)!=n:raise ValueError('ACE role count mismatch: '+role)
        roles[role]=rows
    for role,split in (('target400','training'),('validation50','validation')):
        if any(any(a[k]!=b[k] for k in ('source','sha256','input_path','input_sha256')) for a,b in zip(roles[role],dataset[split])):
            raise ValueError('Prepared Golden role identity mismatch')
    inputs=dict(discovery=roles['discovery50'],random=roles['random2000'],
        prepared_manifest_sha256=config['ace_input_manifest_sha256'],protocol_sha256=PROTOCOL_SHA256,
        selection=prepared['selection'])
    for a,b in zip(inputs['discovery'],dataset['training'][:50]):
        if any(a[k]!=b[k] for k in ('source','sha256','input_sha256')):raise ValueError('Discovery is not fixed400 prefix50')
    pool=inputs['random']
    if len({r['source'] for r in pool})!=2000 or [r['prepared_name'] for r in pool]!=sorted(r['prepared_name'] for r in pool):
        raise ValueError('Random source IDs/order invalid')
    train_root=(Path(prepared['config']['licensed_root'])/'train').resolve()
    if any(not Path(r['source']).resolve().is_relative_to(train_root) for r in pool):raise ValueError('Random contains non-training source')
    for role in (inputs['discovery'],pool,dataset['training']):
        overlap=role_overlap(role,dataset['validation'])
        if overlap['count'] or overlap['content_count']:raise ValueError('Fixed validation overlaps ACE fitting/gradient population')
    if config['stage']=='features':
        verify_inputs(config)
        for r in inputs['discovery']+pool:
            checked_file(Path(r['source']),r['sha256']);checked_file(Path(r['input_path']),r['input_sha256'])
    science={}
    for name in SCIENCE_FILES:
        if name.startswith('input_masking/'):
            proof=launch['masking'][Path(name).name]
            if not proof['match'] or proof['installed_sha256']!=proof['expected_sha256']:raise ValueError('Mask source attestation failed')
            expected=proof['installed_sha256']
        else:expected=golden['source_sha256'][name]
        science[name]=sha256(checked_file(ROOT/name,expected))
    signature=dict(protocol_sha256=PROTOCOL_SHA256,ace_input_manifest_sha256=config['ace_input_manifest_sha256'],
        dataset_sha256=config['dataset_manifest_sha256'],classifier_sha256=config['resnet_checkpoint_sha256'],
        precision=config['precision'],model_default_cfg=discovery['model_default_cfg'],science_sha256=science,
        helper_sha256={n:sha256(ROOT/n) for n in ('hpc/evaluate_reference.py','hpc/mcd_reference.py')})
    versions={}
    for package,version in launch['versions'].items():
        if package in ('numpy','scipy','torch','torchvision','timm','scikit-learn','scikit-image','scikit-dimension'):
            versions[package]=importlib.metadata.version(package)
            if versions[package]!=version:raise ValueError('Changed library: '+package)
    return dict(dataset=dataset,inputs=inputs,model_cfg=discovery['model_default_cfg'],versions=versions,
                signature=signature,signature_sha256=digest(signature))


def stage_cache(config,prefix,source,stage):
    root=Path(config[prefix+'_dir'])
    manifest=load(checked_file(root/'ace_manifest.json',config[prefix+'_manifest_sha256']))
    if (manifest['status']!='PASS' or manifest['stage']!=stage or manifest['signature']!=source['signature']
        or manifest['implementation_sha256']!=sha256(Path(__file__))):raise ValueError('ACE stage cache identity mismatch')
    for name,r in manifest['files'].items():checked_file(under(root,name),r['sha256'],r['bytes'])
    return root,manifest


def image_from_row(row):
    import classes
    checked_file(Path(row['input_path']),row['input_sha256'])
    return classes.ImageClass(row['input_path'],300)


def unpack_masks(path):
    with np.load(path,allow_pickle=False) as z:
        shape=tuple(z['shape']);bits=z['bits']
    return np.unpackbits(bits,bitorder='big')[:int(np.prod(shape))].reshape(shape).astype(np.float32)


def slic_observed(image,folder):
    """Call original SLIC once per scale; preserve raw labels including omitted maximum."""
    import classes
    original=classes.slic;calls=[]
    def observe(*args,**kwargs):
        labels=original(*args,**kwargs)
        np.savez_compressed(folder/f'slic_{len(calls):02d}.npz',labels=labels)
        calls.append(dict(n_segments=kwargs['n_segments'],max_label=int(labels.max()),
            labels_considered=list(range(int(labels.max()))),omitted_max_pixels=int((labels==labels.max()).sum())))
        return labels
    classes.slic=observe
    try:image._segment_slic()
    finally:classes.slic=original
    write_json(folder/'slic.json',dict(calls=calls,retained_masks=len(image.segments),
        max_label_omission='preserved released-code behavior; not corrected'))
    return calls


def extract_role(model,rows,role,out,mark,note,segmented=True):
    """Streaming ImageClass/Dataset items, original global batch8 and actual layer hook."""
    import torch
    import classes
    from torch.utils.data import DataLoader,IterableDataset
    from utils import utils_general
    folder=out/role;folder.mkdir();(folder/'batches').mkdir()
    records=[];image_records=[]
    class Items(IterableDataset):
        def __iter__(self):
            for image_index,row in enumerate(rows):
                im=image_from_row(row);directory=folder/f'image_{image_index:04d}';directory.mkdir()
                if segmented:
                    slic_observed(im,directory)
                    save_masks(directory/'masks.npz',np.stack([s.mask for s in im.segments]))
                else:im.segments=[classes.SegmentClass(np.ones(im.img_numpy.shape[:2]),im)]
                count=len(im.segments);first=len(records)
                for j,seg in enumerate(im.segments):
                    records.append(dict(id=f'{role}/I{image_index:04d}/S{j:04d}',row=len(records),image_index=image_index,
                        segment_index=j,input=row,mask_shape=list(seg.mask.shape),mask_dtype=str(seg.mask.dtype),
                        mask_sha256=hashlib.sha256(seg.mask.tobytes()).hexdigest(),mask_pixels=int(seg.mask.sum())))
                image_records.append(dict(image_index=image_index,input=row,rows=[first,first+count],loaded_size=list(im.img_pil.size)))
                write_json(directory/'rows.json',records[first:first+count])
                dataset=classes.ConceptDatasetClass([im],model.default_cfg,cropping_mode=1 if segmented else 0,
                    use_masks=False,masking_mode=None,erosion_threshold=1.0)
                for j in range(len(dataset)):yield dataset[j]
    loader=DataLoader(Items(),batch_size=8,shuffle=False,collate_fn=utils_general.custom_collate)
    captured,hook=utils_general.get_activation_hook(model.get_submodule('global_pool'),None)
    arrays=[];groups=[];cursor=0
    try:
        with torch.no_grad():
            for batch,values in enumerate(loader):
                logits=model(values).detach().clone().cpu().numpy();features=captured.pop()
                if features.shape!=(len(logits),2048) or logits.shape[1]!=1000 or not np.isfinite(features).all() or not np.isfinite(logits).all():
                    raise ValueError('Invalid ACE feature/logit shape or values')
                groups.append(list(range(cursor,cursor+len(features))));cursor+=len(features)
                np.savez_compressed(folder/'batches'/f'{batch:05d}.npz',features=features,logits=logits)
                arrays.append((features,logits));mark(role=role,feature_rows=cursor,batches=batch+1)
    finally:hook.remove()
    X=np.concatenate([v[0] for v in arrays]);Y=np.concatenate([v[1] for v in arrays])
    if len(X)!=len(records):raise ValueError('Feature record ordering/count mismatch')
    zero=np.all(X==0,axis=1)
    np.savez_compressed(folder/'features.npz',features=X,logits=Y,zero_mask=zero)
    write_json(folder/'rows.json',records);write_json(folder/'images.json',image_records);write_json(folder/'batches.json',groups)
    W=model.fc.weight.detach().cpu().numpy();b=model.fc.bias.detach().cpu().numpy()
    error=float(np.abs(X@W.T+b-Y).max());np.testing.assert_allclose(X@W.T+b,Y,atol=1e-4,rtol=1e-4)
    if zero.any():note('zero_features',dict(role=role,rows=np.flatnonzero(zero).tolist()),
        'Validation retained/unassigned; discovery/random deletion follows original source. Effective random count must remain authorized2000.')
    return dict(images=len(rows),raw_rows=len(X),zero_rows=int(zero.sum()),retained_rows=int((~zero).sum()),fc_max_abs_error=error)


def load_role(root,role):
    with np.load(root/role/'features.npz',allow_pickle=False) as z:
        X,Y,zero=z['features'],z['logits'],z['zero_mask']
    records=load(root/role/'rows.json')
    if len(records)!=len(X) or not np.array_equal(zero,np.all(X==0,axis=1)):raise ValueError('ACE row identity/zero mask mismatch')
    return X,Y,zero,records


def features(config,out,source,mark,note):
    import torch
    checked_file(Path(config['resnet_checkpoint']),config['resnet_checkpoint_sha256'])
    model=make_model(config,source)
    np.savez_compressed(out/'classifier.npz',weight=model.fc.weight.detach().cpu().numpy(),bias=model.fc.bias.detach().cpu().numpy())
    result={}
    result['discovery']=extract_role(model,source['inputs']['discovery'],'discovery',out,mark,note)
    result['random']=extract_role(model,source['inputs']['random'],'random',out,mark,note,False)
    formal=rng_snapshot()
    # Validation is independent look-ahead work; its loader must not move formal fit RNG.
    result['validation']=extract_role(model,source['dataset']['validation'],'validation',out,mark,note)
    write_json(out/'validation_lookahead_rng.json',dict(before=formal,after=rng_snapshot(),restored=True))
    restore_rng(formal);write_json(out/'formal_rng.json',formal)
    if result['random']['retained_rows']!=2000:
        raise ValueError('Effective random pool differs from approved2000; preserve evidence, no replacement or silent shrink')
    return dict(roles=result,formal_rng_sha256=sha256(out/'formal_rng.json'),
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),precision=runtime_precision(torch))


def clone_roles(python_state,pool_indices,training_rows):
    clone=random.Random(0);clone.setstate(tuples(python_state))
    control=clone.sample(list(pool_indices),50);after_control=clone.getstate()
    candidates=list(range(len(training_rows)));clone.shuffle(candidates)
    return dict(control_indices=control,gradient_indices=candidates[:50],gradient_candidate_order=candidates,
        python_before=python_state,python_after_control=after_control,python_after_gradient=clone.getstate(),
        planning='cloned Python state only; no formal RNG consumption')


def objects_from_rows(X,records,keep):
    images=[];by_image={};segments=[]
    for i in np.flatnonzero(keep):
        row=records[i];key=row['image_index']
        if key not in by_image:
            by_image[key]=SimpleNamespace(filename=row['input']['input_path'],segments=[]);images.append(by_image[key])
        seg=SimpleNamespace(org_img=by_image[key],model_act=X[i],ace_row=int(i),ace_record=row)
        by_image[key].segments.append(seg);segments.append(seg)
    return images,segments


def candidate_records(explainer,records):
    candidates=[];eligible={int(c.label) for c in explainer.concepts}
    bylabel={int(c.label):c for c in explainer.clustering.clusters}
    for label in range(25):
        cluster=bylabel.get(label)
        if cluster is None:
            candidates.append(dict(id=f'ACE-R-golden_retriever-K{label:02d}',cluster_id=label,members=[],size=0,
                coverage=0,structural_pass=False,filter_reasons=['empty/unobserved cluster label']));continue
        members=[s.ace_row for s in cluster.segments];images={s.org_img.filename for s in cluster.segments}
        size=len(members);coverage=len(images)/len(explainer.class_imgs);reasons=[]
        if size<50:reasons.append('size < 50')
        if coverage<.5:reasons.append('image coverage < .5')
        candidates.append(dict(id=f'ACE-R-golden_retriever-K{label:02d}',cluster_id=label,members=members,
            member_ids=[records[i]['id'] for i in members],size=size,unique_images=len(images),coverage=coverage,
            structural_pass=label in eligible,filter_reasons=reasons,positive_rule='first50 source centroid-distance order',
            nominal_p_status='not yet computed' if label in eligible else 'not fitted: structural filter'))
    return candidates


@contextmanager
def instrument_cav(concept,pool_ids,out,mark,deferred_cuda):
    """Observe source choice, split and SGD calls; exactly one real call each."""
    import classes
    from utils import utils_ace
    choice=np.random.choice;split=utils_ace.train_test_split
    estimator=utils_ace.linear_model.SGDClassifier;fit_method=estimator.fit
    accuracy=utils_ace.metrics.accuracy_score;members=classes.ClusterClass.get_segments
    current={};history=[];negative_ids=np.asarray(pool_ids,dtype=int);positive=[]
    out.mkdir()
    def persist():
        if current:write_json(out/f"round_{current['round']:02d}.json",finite_json(current))
    def chosen(*args,**kwargs):
        nonlocal negative_ids,current
        called_by=inspect.currentframe().f_back.f_code
        if called_by is not classes.ConceptClass.train_cavs.__code__:return choice(*args,**kwargs)
        current=dict(round=len(history),status='SELECTING',rng_before_choice=rng_snapshot(deferred_cuda))
        history.append(current);persist()
        selected=choice(*args,**kwargs)
        negative_ids=negative_ids[selected]
        current.update(status='FITTING',choice_local_indices=selected.tolist(),negative_pool_rows=negative_ids.tolist(),
            positive_rows=positive.copy(),positive_role='random' if concept.label==-1 else 'discovery',rng_after_choice=rng_snapshot(deferred_cuda))
        persist();return selected
    def selected_members(obj,*args,**kwargs):
        result=members(obj,*args,**kwargs)
        if obj is concept:
            positive[:]=[s.ace_row for s in result]
            write_json(out/'positive_members.json',[s.ace_record for s in result])
        return result
    def split_observed(*arrays,**kwargs):
        result=split(*arrays,np.arange(len(arrays[0])),**kwargs)
        current.update(train_indices=result[-2].tolist(),test_indices=result[-1].tolist(),
            X_sha256=array_sha(arrays[0]),y_sha256=array_sha(arrays[1]))
        # One original RNG/split operation, with an extra passive index array.
        persist();return result[:-2]
    def fitted(model,X,y,*args,**kwargs):
        current.update(rng_before_sgd=rng_snapshot(deferred_cuda),sgd_params=finite_json(model.get_params()))
        persist()
        result=fit_method(model,X,y,*args,**kwargs)
        current.update(rng_after_sgd=rng_snapshot(deferred_cuda),sgd_n_iter=int(model.n_iter_),
            sgd_t=float(model.t_),weight_sha256=array_sha(model.coef_[0]))
        current['_model_arrays']=dict(weight=model.coef_[0].copy(),intercept=model.intercept_.copy(),classes=model.classes_.copy())
        np.savez_compressed(out/f"estimator_{current['round']:02d}.npz",**current['_model_arrays'])
        # Private arrays excluded from JSON until written below.
        return result
    def scored(y_pred,y_test,*args,**kwargs):
        value=accuracy(y_pred,y_test,*args,**kwargs)
        arrays=current.pop('_model_arrays')
        np.savez_compressed(out/f"round_{current['round']:02d}.npz",**arrays,
            test_predictions=y_pred,test_labels=y_test,train_indices=current['train_indices'],test_indices=current['test_indices'],
            choice_local_indices=current['choice_local_indices'],negative_pool_rows=current['negative_pool_rows'],
            positive_rows=positive)
        current.update(status='PASS',accuracy=float(value),file=f"round_{current['round']:02d}.npz")
        persist();mark(concept_label=int(concept.label),completed_cav_rounds=current['round']+1)
        return value
    np.random.choice=chosen;utils_ace.train_test_split=split_observed;estimator.fit=fitted
    utils_ace.metrics.accuracy_score=scored;classes.ClusterClass.get_segments=selected_members
    try:yield history
    except BaseException as exc:
        current.pop('_model_arrays',None)
        current.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());persist();raise
    finally:
        np.random.choice=choice;utils_ace.train_test_split=split;estimator.fit=fit_method
        utils_ace.metrics.accuracy_score=accuracy;classes.ClusterClass.get_segments=members
        write_json(out/'rounds.json',finite_json(history))


@contextmanager
def observe_kmeans(out,deferred_cuda=None):
    """Record defaults through the actual k_means function and its real estimator fit."""
    import classes
    from sklearn.cluster import KMeans
    function=classes.k_means;fit=KMeans.fit;record={}
    def called(*args,**kwargs):
        bound=inspect.signature(function).bind(*args,**kwargs);bound.apply_defaults()
        record.update(status='RUNNING',function='sklearn.cluster.k_means',rng_before=rng_snapshot(deferred_cuda),
            sklearn_version=importlib.metadata.version('scikit-learn'),signature=str(inspect.signature(function)),
            supplied_keys=list(kwargs),effective_arguments={k:finite_json(v) for k,v in bound.arguments.items() if k!='X'})
        write_json(out/'kmeans_execution.json',record)
        return function(*args,**kwargs)
    def fitted(model,*args,**kwargs):
        record['estimator_params']=finite_json(model.get_params())
        value=fit(model,*args,**kwargs)
        record.update(status='PASS',resolved_n_init=int(model._n_init),n_iter=int(model.n_iter_),rng_after=rng_snapshot(deferred_cuda))
        write_json(out/'kmeans_execution.json',record);return value
    classes.k_means=called;KMeans.fit=fitted
    try:yield record
    except BaseException as exc:
        record.update(status='FAILED',error=f'{type(exc).__name__}: {exc}');write_json(out/'kmeans_execution.json',record);raise
    finally:classes.k_means=function;KMeans.fit=fit


def cav(config,out,source,mark,note):
    import classes
    from concept_explainer import ConceptExplainer
    root,manifest=stage_cache(config,'feature',source,'features')
    formal=load(root/'formal_rng.json');restore_rng(formal);deferred=formal['torch_cuda']
    X,_,zero,records=load_role(root,'discovery');R,_,rz,random_records=load_role(root,'random')
    if np.count_nonzero(~rz)!=2000:raise ValueError('Effective random pool must contain2000 rows')
    images,segments=objects_from_rows(X,records,~zero)
    # Keep images with zero retained segments in the coverage denominator, as source does.
    byname={im.filename:im for im in images}
    images=[byname.get(r['input_path'],SimpleNamespace(filename=r['input_path'],segments=[])) for r in source['inputs']['discovery']]
    explainer=ConceptExplainer.__new__(ConceptExplainer);explainer.class_imgs=images
    with observe_kmeans(out,deferred):
        explainer.create_concepts(cluster_algo='kmeans',n_clusters=25,norm_acts=False,min_size=50,min_coverage=.5,max_samples=50)
    candidates=candidate_records(explainer,records);write_json(out/'all_candidates.json',candidates)
    clusters=explainer.clustering.clusters
    np.savez_compressed(out/'clusters.npz',labels=explainer.clustering.labels,retained_raw_rows=np.flatnonzero(~zero),
        cluster_ids=np.asarray([c.label for c in clusters]),centroids=np.stack([c.centroid for c in clusters]))
    _,random_segments=objects_from_rows(R,random_records,~rz)
    state=rng_snapshot(deferred);plan=clone_roles(state['python'],[s.ace_row for s in random_segments],source['dataset']['training'])
    plan['gradient_inputs']=[source['dataset']['training'][i] for i in plan['gradient_indices']]
    plan['control_inputs']=[random_records[i]['input'] for i in plan['control_indices']]
    write_json(out/'frozen_roles.json',plan);write_json(out/'rng_before_control.json',state)
    control_sample=random.sample(random_segments,50)
    if [s.ace_row for s in control_sample]!=plan['control_indices'] or tuples(random.getstate())!=tuples(plan['python_after_control']):
        raise ValueError('Formal control selection diverged from cloned frozen plan')
    control=classes.ConceptClass(-1,control_sample,False,50)
    overlap=role_overlap(plan['gradient_inputs'],source['inputs']['discovery'])
    roles={'discovery':source['inputs']['discovery'],'gradient':plan['gradient_inputs'],
        'random_pool':source['inputs']['random'],'control_positive':plan['control_inputs'],'validation':source['dataset']['validation']}
    role_matrix={f'{a}__{b}':role_overlap(roles[a],roles[b]) for i,a in enumerate(roles) for b in list(roles)[i+1:]}
    write_json(out/'role_overlaps.json',role_matrix)
    (out/'cavs').mkdir()
    for concept in explainer.concepts+[control]:
        label=int(concept.label);folder=out/'cavs'/('control' if label==-1 else f'K{label:02d}')
        with instrument_cav(concept,np.flatnonzero(~rz),folder,mark,deferred) as history:
            concept.train_cavs(R[~rz],mode='sklearn',n_runs=20,n_epochs=None,batch_size=8)
        if len(history)!=20:raise ValueError('CAV round count changed')
        positive=load(folder/'positive_members.json');posrows=[r['input'] for r in positive]
        for record in history:
            neg=[random_records[i]['input'] for i in record['negative_pool_rows']]
            record['positive_negative_image_overlap']=role_overlap(posrows,neg)
            combined=positive+[random_records[i] for i in record['negative_pool_rows']]
            record['train_test_image_overlap']=role_overlap([combined[i]['input'] for i in record['train_indices']],
                                                           [combined[i]['input'] for i in record['test_indices']])
        write_json(folder/'rounds.json',finite_json(history))
        weights=np.stack([concept.cavs[i]['weight_vector'] for i in range(20)])
        np.savez_compressed(folder/'weights.npz',weights=weights,accuracy=[concept.cavs[i]['accuracy'] for i in range(20)])
        if not np.isfinite(weights).all() or np.any(np.linalg.norm(weights,axis=1)==0):raise ValueError('Invalid CAV weight/norm; no replacement negatives')
        if tuples(random.getstate())!=tuples(plan['python_after_control']):raise ValueError('CAV/instrumentation unexpectedly consumed Python RNG')
        if any(set(r['negative_pool_rows'])!=set(history[0]['negative_pool_rows']) for r in history[1:]):
            raise ValueError('Released-code negative overwrite was not preserved')
    gradient_order=list(range(400));random.shuffle(gradient_order)
    if gradient_order!=plan['gradient_candidate_order'] or tuples(random.getstate())!=tuples(plan['python_after_gradient']):
        raise ValueError('Formal post-CAV gradient shuffle diverged from frozen plan')
    state=rng_snapshot(deferred);write_json(out/'formal_rng.json',state)
    note('nominal_p_protocol',dict(rounds=20,paired_negative_groups_shared=False),
         'Original paired nominal p<.01 will determine filtering and both evaluation directions; no independent-repeat calibration claimed')
    note('negative_overwrite_preserved',dict(concepts=len(explainer.concepts),control=True),
         'Each concept/control repeats its first negative50 identities over20 reordered fits; no swapping on conflict')
    return dict(candidate_clusters=len(clusters),structurally_eligible=len(explainer.concepts),
        eligible_cluster_ids=[int(c.label) for c in explainer.concepts],feature_manifest_sha256=config['feature_manifest_sha256'],
        frozen_roles_sha256=sha256(out/'frozen_roles.json'),gradient_discovery_overlap=overlap,
        formal_rng_sha256=sha256(out/'formal_rng.json'),cav_fits=20*(len(explainer.concepts)+1))


def gradient_cache(model,rows,out,mark):
    """Actual original backward hook and mean CE, batch8, no analytic replacement."""
    import torch
    import classes
    from torch.utils.data import DataLoader
    from utils import utils_general
    images=[image_from_row(r) for r in rows]
    for im in images:im.segments=[classes.SegmentClass(np.ones(im.img_numpy.shape[:2]),im)]
    data=classes.ConceptDatasetClass(images,model.default_cfg,0,False,None)
    loader=DataLoader(data,batch_size=8,shuffle=False,collate_fn=utils_general.custom_collate)
    gradients,handle=utils_general.get_activation_hook(model.get_submodule('global_pool'),207)
    pooled,pool_hook=utils_general.get_activation_hook(model.get_submodule('global_pool'),None)
    folder=out/'gradients';folder.mkdir();parts=[];cursor=0;groups=[]
    try:
        for batch,inputs in enumerate(loader):
            logits=model(inputs)
            model.zero_grad();targets=torch.full((logits.size(0),),207,dtype=torch.long,device=utils_general.DEVICE)
            loss=torch.nn.functional.cross_entropy(logits,targets);loss.backward()
            g=gradients.pop();h=pooled.pop();y=logits.detach().clone().cpu().numpy()
            if g.shape!=(len(y),2048) or not np.isfinite(g).all() or not np.isfinite(y).all():raise ValueError('Invalid CE gradients/logits')
            np.savez_compressed(folder/f'batch_{batch:02d}.npz',gradients=g,features=h,logits=y,loss=float(loss.detach().cpu()))
            parts.append((g,h,y));groups.append(list(range(cursor,cursor+len(g))));cursor+=len(g)
            mark(gradient_images=cursor)
    finally:handle.remove();pool_hook.remove()
    G=np.concatenate([p[0] for p in parts]);H=np.concatenate([p[1] for p in parts]);Y=np.concatenate([p[2] for p in parts])
    if len(G)!=50:raise ValueError('Expected frozen50 gradient images')
    np.savez_compressed(out/'gradients.npz',gradients=G,features=H,logits=Y)
    write_json(out/'gradient_inputs.json',dict(inputs=rows,global_batches=groups,objective=PROTOCOL['gradient_objective']))
    return G


def score_candidates(cavroot,cavmanifest,G,out,note):
    import classes
    from utils import utils_ace
    from scipy.linalg import norm
    eligible=cavmanifest['result']['eligible_cluster_ids'];concepts=[];arrays={};tests=[]
    for label in eligible+[-1]:
        name='control' if label==-1 else f'K{label:02d}'
        with np.load(cavroot/'cavs'/name/'weights.npz',allow_pickle=False) as z:weights=z['weights']
        concept=classes.ConceptClass.__new__(classes.ConceptClass);concept.label=label
        concept.cavs={i:{'weight_vector':w} for i,w in enumerate(weights)}
        concept.calc_tcav_scores(G)
        dots=np.stack([G@(w/norm(w)) for w in weights])
        scores=np.asarray(concept.tcav_scores)
        if not np.isfinite(dots).all() or not np.array_equal(scores,(dots<0).mean(1)):raise ValueError('TCAV sign replay mismatch')
        arrays[name+'_dot_products']=dots;arrays[name+'_scores']=scores
        if label==-1:control=concept
        else:concepts.append(concept)
    original_test=utils_ace.stats.ttest_rel
    captured=[]
    def observed(*args,**kwargs):
        value=original_test(*args,**kwargs)
        captured.append(dict(statistic=float(value.statistic),pvalue=float(value.pvalue),
                             df=float(value.df) if hasattr(value,'df') else len(args[0])-1))
        return value
    utils_ace.stats.ttest_rel=observed
    try:
        for c in concepts:
            c.calc_p_value(control.tcav_scores)
            result=captured[-1];difference=float(np.mean(c.tcav_scores)-np.mean(control.tcav_scores))
            tests.append(dict(cluster_id=int(c.label),**result,mean_tcav=float(np.mean(c.tcav_scores)),
                control_mean=float(np.mean(control.tcav_scores)),mean_difference=difference,
                direction='higher' if difference>0 else 'lower' if difference<0 else 'equal',
                retained=bool(c.p_value<.01),pvalue_finite=bool(np.isfinite(c.p_value)),statistic_finite=bool(np.isfinite(result['statistic'])),
                pvalue_representation=repr(c.p_value),statistic_representation=repr(result['statistic']),nominal=True,
                filter_reason='nominal p < .01' if c.p_value<.01 else 'nominal p >= .01 or NaN; original strict comparison'))
            if not np.isfinite(c.p_value):note('nonfinite_nominal_p',finite_json(tests[-1]),'Original NaN comparison excludes concept; not a replacement test')
    finally:utils_ace.stats.ttest_rel=original_test
    arrays.update(p_values=np.asarray([c.p_value for c in concepts]),cluster_ids=np.asarray(eligible),
                  test_statistics=np.asarray([r['statistic'] for r in tests]))
    np.savez_compressed(out/'tcav_statistics.npz',**arrays)
    write_json(out/'nominal_tests.json',finite_json(tests))
    return concepts,tests


def evaluation(config,out,source,mark,note):
    import classes
    import benchmark_methods as benchmark
    from concept_explainer import ConceptExplainer
    root,_=stage_cache(config,'feature',source,'features');cavroot,cm=stage_cache(config,'cav',source,'cav')
    if cm['result']['feature_manifest_sha256']!=config['feature_manifest_sha256']:raise ValueError('CAV belongs to another feature run')
    checked_file(Path(config['resnet_checkpoint']),config['resnet_checkpoint_sha256'])
    model=make_model(config,source);formal=load(cavroot/'formal_rng.json');restore_rng(formal)
    plan=load(checked_file(cavroot/'frozen_roles.json',cm['result']['frozen_roles_sha256']))
    G=gradient_cache(model,plan['gradient_inputs'],out,mark)
    concepts,tests=score_candidates(cavroot,cm,G,out,note)
    candidates=load(cavroot/'all_candidates.json');testmap={r['cluster_id']:r for r in tests}
    for c in candidates:
        if c['cluster_id'] in testmap:c['nominal_test']=testmap[c['cluster_id']]
    write_json(out/'all_candidates_with_tests.json',finite_json(candidates))
    explainer=ConceptExplainer.__new__(ConceptExplainer);explainer.concepts=concepts;explainer.max_shortest_side=300
    explainer.clustering=classes.ClusterSpaceClass.__new__(classes.ClusterSpaceClass);explainer.clustering.norm_acts=False
    with np.load(cavroot/'clusters.npz',allow_pickle=False) as z:
        explainer.clustering.clusters=[SimpleNamespace(label=int(k),centroid=v) for k,v in zip(z['cluster_ids'],z['centroids'])]
    X,_,zero,records=load_role(root,'validation');info=load(root/'validation/images.json')
    images=[];assignment=[];last_cluster={'label':None}
    original_closest=explainer.clustering.get_clostest_cluster
    def observed_closest(x):
        cluster=original_closest(x);last_cluster['label']=int(cluster.label);return cluster
    explainer.clustering.get_clostest_cluster=observed_closest
    for i,r in enumerate(source['dataset']['validation']):
        im=image_from_row(r);masks=unpack_masks(root/'validation'/f'image_{i:04d}'/'masks.npz');a,b=info[i]['rows']
        if len(masks)!=b-a:raise ValueError('Validation mask-feature count mismatch')
        im.segments=[]
        for j,(mask,x) in enumerate(zip(masks,X[a:b])):
            record=records[a+j]
            if record['image_index']!=i or record['segment_index']!=j or hashlib.sha256(mask.tobytes()).hexdigest()!=record['mask_sha256']:
                raise ValueError('Validation mask-feature identity mismatch')
            seg=classes.SegmentClass(mask,im);seg.model_act=x;im.segments.append(seg)
            last_cluster['label']=None
            ci=explainer.get_matching_concept_idx(x)
            assignment.append(dict(row=a+j,segment_id=record['id'],zero=bool(zero[a+j]),
                raw_nearest_cluster_id=last_cluster['label'],eligible_concept_index=ci,cluster_id=None if ci is None else int(concepts[ci].label),
                passes_nominal_p=False if ci is None else bool(concepts[ci].p_value<.01)))
        images.append(im)
    write_json(out/'validation_assignments.json',assignment)
    curves={};folder=out/'evaluation_data';folder.mkdir()
    for mode in ('sdc','ssc'):
        counts=[];fractions=[]
        def trajectories():
            for i,im in enumerate(images):
                masked=benchmark.iter_mask_imgs_ace(explainer,[im],mode)[0]
                masks=np.stack([s.mask for s in masked.segments]);save_masks(folder/f'ace_{mode}_masks_{i:03d}.npz',masks)
                counts.append(len(masked.segments));fractions.append([float(100*np.mean(s.mask)) for s in masked.segments])
                write_json(folder/f'ace_{mode}_counts_partial.json',counts)
                yield i,masked
        with (folder/f'ace_{mode}_partial_logits.f32').open('xb') as partial,(folder/f'ace_{mode}_partial_batches.jsonl').open('x') as batchfile:
            def checkpoint(values,ids):
                values.tofile(partial);partial.flush();batchfile.write(json.dumps(ids)+'\n');batchfile.flush()
            logits,batches=predict_stream(model,trajectories(),8,lambda n,b:mark(mode=mode,predicted_states=n,batches=b),on_batch=checkpoint)
        offsets=np.r_[0,np.cumsum(counts)];correct=logits.argmax(1)==207
        grouped=[correct[a:b].tolist() for a,b in zip(offsets[:-1],offsets[1:])]
        avg,std=benchmark.calc_avg_and_std(grouped,50);pixels,pstd=benchmark.calc_avg_and_std(fractions,50)
        contributors=[sum(n>j for n in counts) for j in range(max(counts))];included=[j for j,n in enumerate(contributors) if n>37.5]
        curves['ace_'+mode]=dict(name=NAME,mode=mode,state_counts=counts,step_indices=included,
            all_step_contributors=contributors,contributors=[contributors[i] for i in included],accuracy_mean=avg.tolist(),
            accuracy_std=std.tolist(),visible_pixel_percent_mean=pixels.tolist(),visible_pixel_percent_std=pstd.tolist(),
            upstream_plot_x_deleted_fraction=(1-.01*pixels).tolist(),total_states=len(logits))
        np.savez_compressed(folder/f'ace_{mode}_predictions.npz',logits=logits,correct=correct,offsets=offsets,visible_pixel_percent=np.concatenate(fractions))
        write_json(folder/f'ace_{mode}_states.json',dict(inputs=source['dataset']['validation'],counts=counts,global_batches=batches,
            concept_order=[int(concepts[i].label) for i in np.argsort([np.mean(c.tcav_scores) for c in concepts])[::-1]],
            nominal_filter=[int(c.label) for c in concepts if c.p_value<.01]))
        write_json(out/'evaluation_curves.json',curves)
    reuse=verify_random_reuse(config,source);write_json(out/'random_reuse.json',reuse)
    write_json(out/'formal_rng.json',rng_snapshot(formal['torch_cuda']))
    return dict(name=NAME,candidate_clusters=len(candidates),structurally_eligible=len(concepts),
        nominal_retained=sum(c.p_value<.01 for c in concepts),curves=curves,random_reuse=reuse,
        caveat='Nominal paired p values determine filtering and both curves; not a calibrated independent-repeat or human test',
        feature_manifest_sha256=config['feature_manifest_sha256'],cav_manifest_sha256=config['cav_manifest_sha256'])


def run(config,output):
    """Parent owns immutable release, submission and collection; worker owns science."""
    import re
    if config['stage'] not in ('features','cav','evaluation'):raise ValueError('Unknown ACE stage')
    if not os.environ.get('SLURM_JOB_ID','').isdigit() or not re.fullmatch(r'bun\d{3}',socket.gethostname().split('.')[0]):
        raise RuntimeError('Slurm compute allocation required')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if list(out.glob('ace_*')):raise FileExistsError('ACE output already used; no retry in place')
    preexisting={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}
    state=dict(status='RUNNING',stage=config['stage'],name=NAME,job_id=os.environ['SLURM_JOB_ID'],
        execution_commit=config['execution_commit'],implementation_sha256=sha256(Path(__file__)),protocol=PROTOCOL,
        config_sha256=digest(config));anomalies=[];started=time.monotonic()
    write_json(out/'ace_config.json',config)
    def mark(**kw):write_json(out/'ace_progress.json',dict(stage=config['stage'],status=state['status'],elapsed=time.monotonic()-started,**kw))
    def note(kind,evidence,impact):
        anomalies.append(dict(kind=kind,stage=config['stage'],evidence=finite_json(evidence),impact=impact,status='RECORDED'))
        write_json(out/'ace_anomalies.json',anomalies)
    try:
        import torch
        import classes
        from concept_explainer import ConceptExplainer
        from threadpoolctl import threadpool_limits
        source=verify_sources(config);state.update(signature=source['signature'],signature_sha256=source['signature_sha256'],versions=source['versions'])
        write_json(out/'ace_inputs.json',source['inputs']);write_json(out/'ace_manifest.json',state)
        if config['stage']=='features':
            random.seed(43);np.random.seed(43);torch.manual_seed(43)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(43)
        state['entry_rng']=rng_snapshot()
        original_warning=warnings.showwarning
        def observed(message,category,filename,lineno,file=None,line=None):
            note('python_warning',dict(message=str(message),category=category.__name__,file=filename,line=lineno),'Retained; assess against scientific stage')
            original_warning(message,category,filename,lineno,file,line)
        with warnings.catch_warnings(),threadpool_limits(limits=int(os.environ['SLURM_CPUS_PER_TASK'])):
            warnings.simplefilter('always');warnings.showwarning=observed
            result=globals()[config['stage']](config,out,source,mark,note)
        state.update(status='PASS',result=result)
        write_json(out/'ace_result.json',finite_json(result));return finite_json(result)
    except BaseException as exc:
        state.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        note('stage_failed',state['traceback'],'Affected task incomplete; preserve evidence and do not retry automatically');raise
    finally:
        state['elapsed_seconds']=time.monotonic()-started;mark();write_json(out/'ace_anomalies.json',anomalies)
        state['files']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha256(p))
            for p in sorted(out.rglob('*')) if p.is_file() and p.name!='ace_manifest.json' and str(p.relative_to(out)) not in preexisting}
        write_json(out/'ace_manifest.json',finite_json(state))
