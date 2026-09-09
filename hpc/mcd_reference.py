"""Golden MCD: immutable GPU features -> CPU fit -> GPU paired evaluation.

Shared-worker entry point: run(config, output). No scheduler, network, pretrained
model download, SAM, resubmission or in-place resume. See MCD_EXECUTION_PROTOCOL.md.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import time
import traceback
from types import SimpleNamespace
import warnings

import numpy as np
from hpc.evaluate_reference import (checked_file, predict_stream, runtime_precision,
                                    save_masks, sha256, under, write_json, random_signature)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_FILES = ('summary.json', 'run_manifest.json', 'resolved_config.json',
                'scientific/discovery.json')
SCIENCE_FILES = ('classes.py', 'concept_explainer.py', 'run_mcd.py',
                 'benchmark_methods.py', 'utils/utils_general.py', 'utils/utils_mcd.py',
                 'input_masking/resnet.py', 'input_masking/sal_layers.py')
PROTOCOL = dict(name='golden_mcd_benchmark_v1', layer='layer4', feature_shape=[2048, 7, 7],
    train_images=400, validation_images=50, max_shortest_side=300, batch_size=8,
    feature_cropping_mode=0, feature_use_masks=False, feature_masking_mode=None,
    erosion_threshold=1.0, spatial_training_l2=True, clustering_l2=True,
    q=0.75, search=list(range(3, 20)), stop_strictly_above=0.5, min_size=0,
    min_coverage=0.0, max_samples=None, dimensionality='FO', alphaFO=0.05,
    kmeans_random_state=43, pca_random_state=42, stage_rng_seed=None, spectral_v0='upstream default (not supplied)',
    prediction_masking_mode=1, prediction_use_masks=True,
    modes=['sdc', 'ssc'], contributor_rule='count > 0.75 * 50',
    exhausted_search='retain k=19 and report threshold_met=false, as upstream')


def load(path):
    return json.loads(Path(path).read_text())


def digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def rng_state():
    s = np.random.get_state()
    return dict(numpy=[s[0], s[1].tolist(), int(s[2]), int(s[3]), float(s[4])],
                python=random.getstate())


def verify_common(config):
    """Pin the original 400/50 manifest/order; method-specific features are new."""
    from hpc.workstream_runtime import verify_inputs
    root = Path(config['golden_run_dir'])
    launch = load(checked_file(Path(config['golden_launch_manifest']), config['golden_launch_sha256']))
    for name in GOLDEN_FILES:
        checked_file(under(root, name), config['golden_files_sha256'][name])
    summary, source, old, discovery = [load(root/name) for name in GOLDEN_FILES]
    if (summary['status'] != 'PASS' or str(summary['run_id']) != '28208840'
            or source['git_dirty'] or source['git_commit'] != summary['git_commit']
            or launch['status'] != 'EXECUTION_COMPLETED' or launch['actual_commit'] != source['git_commit']
            or source['resolved_config_sha256'] != sha256(root/'resolved_config.json')
            or summary['resolved_config_sha256'] != sha256(root/'resolved_config.json')):
        raise ValueError('Golden reference provenance mismatch')
    for name in ('class_name', 'source_dir', 'dataset_manifest_sha256', 'precision',
                 'resnet_checkpoint_sha256', 'batch_size', 'model_name', 'max_shortest_side'):
        if config[name] != old[name]:
            raise ValueError('Changed fixed Golden setting: '+name)
    if config['class_name'] != 'golden_retriever' or config['batch_size'] != 8:
        raise ValueError('Only the authorized Golden batch-8 protocol is implemented')
    if config['random_seeds'] != {'sdc':43,'ssc':43} or config['layer_name'] != 'layer4':
        raise ValueError('MCD requires layer4 and fixed Random seeds')
    if source['precision'] != config['precision']:
        raise ValueError('Source runtime precision mismatch')
    if config.get('protocol', PROTOCOL) != PROTOCOL:
        raise ValueError('Scientific protocol override requires separate review')
    verify_inputs(config)  # Slurm-only caller; verifies original and actual file hashes.
    dataset = load(config['dataset_manifest'])
    for split in ('training', 'validation'):
        rows = dataset[split]
        if [r['input_sha256'] for r in rows] != [r['sha256'] for r in source['input_files'][split]]:
            raise ValueError('Actual input order differs from frozen Golden run')
        if [r['prepared_name'] for r in rows] != sorted(r['prepared_name'] for r in rows):
            raise ValueError('Fixed order is not upstream benchmark sorted input order')
    hashes = {}
    for name in SCIENCE_FILES:
        if name.startswith('input_masking/'):
            attestation=launch['masking'][Path(name).name]
            if not attestation['match'] or attestation['expected_sha256'] != attestation['installed_sha256']:
                raise ValueError('Golden installed masking code was not attested')
            expected=attestation['installed_sha256']
        else:expected=source['source_sha256'][name]
        hashes[name] = sha256(checked_file(ROOT/name, expected))
    checked_file(Path(config['resnet_checkpoint']), config['resnet_checkpoint_sha256'])
    signature = dict(protocol=PROTOCOL, dataset_sha256=config['dataset_manifest_sha256'],
        input_hashes={s:[r['input_sha256'] for r in dataset[s]] for s in ('training','validation')},
        classifier_sha256=config['resnet_checkpoint_sha256'], precision=config['precision'],
        science_sha256=hashes, model_default_cfg=discovery['model_default_cfg'])
    return dict(dataset=dataset, golden=source, golden_versions=launch['versions'], signature=signature,
                signature_sha256=digest(signature), model_cfg=discovery['model_default_cfg'])


def verified_stage(config, prefix, source, expected_stage):
    root = Path(config[prefix+'_dir'])
    manifest = load(checked_file(root/'mcd_manifest.json', config[prefix+'_manifest_sha256']))
    if (manifest['status'] != 'PASS' or manifest['stage'] != expected_stage
            or manifest['signature_sha256'] != source['signature_sha256']
            or manifest['signature'] != source['signature']
            or manifest['implementation_sha256'] != sha256(Path(__file__))):
        raise ValueError('Incompatible/incomplete stage cache: '+prefix)
    for rel, item in manifest['files'].items():
        checked_file(under(root,rel), item['sha256'], item['bytes'])
    return root, manifest


def make_model(config, source):
    import torch
    import timm
    from utils import utils_general
    if importlib.metadata.version('timm') != '0.6.13':
        raise ValueError('Expected pinned timm 0.6.13')
    for name in ('resnet.py','sal_layers.py'):
        checked_file(Path(timm.__file__).parent/'models'/name,
                     source['signature']['science_sha256']['input_masking/'+name])
    if runtime_precision(torch) != config['precision']:
        raise ValueError('Runtime precision changed; flags must be configured by launcher')
    if not config['device'].startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('GPU stage requires its allocated CUDA device')
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    utils_general.DEVICE = config['device']
    torch.cuda.set_device(config['device'])
    model = timm.create_model('resnet50', pretrained=False)
    model.load_state_dict(torch.load(config['resnet_checkpoint'], map_location='cpu'), strict=True)
    model.eval().to(config['device'])
    if json.loads(json.dumps(model.default_cfg)) != source['model_cfg']:
        raise ValueError('Model input preprocessing identity mismatch')
    return model


def load_images(rows):
    import classes
    # No loader try/skip, no shuffle, no substitutions. Hashes checked beforehand.
    images = [classes.ImageClass(r['input_path'], 300) for r in rows]
    for im in images:
        im.segments = [classes.SegmentClass(np.ones(im.img_numpy.shape[:2]), im)]
    return images


def fc_check(maps, logits, weight, bias):
    if maps.ndim != 4 or tuple(maps.shape[1:]) != (2048,7,7):
        raise ValueError('Unexpected layer4 shape; rectangular-map source indexing needs review')
    if not np.isfinite(maps).all() or not np.isfinite(logits).all():
        raise ValueError('Nonfinite full-image features/logits')
    pooled = maps.mean(axis=(2,3))
    reconstructed = pooled@weight.T+bias
    np.testing.assert_allclose(reconstructed, logits, rtol=1e-4, atol=1e-4)
    return dict(rows=len(maps), logits_per_image=1000, rtol=1e-4, atol=1e-4,
                max_abs_error=float(np.max(np.abs(reconstructed-logits))))


def features(config, out, source, mark, note):
    import torch
    import classes
    from torch.utils.data import DataLoader
    from utils import utils_general
    model = make_model(config, source)
    W = model.fc.weight.detach().cpu().numpy(); bias = model.fc.bias.detach().cpu().numpy()
    np.savez_compressed(out/'classifier.npz', weight=W, bias=bias)
    result = {}
    for split in ('training','validation'):
        rows = source['dataset'][split]; images = load_images(rows)
        data = classes.ConceptDatasetClass(images, model.default_cfg, cropping_mode=0,
            use_masks=False, masking_mode=None, erosion_threshold=1.0)
        loader = DataLoader(data, batch_size=8, shuffle=False, collate_fn=utils_general.custom_collate)
        partial = out/(split+'_batches'); partial.mkdir()
        # Same forward batch/order/hook as compute_activations; checkpoint each batch.
        acts, handle = utils_general.get_activation_hook(model.get_submodule('layer4'), None)
        values=[]; groups=[]
        try:
            with torch.no_grad():
                for batch, inputs in enumerate(loader):
                    logits=model(inputs).detach().clone().cpu().numpy()
                    fmap=acts.pop()
                    first=batch*8
                    np.savez_compressed(partial/f'{batch:04d}.npz', feature_maps=fmap, logits=logits)
                    groups.append(list(range(first,first+len(logits))))
                    values.append((fmap,logits))
                    mark(split=split, completed_images=first+len(logits), batches=batch+1)
        finally:
            handle.remove()
        maps=np.concatenate([v[0] for v in values]); logits=np.concatenate([v[1] for v in values])
        if len(maps)!=len(rows) or logits.shape!=(len(rows),1000):
            raise ValueError('Extraction image/logit count mismatch')
        check=fc_check(maps,logits,W,bias)
        np.savez_compressed(out/(split+'_features.npz'),feature_maps=maps,logits=logits)
        write_json(out/(split+'_inputs.json'),dict(inputs=rows,global_batches=groups,
            loaded_image_sizes=[list(im.img_pil.size) for im in images],
            spatial_order='image, row, column; feature_maps N,C,H,W -> N,H,W,C',
            zero_spatial_rows=int(np.all(maps==0,axis=1).sum())))
        result[split]=check
    return dict(features=result, model_default_cfg=source['model_cfg'],
        gpu_name=torch.cuda.get_device_name(),precision=runtime_precision(torch),
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())


def load_features(root, split):
    with np.load(root/(split+'_features.npz'),allow_pickle=False) as z:
        return z['feature_maps'],z['logits']


def basis_gate(bases):
    from scipy.linalg import svdvals
    if not bases or any(b.ndim!=2 or not len(b) or not np.isfinite(b).all() for b in bases):
        raise ValueError('Empty or nonfinite concept/complement basis')
    C=np.concatenate(bases)
    s=svdvals(C); cutoff=max(C.shape)*np.finfo(C.dtype).eps*s[0]
    rank=int((s>cutoff).sum())
    if C.shape != (2048,2048) or rank != 2048:
        raise ValueError('Non-square/rank-deficient full basis; upstream complement boundary requires review')
    return C,dict(shape=list(C.shape),rank=rank,rank_cutoff=float(cutoff),
                  condition_2=float(s[0]/s[-1]),basis_shapes=[list(b.shape) for b in bases])


def project_spatial_image(raw, bases, factor, target_weight):
    """Block LU application of the original oblique formula, retaining f32 division."""
    from scipy.linalg import lu_solve,norm
    divisor=norm(raw,axis=1).max()
    if divisor==0:raise ValueError('All-zero image would trigger upstream 0/0 normalization')
    coefficients=lu_solve(factor,(raw/divisor).T).T
    raw_coefficients=lu_solve(factor,raw.T).T
    scores=np.empty((len(raw),len(bases)));relevance=np.empty_like(scores)
    recon=np.zeros_like(raw,dtype=np.float64);offset=0
    for j,basis in enumerate(bases):
        end=offset+len(basis)
        component=coefficients[:,offset:end]@basis
        raw_component=raw_coefficients[:,offset:end]@basis
        scores[:,j]=norm(component,axis=1)
        relevance[:,j]=raw_component@target_weight
        recon+=raw_component;offset=end
    return scores,relevance,recon


def cached_spatial_replay(maps, logits, bases, W, bias, target, out, split, mark):
    """Original per-image normalization BEFORE solve. No fit or forward pass."""
    from scipy.linalg import lu_factor,norm
    from utils import utils_mcd
    C,gate=basis_gate(bases); factor=lu_factor(C.T)
    x=maps.transpose(0,2,3,1).reshape(-1,2048); n=len(x)
    scores=np.empty((n,len(bases))); relevance=np.empty_like(scores)
    feat_err=np.empty(n); reconstructed_logits=[]
    controls=[]
    # Small image chunks bound memory; retain original float32 division.
    for i in range(len(maps)):
        a,b=i*49,(i+1)*49;raw=x[a:b]
        scores[a:b],relevance[a:b],recon=project_spatial_image(raw,bases,factor,W[target])
        feat_err[a:b]=norm(recon-raw,axis=1)/np.maximum(norm(raw,axis=1),1e-12)
        reconstructed_logits.append(recon.mean(0)@W.T+bias)
        if i==0:
            literal=utils_mcd.batch_concept_activations(raw,bases,True)
            np.testing.assert_allclose(scores[a:b],literal,rtol=1e-5,atol=1e-5)
            controls.append(dict(image_index=0,rows=49,max_abs_score_error=float(np.abs(scores[a:b]-literal).max())))
        if i%25==0:mark(split=split,reconstructed_images=i+1)
    reconstructed_logits=np.asarray(reconstructed_logits)
    if not all(np.isfinite(v).all() for v in (scores,relevance,feat_err)) or feat_err.max()>1e-4:
        raise ValueError('Nonfinite or invalid cached spatial decomposition')
    np.testing.assert_allclose(reconstructed_logits,logits,rtol=1e-4,atol=1e-4)
    target_logits=relevance.reshape(len(maps),49,-1).sum(2).mean(1)+bias[target]
    np.testing.assert_allclose(target_logits,logits[:,target],rtol=1e-4,atol=1e-4)
    np.savez_compressed(out/(split+'_spatial.npz'),features=x,scores=scores,
        local_relevance=relevance,assignments=scores.argmax(1),feature_reconstruction_relative_error=feat_err,
        reconstructed_image_logits=reconstructed_logits)
    return dict(basis=gate,spatial_rows=n,images=len(maps),normalization='per image, norm_batch=True',
        max_feature_relative_error=float(feat_err.max()),all_1000_image_logit_max_abs_error=float(np.abs(reconstructed_logits-logits).max()),
        feature_rtol=1e-4,logit_rtol=1e-4,logit_atol=1e-4,literal_controls=controls)


def fit(config,out,source,mark,note):
    from scipy import sparse
    from skdim.id import lPCA
    import torch
    import classes
    from concept_explainer import ConceptExplainer
    from run_mcd import prepare_imgs_for_mcd
    from utils import utils_mcd,utils_general
    root,feature_manifest=verified_stage(config,'feature',source,'features')
    if lPCA(ver='FO').alphaFO != 0.05:raise ValueError('Installed FO default changed')
    maps,logits=load_features(root,'training')
    # Original rearrangement only needs filename/segments, not decoded images or GPU.
    images=[]
    for i,row in enumerate(source['dataset']['training']):
        im=SimpleNamespace(filename=row['input_path'],segments=[])
        segment=classes.SegmentClass(np.ones((7,7)),im);segment.model_act=maps[i]
        im.segments=[segment];images.append(im)
    prepare_imgs_for_mcd(images)
    keep=np.any(maps.transpose(0,2,3,1).reshape(-1,2048)!=0,axis=1)
    if (~keep).any():note('zero_spatial_training_rows',int((~keep).sum()),'Excluded from clustering by original prepare_imgs_for_mcd; retained in raw reconstruction')
    spatial_ids=np.flatnonzero(keep)
    X=np.stack([s.model_act for im in images for s in im.segments])
    if len(X)!=len(spatial_ids) or not np.isfinite(X).all():raise ValueError('Spatial row mapping mismatch')
    row_by_object={id(s):j for j,s in enumerate(s for im in images for s in im.segments)}
    np.savez_compressed(out/'clustering_input.npz',features=X,raw_spatial_indices=spatial_ids,zero_mask=~keep)
    write_json(out/'training_mapping.json',[dict(row=i,raw_spatial_index=int(raw),image_index=int(raw//49),
        feature_y=int(raw%49//7),feature_x=int(raw%7),input_path=source['dataset']['training'][raw//49]['input_path']) for i,raw in enumerate(spatial_ids)])
    with np.load(root/'classifier.npz',allow_pickle=False) as z:W,bias=z['weight'],z['bias']
    explainer=ConceptExplainer.__new__(ConceptExplainer)
    explainer.class_imgs=images;explainer.target_class='golden_retriever';explainer.max_shortest_side=300
    explainer.model=SimpleNamespace(default_cfg=source['model_cfg'],fc=SimpleNamespace(weight=torch.from_numpy(W),bias=torch.from_numpy(bias)))
    target=utils_general.get_imagenet_class_index('golden_retriever')
    sscdir=out/'ssc';sscdir.mkdir();search=out/'search';search.mkdir()
    trace=[];calls=[]
    original_ssc=utils_mcd.compute_sparse_repr_matrix
    original_outlier=utils_mcd.get_outlier_mask
    def capture_outlier(matrix,percentile):
        mask=original_outlier(matrix,percentile)
        np.save(sscdir/'first_pass_outlier_mask.npy',mask)
        write_json(sscdir/'outlier_rule.json',dict(q=percentile,comparison='row L1 > np.quantile(row L1,q)',rows=len(mask),outliers=int(mask.sum())))
        return mask
    utils_mcd.get_outlier_mask=capture_outlier
    def capture_ssc(*args,**kwargs):
        acts=kwargs.get('acts',args[0] if args else None);idx=len(calls)+1
        record=dict(pass_number=idx,rows=len(acts),status='RUNNING',started=time.time())
        calls.append(record);write_json(out/'ssc_trace.json',calls);mark(ssc_pass=idx,rows=len(acts))
        try:
            matrix=original_ssc(*args,**kwargs) # q=.75 caller makes exactly two invocations
            sparse.save_npz(sscdir/f'pass_{idx}.npz',matrix)
            record.update(status='PASS',shape=list(matrix.shape),nnz=int(matrix.nnz),elapsed_seconds=time.time()-record['started'])
            write_json(out/'ssc_trace.json',calls)
            return matrix
        except BaseException as exc:
            record.update(status='FAILED',error=str(exc));write_json(out/'ssc_trace.json',calls);raise
    utils_mcd.compute_sparse_repr_matrix=capture_ssc
    try:
        for k in PROTOCOL['search']:
            attempt=search/f'k{k:02d}';attempt.mkdir();fo=[]
            record=dict(k=k,status='RUNNING',rng_before=rng_state());trace.append(record)
            write_json(out/'search_trace.json',trace);mark(search_k=k)
            original_dim=classes.ConceptClass._estimate_dim
            original_members=classes.ClusterClass.get_segments
            members=[]
            def capture_members(concept,*args,**kwargs):
                selected=original_members(concept,*args,**kwargs)
                if isinstance(concept,classes.ConceptClass):
                    members.append(dict(cluster_id=int(concept.label),mode=kwargs.get('mode',args[0] if args else 'max'),
                        clustering_rows=[row_by_object[id(s)] for s in selected]))
                    write_json(attempt/'pca_member_order.json',members)
                return selected
            classes.ClusterClass.get_segments=capture_members
            def capture_dim(concept,acts,mode):
                value=original_dim(concept,acts,mode)
                fo.append(dict(cluster_id=int(concept.label),estimated_dimension=int(value),samples=len(acts),mode=mode,alphaFO=0.05))
                write_json(attempt/'FO.json',fo);return value
            classes.ConceptClass._estimate_dim=capture_dim
            try:
                explainer.create_concepts(cluster_algo='sparse_subspace_clustering',n_clusters=k,
                    norm_acts=True,min_size=0,min_coverage=0.0,max_samples=None,
                    outlier_percentile=0.75,folderpath=str(sscdir))
                np.savez_compressed(attempt/'cluster_membership.npz',labels=explainer.clustering.labels,
                                    outlier_mask=explainer.clustering.outlier_mask)
                explainer.compute_concept_subspace_bases(subspace_dimensionality=None,est_mode='FO',compt_princ_angl=False)
                values={f'concept_basis_{i:03d}':b for i,b in enumerate(explainer.concept_bases)}
                values.update(complement_basis=explainer.compl_basis,cluster_labels=explainer.clustering.labels,
                    outlier_mask=explainer.clustering.outlier_mask,retained_cluster_labels=np.array([c.label for c in explainer.concepts]),fc_weight=W,fc_bias=bias)
                np.savez_compressed(attempt/'discovery.npz',**values)
                _,gate=basis_gate(explainer.concept_bases+[explainer.compl_basis])
                completeness=float(utils_mcd.calc_completeness(W[target],explainer.concept_bases))
                if not np.isfinite(completeness):raise ValueError('Nonfinite completeness')
                record.update(status='PASS',completeness=completeness,threshold_met=completeness>0.5,
                    learned_concepts=len(explainer.concepts),dimensions=[len(b) for b in explainer.concept_bases],
                    outliers=int(explainer.clustering.outlier_mask.sum()),basis_check=gate,rng_after=rng_state())
                write_json(out/'search_trace.json',trace)
            except BaseException as exc:
                record.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
                write_json(out/'search_trace.json',trace);raise
            finally:
                classes.ConceptClass._estimate_dim=original_dim
                classes.ClusterClass.get_segments=original_members
            if record['threshold_met']:break
    finally:
        utils_mcd.compute_sparse_repr_matrix=original_ssc
        utils_mcd.get_outlier_mask=original_outlier
    if len(calls)!=2:raise ValueError('Expected exactly two original SSC computations in fresh fit')
    if not record['threshold_met']:
        note('search_threshold_not_met','k=19 retained as in upstream; not a >.5 discovery', 'scientific limitation')
    # Preserve the selected fit unchanged, along with every attempted k.
    np.savez_compressed(out/'discovery.npz',**values)
    importance,_=explainer.concept_quantification(W[target])
    if not np.isfinite(importance).all():raise ValueError('Nonfinite importance')
    checks={}
    for split in ('training','validation'):
        fmap,y=load_features(root,split)
        checks[split]=cached_spatial_replay(fmap,y,explainer.concept_bases+[explainer.compl_basis],W,bias,target,out,split,mark)
        write_json(out/(split+'_mapping.json') if split=='validation' else out/'training_raw_mapping.json',
            [dict(raw_spatial_index=i,image_index=i//49,feature_y=i%49//7,feature_x=i%7,
                input_path=source['dataset'][split][i//49]['input_path']) for i in range(len(fmap)*49)])
    return dict(selected_k=k,threshold_met=record['threshold_met'],completeness=record['completeness'],
        concepts=len(explainer.concepts),global_importance=importance.tolist(),checks=checks,
        feature_manifest_sha256=config['feature_manifest_sha256'],ssc_passes=calls,
        learned_rank_order=np.argsort(importance[:-1])[::-1].tolist())


class MCDCachedScores:
    """Original iter_mask_imgs_mcd adapter; validates raw row order and 49-cell grouping."""
    def __init__(self,model,bases,raw,scores,relevance):
        self.model=model;self.concept_bases=bases;self.max_shortest_side=300
        self.raw=raw;self.scores=scores;self.relevance=relevance

    def concept_activations(self,acts,batch_sizes,norm_batch):
        if batch_sizes!=49 or norm_batch is not True or not np.array_equal(acts,self.raw):
            raise ValueError('MCD cached score normalization/order mismatch')
        return self.scores

    def concept_relevances(self,acts):
        if not np.array_equal(acts,self.raw):raise ValueError('MCD relevance row mismatch')
        return self.relevance


def trajectory_details(adapter,image):
    """Observe the SAME resize/argmax/mean/order; never replace original trajectory."""
    from skimage.transform import resize
    k=len(adapter.concept_bases);shape=adapter.model.default_cfg['input_size'][1:]
    scores=resize(adapter.scores.reshape(1,7,7,k+1).transpose(0,3,1,2),(1,k+1,*shape))[0]
    rel=resize(adapter.relevance.reshape(1,7,7,k+1).transpose(0,3,1,2),(1,k+1,*shape))[0]
    assignment=scores.argmax(0)
    # None is JSON representation ONLY; original uses NaN for absent concepts.
    present=[bool(np.any(assignment==j)) for j in range(k)]
    importance=np.array([rel[j,assignment==j].mean() if present[j] else np.nan for j in range(k)])
    return assignment,dict(importance=[float(x) if np.isfinite(x) else None for x in importance],
        absent_concepts=[j for j,p in enumerate(present) if not p],
        concept_order=np.argsort(importance)[::-1].tolist(),
        importance_definition='mean resized local relevance over pixel argmax members; absent=NaN in original, null in JSON',
        assignment_counts=np.bincount(assignment.ravel(),minlength=k+1).tolist())


def verify_random_reuse(config,source):
    """Require a complete signed contract; mismatch never triggers new Random predictions."""
    spec=config.get('random_reuse')
    if not spec:return dict(status='NOT_REQUESTED',new_random_predictions=0)
    folder=Path(spec['directory'])
    contract=load(checked_file(Path(spec['contract_path']),spec['contract_sha256']))
    expected=random_signature(config,source['model_cfg'],source['dataset']['validation'],source['signature']['science_sha256'])
    if contract['signature']!=expected:raise ValueError('Random reuse signature differs; no automatic recomputation')
    # Contract is an explicit, hash-bound provenance audit prepared by coordinator,
    # not an inferred trust in a filename or a convenient matching curve.
    if contract.get('status')!='PASS' or str(contract.get('source_job'))!='28214892':
        raise ValueError('Random producer provenance not accepted')
    for rel,item in contract['files'].items():checked_file(under(folder,rel),item['sha256'],item['bytes'])
    needed={'evaluation_sources.json','evaluation_config.json','evaluation_curves.json'}
    needed.update(f'evaluation_data/rdm_{m}_{suffix}' for m in ('sdc','ssc') for suffix in ('predictions.npz','states.json'))
    needed.update(f'evaluation_data/rdm_{m}_masks_{i:03d}.npz' for m in ('sdc','ssc') for i in range(50))
    if not needed.issubset(contract['files']):raise ValueError('Random reuse omits required source/mask/prediction evidence')
    src=load(folder/'evaluation_sources.json');cfg=load(folder/'evaluation_config.json')
    if (src['inputs']!=source['dataset']['validation'] or src['classifier_sha256']!=config['resnet_checkpoint_sha256']
            or src['precision']!=config['precision'] or cfg['batch_size']!=8 or cfg['random_seeds']!={'sdc':43,'ssc':43}):
        raise ValueError('Random contract conflicts with producer records')
    curves=load(folder/'evaluation_curves.json')
    return dict(status='REUSED',source_job='28214892',contract_sha256=spec['contract_sha256'],
                directory=str(folder),new_random_predictions=0,curves={f'rdm_{m}':curves[f'rdm_{m}'] for m in ('sdc','ssc')})


def evaluation(config,out,source,mark,note):
    import torch
    import benchmark_methods as benchmark
    feature_root,_=verified_stage(config,'feature',source,'features')
    fit_root,fit_manifest=verified_stage(config,'fit',source,'fit')
    if fit_manifest['result']['feature_manifest_sha256']!=config['feature_manifest_sha256']:
        raise ValueError('Fit belongs to another feature extraction')
    model=make_model(config,source)
    with np.load(fit_root/'discovery.npz',allow_pickle=False) as z:
        bases=[z[k] for k in sorted(z.files) if k.startswith('concept_basis_')]
        if not np.array_equal(model.fc.weight.detach().cpu().numpy(),z['fc_weight']) or not np.array_equal(model.fc.bias.detach().cpu().numpy(),z['fc_bias']):
            raise ValueError('Fit/evaluation classifier differs')
    maps,_=load_features(feature_root,'validation')
    with np.load(fit_root/'validation_spatial.npz',allow_pickle=False) as z:
        raw,scores,relevance=z['features'],z['scores'],z['local_relevance']
    if not np.array_equal(raw,maps.transpose(0,2,3,1).reshape(-1,2048)):
        raise ValueError('Validation raw feature identity mismatch')
    images=load_images(source['dataset']['validation'])
    for i,im in enumerate(images):im.segments[0].model_act=maps[i]
    data=out/'evaluation_data';data.mkdir();curves={}
    from utils import utils_general
    target=utils_general.get_imagenet_class_index(config['class_name'])
    for mode in ('sdc','ssc'):
        counts=[];fractions=[];details=[]
        def trajectories():
            for i,im in enumerate(images):
                adapter=MCDCachedScores(model,bases,raw[i*49:(i+1)*49],scores[i*49:(i+1)*49],relevance[i*49:(i+1)*49])
                # Direct original rule including empty means, sorting, complement exclusion and endpoint break.
                masked=benchmark.iter_mask_imgs_mcd(adapter,[im],mode)[0]
                assignment,detail=trajectory_details(adapter,im)
                np.save(data/f'{mode}_pixel_assignments_{i:03d}.npy',assignment)
                save_masks(data/f'mcd_{mode}_masks_{i:03d}.npz',np.stack([s.mask for s in masked.segments]))
                counts.append(len(masked.segments));fractions.append([float(100*np.mean(s.mask)) for s in masked.segments])
                details.append(dict(image_index=i,input=source['dataset']['validation'][i],states=len(masked.segments),**detail))
                write_json(data/f'mcd_{mode}_states_partial.json',details)
                yield i,masked
        with (data/f'mcd_{mode}_partial_logits.f32').open('xb') as partial, (data/f'mcd_{mode}_partial_batches.jsonl').open('x') as groups:
            def checkpoint(values,identities):
                if values.dtype!=np.float32:raise ValueError('Unexpected logits dtype')
                values.tofile(partial);partial.flush();groups.write(json.dumps(identities)+'\n');groups.flush()
            logits,batches=predict_stream(model,trajectories(),8,
                lambda n,b:mark(mode=mode,predicted_states=n,batches=b),on_batch=checkpoint)
        offsets=np.r_[0,np.cumsum(counts)];correct=logits.argmax(1)==target
        grouped=[correct[a:b].tolist() for a,b in zip(offsets[:-1],offsets[1:])]
        avg,std=benchmark.calc_avg_and_std(grouped,50);pixels,pstd=benchmark.calc_avg_and_std(fractions,50)
        contributors=[sum(n>j for n in counts) for j in range(max(counts))];included=[j for j,n in enumerate(contributors) if n>37.5]
        curve=dict(mode=mode,setting='mcd',state_counts=counts,step_indices=included,
            all_step_contributors=contributors,contributors=[contributors[j] for j in included],
            accuracy_mean=avg.tolist(),accuracy_std=std.tolist(),visible_pixel_percent_mean=pixels.tolist(),
            visible_pixel_percent_std=pstd.tolist(),upstream_plot_x_deleted_fraction=(1-.01*pixels).tolist(),total_states=len(logits))
        curves['mcd_'+mode]=curve
        np.savez_compressed(data/f'mcd_{mode}_predictions.npz',logits=logits,correct=correct,offsets=offsets,visible_pixel_percent=np.concatenate(fractions))
        write_json(data/f'mcd_{mode}_states.json',dict(images=details,global_batches=batches))
        write_json(out/'evaluation_curves.json',curves)
        note('absent_concept_empty_mean',dict(mode=mode,per_image=[dict(image_index=d['image_index'],absent=d['absent_concepts']) for d in details if d['absent_concepts']]),
             'Original NaN importance only for absent masks; masks skipped; ordering preserved')
    reused=verify_random_reuse(config,source);write_json(out/'random_reuse.json',reused)
    return dict(concepts=len(bases),selected_k=fit_manifest['result']['selected_k'],
        threshold_met=fit_manifest['result']['threshold_met'],fit_manifest_sha256=config['fit_manifest_sha256'],
        curves=curves,random_reuse=reused,precision=runtime_precision(torch),
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),scope='Golden only; not a ten-class baseline or human-study result')


def archive_large_ssc(out, limit=128*1024**2, chunk_bytes=64*1024**2):
    """Lossless byte chunks AFTER source SSC cache use; collector max is 256 MiB.

    Concatenate listed chunks to restore the original .npz; verify original hash.
    Only newly created MCD SSC files are archived, never source/input caches.
    """
    archives=[]
    for path in sorted((out/'ssc').glob('*.npz')):
        if path.stat().st_size <= limit:continue
        record=dict(original_path=str(path.relative_to(out)),bytes=path.stat().st_size,
                    sha256=sha256(path),chunks=[])
        with path.open('rb') as stream:
            i=0
            while True:
                block=stream.read(chunk_bytes)
                if not block:break
                target=path.with_name(path.name+f'.part{i:04d}')
                with target.open('xb') as dest:dest.write(block)
                record['chunks'].append(dict(path=str(target.relative_to(out)),bytes=len(block),sha256=sha256(target)))
                i+=1
        archives.append(record)
        write_json(out/'ssc_archives.json',archives)
        path.unlink()
    return archives


def run(config:dict,output:Path)->dict:
    """Route from workstream_runtime; caller owns job submission and output collection."""
    stage=config['stage']
    if stage not in ('features','fit','evaluation'):raise ValueError('Unknown MCD stage')
    import re
    if not os.environ.get('SLURM_JOB_ID','').isdigit() or not re.fullmatch(r'bun\d{3}',socket.gethostname().split('.')[0]):
        raise RuntimeError('All MCD work requires a Slurm compute allocation')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if commit!=config['execution_commit'] or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Wrong or dirty execution release')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if any(out.glob('mcd_*')):raise FileExistsError('MCD output already used; no in-place retry')
    for key in ('feature_dir','fit_dir','golden_run_dir','source_dir'):
        if key in config and (out.resolve().is_relative_to(Path(config[key]).resolve()) or Path(config[key]).resolve().is_relative_to(out.resolve())):
            raise ValueError('Output overlaps immutable inputs/cache')
    preexisting={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}
    started=time.monotonic();anomalies=[]
    record=dict(status='RUNNING',stage=stage,job_id=os.environ['SLURM_JOB_ID'],commit=commit,
        implementation_sha256=sha256(Path(__file__)),config_sha256=digest(config),protocol=PROTOCOL)
    write_json(out/'mcd_config.json',config);write_json(out/'mcd_manifest.json',record)
    def mark(**extra):write_json(out/'mcd_progress.json',dict(status=record['status'],stage=stage,elapsed_seconds=time.monotonic()-started,**extra))
    def note(code,evidence,impact):
        anomalies.append(dict(code=code,stage=stage,job_id=record['job_id'],evidence=evidence,impact=impact,status='RECORDED'))
        write_json(out/'mcd_anomalies.json',anomalies)
    try:
        from threadpoolctl import threadpool_limits
        cpus=int(os.environ['SLURM_CPUS_PER_TASK'])
        os.environ['LOKY_MAX_CPU_COUNT']=str(cpus)
        os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
        # SSC parallel workers are bounded by allocation; numerical thread scheduling only.
        import torch
        torch.set_num_threads(cpus)
        # Do not introduce a spectral RNG/v0 protocol absent from benchmark_methods.
        # Save process RNG states; these do not attest ARPACK's internal start vector.
        record['rng_start']=rng_state()
        record['versions']={k:importlib.metadata.version(k) for k in ('numpy','scipy','torch','torchvision','timm','scikit-learn','scikit-image','scikit-dimension','joblib')}
        source=verify_common(config)
        for name,expected in source['golden_versions'].items():
            if name in record['versions'] and record['versions'][name] != expected:
                raise ValueError('Library version differs from Golden: '+name)
        record.update(signature=source['signature'],signature_sha256=source['signature_sha256'])
        write_json(out/'mcd_inputs.json',source['dataset']);write_json(out/'mcd_manifest.json',record);mark()
        original_warning=warnings.showwarning
        def capture(message,category,filename,lineno,file=None,line=None):
            note('python_warning',dict(message=str(message),category=category.__name__,filename=filename,line=lineno),'Inspect in context; not automatically a scientific failure')
            original_warning(message,category,filename,lineno,file,line)
        with warnings.catch_warnings(),threadpool_limits(limits=1):
            warnings.simplefilter('always');warnings.showwarning=capture
            result=globals()[stage](config,out,source,mark,note)
        record.update(status='PASS',result=result,rng_end=rng_state())
        write_json(out/'mcd_result.json',result)
        return result
    except BaseException as exc:
        record.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        note('stage_failed',record['traceback'],'Affected stage incomplete; no automatic retry')
        raise
    finally:
        record['elapsed_seconds']=time.monotonic()-started
        write_json(out/'mcd_anomalies.json',anomalies);mark()
        archive_large_ssc(out)
        record['files']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha256(p))
            for p in sorted(out.rglob('*')) if p.is_file() and p.name!='mcd_manifest.json' and str(p.relative_to(out)) not in preexisting}
        write_json(out/'mcd_manifest.json',record)
