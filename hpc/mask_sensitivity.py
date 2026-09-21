"""Approved final-mask sensitivity; frozen discovery scores, no SAM or refitting.

Raw row slots never move between original batches. Empty masks have missing
features/scores and explicit status; zero features have no valid concept label.
An unchanged control must pass before perturbations, including reduced batches.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.linalg import lu_factor, lu_solve, norm
from hpc.final_mask_intervention import FinalMaskInterventionDataset
from utils.run_tracking import atomic_json, sha256, utc_now

CONDITIONS = [('identity', 0), ('erosion', 1), ('erosion', 2), ('dilation', 1), ('dilation', 2)]
ROOT = Path(__file__).resolve().parents[1]


def checked(path, expected):
    path = Path(path)
    if sha256(path) != expected:
        raise ValueError('Source identity mismatch: '+str(path))
    return path


def bytes_sha(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


class ObliqueScores:
    """Same coefficients/component norms as released subspace_projection.

    LU factorization/multiple RHS reuse changes operation order only. The saved
    discovery scores, assignments and reconstruction are mandatory controls.
    """
    def __init__(self, bases):
        self.bases = bases
        self.union = np.concatenate(bases)
        if self.union.shape != (2048, 2048) or not np.isfinite(self.union).all():
            raise ValueError('Expected finite complete 2048-dimensional saved basis')
        self.factor = lu_factor(self.union.T)

    def __call__(self, features):
        coefficients = lu_solve(self.factor, features.T).T
        lengths = np.asarray([norm(row) for row in features], dtype=features.dtype)
        if not (lengths > 0).all():
            raise ValueError('Only nonzero finite features may receive concept scores')
        scores = np.empty((len(features), len(self.bases)), dtype=np.float64)
        recon = np.zeros(features.shape, dtype=np.float64)
        offset = 0
        for i, basis in enumerate(self.bases):
            component = coefficients[:, offset:offset+len(basis)] @ basis
            scores[:, i] = norm(component, axis=1)/lengths
            recon += component
            offset += len(basis)
        error = norm(recon-features, axis=1)/lengths
        if not np.isfinite(scores).all() or not np.isfinite(error).all() or error.max(initial=0) > 1e-4:
            raise ValueError('Oblique decomposition reconstruction failed')
        return scores, error


def identity_check(actual, expected, scorer, expected_scores, expected_assignments):
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    zero = np.all(expected == 0, axis=1)
    np.testing.assert_array_equal(np.all(actual == 0, axis=1), zero)
    valid = ~zero
    relative = norm(actual[valid]-expected[valid], axis=1)/norm(expected[valid], axis=1)
    if relative.max(initial=0) > 1e-4:
        raise ValueError('Unchanged-feature relative tolerance exceeded')
    scores, _ = scorer(actual[valid])
    np.testing.assert_allclose(scores, expected_scores[valid], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(scores.argmax(1), expected_assignments[valid])
    return dict(status='PASS', rows=len(actual), zero_rows=int(zero.sum()),
                max_abs_feature=float(np.max(np.abs(actual-expected), initial=0)),
                max_relative_feature=float(relative.max(initial=0)),
                max_abs_score=float(np.max(np.abs(scores-expected_scores[valid]), initial=0)))


def run(config, out):
    import torch
    import timm
    import importlib.metadata
    from classes import ImageClass, SegmentClass, ConceptDatasetClass
    from utils import utils_general
    from hpc.evaluate_reference import runtime_precision
    from hpc.workstream_runtime import event

    started = time.time()
    payload_file = checked(config['payload'], config['payload_sha256'])
    payload = json.loads(payload_file.read_text())
    if payload['authorization'] != 'APPROVED' or payload['conditions'] != [list(x) for x in CONDITIONS]:
        raise ValueError('Unapproved or changed sensitivity conditions')
    shutil.copy2(payload_file, out/'frozen_protocol.json')
    source = payload['classes'][config['class_name']]
    ledger = source['regions']; n = len(ledger)
    def mark(stage, **details):
        atomic_json(out/'sensitivity_progress.json', dict(stage=stage, class_name=config['class_name'],
                    utc=utc_now(), elapsed_seconds=time.time()-started, **details))
        print(stage, details, flush=True)
    mark('source_identity')
    for f in source['files']:
        checked(f['path'], f['sha256'])
    for name, expected in source['science_sha256'].items():
        checked(ROOT/name, expected)
    old = json.loads(Path(source['config']).read_text())
    dc = json.loads(Path(source['discovery_json']).read_text())
    dataset = json.loads(Path(old['dataset_manifest']).read_text())
    if old['class_name'] != config['class_name'] or old['batch_size'] != 8 or len(dataset['validation']) != 50:
        raise ValueError('Frozen class/batch/input count mismatch')
    checked(old['resnet_checkpoint'], old['resnet_checkpoint_sha256'])
    if runtime_precision(torch) != old['precision'] or importlib.metadata.version('timm') != '0.6.13':
        raise ValueError('Runtime precision or timm mismatch')
    for name in ('resnet.py', 'sal_layers.py'):
        checked(Path(timm.__file__).parent/'models'/name, sha256(ROOT/'input_masking'/name))
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated CUDA GPU required')
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    utils_general.DEVICE = 'cuda:0'
    torch.cuda.set_device('cuda:0')
    model = timm.create_model('resnet50', pretrained=False)
    model.load_state_dict(torch.load(old['resnet_checkpoint'], map_location='cpu'), strict=True)
    model.eval().to('cuda:0')
    if json.loads(json.dumps(model.default_cfg)) != dc['model_default_cfg']:
        raise ValueError('Model configuration changed')
    atomic_json(out/'model_identity.json', dict(classifier_sha256=old['resnet_checkpoint_sha256'],
        precision=runtime_precision(torch), default_cfg=model.default_cfg, device=torch.cuda.get_device_name(),
        source_commit=source['binding']['source_commit'], batch_size=8,
        masks='Frozen SAM cache; no SAM execution', source_binding=source['binding']))
    images=[]
    for row in dataset['validation']:
        checked(row['input_path'], row['input_sha256'])
        image=ImageClass(row['input_path'], old['max_shortest_side'])
        masks=np.load(Path(source['binding']['source_cache'])/'segments_validation'/(row['prepared_name']+'_sam.npy'), allow_pickle=False)
        image.segments=[SegmentClass(m, image) for m in masks]
        images.append(image)
    original=ConceptDatasetClass(images, model.default_cfg, 0, True, -1, .25)
    if len(original) != n:
        raise ValueError('Raw region count mismatch')
    for i, r in enumerate(ledger):
        if r['raw_feature_row'] != i or bytes_sha(original[i][1].numpy()[0].astype(bool)) != r['effective_mask_sha256']:
            raise ValueError('Effective mask/row identity mismatch at '+str(i))
    atomic_json(out/'region_ledger.json', ledger)
    atomic_json(out/'input_manifest.json', dataset)
    raw=np.load(source['raw_features'], allow_pickle=False)
    if raw.shape != (n, 2048) or not np.isfinite(raw).all():
        raise ValueError('Raw feature shape/nonfinite mismatch')
    with np.load(source['validation_npz'], allow_pickle=False) as z:
        saved={k:z[k] for k in z.files}
    with np.load(source['basis_npz'], allow_pickle=False) as z:
        bases=[z[k] for k in sorted(z.files) if k.startswith('concept_basis_')]+[z['complement_basis']]
    scorer=ObliqueScores(bases)
    zero=np.all(raw==0, axis=1); valid=~zero
    np.testing.assert_array_equal(raw[valid], saved['features'])
    old_scores=np.full((n,len(bases)),np.nan)
    old_assign=np.full(n,-1,dtype=np.int32)
    old_scores[valid]=saved['concept_activations'];old_assign[valid]=saved['assignments']
    replay,_=scorer(raw[valid])
    np.testing.assert_allclose(replay,old_scores[valid],rtol=1e-5,atol=1e-5)
    np.testing.assert_array_equal(replay.argmax(1),old_assign[valid])
    shutil.copy2(source['basis_npz'],out/'fixed_discovery.npz')
    area=np.array([r['baseline_effective_pixels'] for r in ledger]);margins=np.full(n,np.nan)
    margins[valid]=np.sort(old_scores[valid],axis=1)[:,-1]-np.sort(old_scores[valid],axis=1)[:,-2]
    cuts={key:np.quantile(values[valid],[.25,.5,.75]).tolist() for key,values in [('area',area),('margin',margins)]}
    atomic_json(out/'baseline_strata.json',dict(cuts=cuts, rule='np.searchsorted(cuts, value, side=right); ties unchanged, empty bins retained', source='Valid baseline only; frozen before inference'))
    batches=source['binding']['inference_batches']
    if batches != [list(range(i,min(i+8,n))) for i in range(0,n,8)]:
        raise ValueError('Original batch slots changed')
    hook_values=[]
    handle=model.global_pool.register_forward_hook(lambda m,a,v:hook_values.append(v.detach().cpu().numpy()))
    def forward(ds,ids):
        hook_values.clear()
        with torch.no_grad():
            logits=model(utils_general.custom_collate([ds[i] for i in ids])).detach().cpu().numpy()
        f=np.concatenate(hook_values,axis=0)
        if f.shape!=(len(ids),2048) or logits.shape!=(len(ids),1000) or not np.isfinite(f).all() or not np.isfinite(logits).all():
            raise ValueError('Nonfinite/unexpected CNN output')
        return f,logits
    controls=[];summary=[]
    try:
        for operation,radius in CONDITIONS:
            name=operation+str(radius);mark(name)
            ds=FinalMaskInterventionDataset(original,operation,radius)
            masks=np.stack([ds[i][1].numpy()[0].astype(bool) for i in range(n)])
            counts=masks.sum((1,2));empty=counts==0
            if [int(v) for v in counts] != [r[name+'_pixels'] for r in ledger]:
                raise ValueError('Geometry differs from approved inventory')
            np.savez_compressed(out/(name+'_masks.npz'),packed=np.packbits(masks,axis=2),shape=np.array(masks.shape))
            features=np.full((n,2048),np.nan,dtype=np.float32);logits=np.full((n,1000),np.nan,dtype=np.float32)
            for bi,ids in enumerate(batches):
                kept=[i for i in ids if not empty[i]]
                if len(kept)!=len(ids) and kept:
                    f,l=forward(original,kept)
                    np.savez_compressed(out/(name+'_batch'+str(bi)+'_unchanged.npz'),rows=kept,features=f,logits=l)
                    control=dict(condition=name,batch=bi,rows=kept,check=identity_check(f,raw[kept],scorer,old_scores[kept],old_assign[kept]))
                    controls.append(control);atomic_json(out/'reduced_batch_controls.json',controls)
                if kept:
                    features[kept],logits[kept]=forward(ds,kept)
                if bi%20==0:mark(name,completed_batches=bi+1,total_batches=len(batches))
            is_zero=(~empty)&np.all(features==0,axis=1);is_valid=~(empty|is_zero)
            scores=np.full((n,len(bases)),np.nan);assign=np.full(n,-1,dtype=np.int32);error=np.full(n,np.nan)
            scores[is_valid],error[is_valid]=scorer(features[is_valid]);assign[is_valid]=scores[is_valid].argmax(1)
            margin=np.full(n,np.nan)
            margin[is_valid]=np.sort(scores[is_valid],axis=1)[:,-1]-np.sort(scores[is_valid],axis=1)[:,-2]
            # Persist all evidence before a gate can raise, including failed controls.
            np.savez_compressed(out/(name+'_science.npz'),features=features,logits=logits,scores=scores,
                assignments=assign,empty=empty,zero=is_zero,valid=is_valid,area=counts,margin=margin,
                reconstruction_relative_error=error,baseline_valid=valid,baseline_assignments=old_assign)
            if operation=='identity':
                gate=identity_check(features,raw,scorer,old_scores,old_assign)
                atomic_json(out/'identity_gate.json',gate)
            record=dict(condition=name,rows=n,empty=int(empty.sum()),zero=int(is_zero.sum()),valid=int(is_valid.sum()),
                baseline_valid=int(valid.sum()),retained=int((valid&is_valid&(assign==old_assign)).sum()),
                baseline_zero_to_valid=int((zero&is_valid).sum()),output_missing_value='NaN with explicit state; assignment -1 is invalid, not a concept')
            if empty.any() or is_zero.any():
                event(out,name,'ROB26-004-empty-or-zero',record,'Retained in ledger and baseline denominator; no valid concept for empty/zero', 'Expected protocol states; record without filtering',status='RECORDED')
            summary.append(record);atomic_json(out/'condition_counts.json',summary)
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>1024**3:
                raise RuntimeError('Per-class 1GiB output cap exceeded; no automatic expansion')
    finally:
        handle.remove()
    mark('COMPLETE_PENDING_ACCEPTANCE')
    return dict(status='COMPLETE_PENDING_ACCEPTANCE',class_name=config['class_name'],conditions=summary,
        controls=controls,source_job=source['binding']['source_job'],elapsed_seconds=time.time()-started,
        analysis='Paired image bootstrap and grouped summaries deferred until user completion notification')
