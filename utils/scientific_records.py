"""Auditable numerical records; observe the released pipeline without changing fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import norm
from utils import utils_mcd


def numerical_checks(acts, logits, weight, bias, bases, sample_indices=None):
    """Check observed FC outputs and released oblique decomposition with bias separate."""
    acts, logits = np.asarray(acts), np.asarray(logits)
    if not all(np.isfinite(x).all() for x in (acts, logits, weight, np.asarray(bias))):
        raise ValueError('Non-finite features, logits or classifier parameters')
    reconstructed = acts @ weight.T + bias
    np.testing.assert_allclose(reconstructed, logits, rtol=1e-4, atol=1e-4,
                               err_msg='global_pool features do not reconstruct observed FC logits')
    result = {'fc_max_abs_error':float(np.max(np.abs(reconstructed-logits))),
              'fc_rtol':1e-4,'fc_atol':1e-4,'fc_checked_segments':int(len(acts))}
    if bases is None:
        return result, {}
    if not all(np.isfinite(b).all() for b in bases):
        raise ValueError('Non-finite fitted basis')
    if sample_indices is None:
        sample_indices=np.unique(np.linspace(0,len(acts)-1,min(8,len(acts)),dtype=int))
    projections=np.stack([utils_mcd.subspace_projection(bases,acts[i]) for i in sample_indices])
    features=acts[sample_indices]
    relative=norm(projections.sum(axis=1)-features,axis=1)/np.maximum(norm(features,axis=1),1e-12)
    if relative.max()>1e-4:
        raise ValueError('Released concept-plus-complement feature reconstruction failed')
    target_weight=weight if weight.ndim==1 else weight[0]
    target_bias=bias if np.ndim(bias)==0 else bias[0]
    relevance=projections @ target_weight
    target_logits=logits[sample_indices] if logits.ndim==1 else logits[sample_indices,0]
    np.testing.assert_allclose(relevance.sum(axis=1)+target_bias,target_logits,rtol=1e-4,atol=1e-4)
    result.update(feature_reconstruction_max_relative_error=float(relative.max()),
                  relevance_logit_max_abs_error=float(np.max(np.abs(relevance.sum(axis=1)+target_bias-target_logits))),
                  decomposition_sample_count=len(sample_indices),bias_added_separately=True,
                  scope='Segment-level sample reconstruction; not a full spatial relevance benchmark')
    return result, {'sample_indices':sample_indices,'sample_features':features,
                    'sample_concept_components':projections,'sample_local_relevance':relevance}


def save_discovery(explainer, output, class_index):
    output=Path(output)/'scientific';output.mkdir(exist_ok=False)
    data={f'concept_basis_{i:03d}':b for i,b in enumerate(explainer.concept_bases)}
    data['complement_basis']=explainer.compl_basis
    data['cluster_labels']=explainer.clustering.labels
    data['outlier_mask']=explainer.clustering.outlier_mask
    data['retained_cluster_labels']=np.asarray([c.label for c in explainer.concepts])
    data['fc_weight']=explainer.model.fc.weight.detach().cpu().numpy()
    data['fc_bias']=explainer.model.fc.bias.detach().cpu().numpy()
    np.savez_compressed(output/'discovery.npz',**data)
    (output/'discovery.json').write_text(json.dumps({'class_index':int(class_index),
        'basis_shapes':[list(b.shape) for b in explainer.concept_bases],
        'complement_shape':list(explainer.compl_basis.shape),
        'cluster_label_order':'training segment order before ClusterClass sorting',
        'model_default_cfg':explainer.model.default_cfg},indent=2,default=str)+'\n')


def save_split(explainer, images, activations, output, split, class_index):
    output=Path(output)/'scientific'
    segments=[s for im in images for s in im.segments]
    acts=np.stack([s.model_act for s in segments]);logits=np.stack([s.model_pred for s in segments])
    weight=explainer.model.fc.weight.detach().cpu().numpy()
    bias=explainer.model.fc.bias.detach().cpu().numpy()
    # All 1000 logits verify the feature hook, then target-class decomposition includes complement.
    fc,_=numerical_checks(acts,logits,weight,bias,None)
    checks,samples=numerical_checks(acts,logits[:,class_index],weight[class_index],bias[class_index],
                                   explainer.concept_bases+[explainer.compl_basis])
    checks['all_classes_fc']=fc
    observed_relevance=explainer.concept_relevances(acts[samples['sample_indices']],n_jobs=1)
    np.testing.assert_allclose(observed_relevance,samples['sample_local_relevance'],rtol=1e-5,atol=1e-5)
    checks['released_concept_relevances_agrees']=True
    checks['finite_concept_activations']=bool(np.isfinite(activations).all())
    if not checks['finite_concept_activations']:
        raise ValueError('Non-finite concept assignments')
    np.savez_compressed(output/(split+'.npz'),features=acts,logits=logits,
                        concept_activations=activations,assignments=np.argmax(activations,axis=1),**samples)
    mapping=[]
    for image_index,im in enumerate(images):
        for segment_index,segment in enumerate(im.segments):
            mapping.append({'image_index':image_index,'segment_index':segment_index,'image_path':im.filename,
                            'mask_shape':list(segment.mask.shape),'mask_dtype':str(segment.mask.dtype),
                            'mask_sha256':hashlib.sha256(segment.mask.tobytes()).hexdigest()})
    (output/(split+'_segments.json')).write_text(json.dumps(mapping,indent=2)+'\n')
    (output/(split+'_checks.json')).write_text(json.dumps(checks,indent=2)+'\n')
    return checks
