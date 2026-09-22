"""Pure proposed S statistics; no dataset loading, inference, cohort selection or CLI.

Not a medical evaluation result. Callers must supply accepted, correctly aligned
effective input masks, assignments and sign labels after separate authorization.
"""
import numpy as np

GROUPS = {'pigment_network': ('TYP','ATP'), 'blue_whitish_veil': ('PRS',),
          'vascular_structures': ('REG','IR'), 'pigmentation': ('REG','IR'),
          'streaks': ('REG','IR'), 'dots_and_globules': ('REG','IR'),
          'regression_structures': ('PRS',)}
CONTRASTS = tuple((sign, group, 'ABS') for sign,groups in GROUPS.items() for group in groups)


def coverage(masks, assignments, valid_feature, concept_ids, processing_complete=True):
    """Union within a concept, full input denominator. Missing is never zero.

The caller excludes OC from concept_ids and reports OC separately. No normalization
across concepts: masks belonging to different concepts may overlap spatially.
"""
    if not processing_complete:
        return {cid: {'value': None, 'status': 'MISSING_PROCESSING'} for cid in concept_ids}
    masks=np.asarray(masks); assignments=np.asarray(assignments); valid=np.asarray(valid_feature)
    if (masks.ndim!=3 or masks.dtype!=bool or masks.shape[1]*masks.shape[2]==0 or
            assignments.shape!=(len(masks),) or valid.shape!=(len(masks),) or valid.dtype!=bool):
        raise ValueError('Boolean effective masks and one aligned assignment/validity row per region required')
    if len(set(concept_ids))!=len(concept_ids):raise ValueError('Duplicate concept ID')
    result={}
    for cid in concept_ids:
        selected=(assignments==cid)&valid
        union=masks[selected].any(axis=0)
        result[cid]=dict(value=float(union.mean()), valid_member_count=int(selected.sum()),
                         status='MEASURED' if selected.any() else 'NO_VALID_ASSIGNED_MEMBER')
    return result


def contrast(values, labels, named_group, reference='ABS'):
    """Higher coverage predicts the named group; no direction flipping or p value."""
    from sklearn.metrics import roc_auc_score
    values=np.asarray(values,dtype=float); labels=np.asarray(labels)
    if values.shape!=labels.shape or values.ndim!=1:raise ValueError('Aligned image-level columns required')
    if np.any(np.isinf(values)) or np.any((values[np.isfinite(values)]<0)|(values[np.isfinite(values)]>1)):
        raise ValueError('Invalid coverage value; only explicit missing NaNs are allowed')
    chosen=(labels==named_group)|(labels==reference); usable=chosen&np.isfinite(values)
    positives=int(np.sum(usable&(labels==named_group))); negatives=int(np.sum(usable&(labels==reference)))
    return dict(group=named_group,reference=reference,group_n=positives,reference_n=negatives,
                missing_n=int(np.sum(chosen&~np.isfinite(values))),other_category_n=int(np.sum(~chosen)),
                auc=float(roc_auc_score(labels[usable]==named_group,values[usable])) if positives and negatives else None,
                status='DESCRIPTIVE_ONLY' if positives and negatives else 'UNDEFINED_EMPTY_GROUP',
                direction='Higher coverage -> named non-absent group; never reversed',
                p_value=None,semantic_identity_or_spatial_localization_test=False)
