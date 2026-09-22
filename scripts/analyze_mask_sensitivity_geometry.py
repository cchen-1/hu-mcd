"""Complete regional geometry/score diagnostics from accepted cached A arrays."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.accept_next_phase_batch import OUT,J,loadz,table
import numpy as np

allrows=[];summary=[]
for entry in J(OUT/'acceptance-both.json')['robustness']['statuses']:
    p=next((OUT/'collections'/entry['job']).glob('*'));ledger=J(p/'region_ledger.json')
    base=loadz(p/'identity0_science.npz');z=loadz(p/'identity0_masks.npz');m0=np.unpackbits(z['packed'],axis=2,count=224).astype(bool)
    completed=entry['completed_conditions'].split(',')
    for name in ['identity0','erosion1','erosion2','dilation1','dilation2']:
        if name in completed:
            z=loadz(p/(name+'_masks.npz'));m=np.unpackbits(z['packed'],axis=2,count=224).astype(bool);d=loadz(p/(name+'_science.npz'))
            both=d['valid']&base['valid'];inter=(m&m0).sum((1,2));union=(m|m0).sum((1,2));iou=np.divide(inter,union,out=np.ones(len(m),float),where=union>0)
            cosine=np.full(len(m),np.nan);relative=cosine.copy();margin=cosine.copy()
            b=base['features'][both];v=d['features'][both];cosine[both]=(b*v).sum(1)/(np.linalg.norm(b,axis=1)*np.linalg.norm(v,axis=1));relative[both]=np.linalg.norm(v-b,axis=1)/np.linalg.norm(b,axis=1)
            scores=d['scores'][both];original=base['assignments'][both];competitor=scores.copy();competitor[np.arange(len(original)),original]=-np.inf
            margin[both]=scores[np.arange(len(original)),original]-competitor.max(1)
            summary.append(dict(class_name=entry['class_name'],condition=name,all_regions=len(m),mean_iou=float(iou.mean()),mean_symmetric_difference_pixels=float((m^m0).sum((1,2)).mean()),both_valid=int(both.sum()),mean_cosine=float(cosine[both].mean()),mean_relative_feature_change=float(relative[both].mean()),mean_original_winner_margin=float(margin[both].mean()),original_winner_negative_margin=int((margin[both]<0).sum())))
        for i,r in enumerate(ledger):
            rec=dict(class_name=entry['class_name'],job=entry['job'],condition=name,region_id=r['region_id'],image_id=r['image_id'],baseline_valid=bool(base['valid'][i]),baseline_assignment=int(base['assignments'][i]),baseline_area=int(base['area'][i]),status='NOT_COMPLETED_FAILED_CLASS',current_assignment=None,current_area=None,symmetric_difference=None,iou=None,cosine=None,relative_feature_change=None,original_winner_margin=None,new_winner_margin=None)
            if name in completed:
                rec.update(status='EMPTY_AFTER_INTERVENTION' if d['empty'][i] else 'NONEMPTY_ZERO_FEATURE' if d['zero'][i] else 'VALID',current_assignment=int(d['assignments'][i]) if d['valid'][i] else None,current_area=int(d['area'][i]),symmetric_difference=int((m[i]^m0[i]).sum()),iou=float(iou[i]))
                for k,arr in [('cosine',cosine),('relative_feature_change',relative),('original_winner_margin',margin),('new_winner_margin',d['margin'])]:rec[k]=float(arr[i]) if np.isfinite(arr[i]) else None
            allrows.append(rec)
table(OUT/'robustness-regions-all-conditions.csv',allrows);table(OUT/'robustness-geometry-score-summary.csv',summary)
print('region-condition rows',len(allrows),'computed',sum(r['status']!='NOT_COMPLETED_FAILED_CLASS' for r in allrows),'explicit missing',sum(r['status']=='NOT_COMPLETED_FAILED_CLASS' for r in allrows))
