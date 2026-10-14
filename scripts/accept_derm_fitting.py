"""Local cached-result acceptance only; no model predictions or remote computation."""
from pathlib import Path
import csv,json,hashlib,random
from collections import Counter
import numpy as np
from PIL import Image
from utils.run_tracking import sha256
from utils.utils_general import get_unique_concept_masks
R=Path(__file__).resolve().parents[1];BASE=R/'artifacts/bunya/ham-derm-discovery-20260924';P=BASE/'collection-28889071'
def read(n):return json.loads((P/n).read_text())
def bh(a):return hashlib.sha256(a.tobytes()).hexdigest()
def main():
 index=read('artifacts.json');config=read('resolved_config.json');result=read('result.json')
 assert index['execution_commit']==config['execution_commit']=='649ce176c7d2e4e17b9e9be196fb0e7c33ef3e42'
 for name,item in index['files'].items():assert sha256(P/name)==item['sha256'] and (P/name).stat().st_size==item['bytes'],name
 assert read('status.json')['status']=='COMPLETED_PENDING_ACCEPTANCE'
 assert config['protocol']==json.loads((R/'configs/medical/derm-r101-fit.approved.json').read_text())['protocol']
 for entry in read('input_binding.json').values():assert entry['sha256']==entry['expected_sha256']
 source=read('fit_input_binding.json')['training'];prior=json.loads((BASE/'collection-28883445/derm7pt_fit_manifest.json').read_text())['rows'];assert source==prior
 assert len(source)==101 and [r['fit_order'] for r in source]==list(range(101))
 raw=read('training_raw_segments.json');assign=read('training_assignments.json');concepts=read('concept_index.json');filters=read('cluster_filters.json')
 rawf=np.load(P/'training_raw_features.npz');f=np.load(P/'scientific/training.npz');cl=np.load(P/'initial_clusters.npz');disc=np.load(P/'scientific/discovery.npz');full=np.load(P/'training_full_images.npz')
 assert np.array_equal(rawf['segment_id'],[s['segment_id'] for s in raw]);assert [s['global_row'] for s in raw]==list(range(len(raw)))
 zero=~np.any(rawf['features']!=0,axis=1);assert np.array_equal(zero,[s['zero_feature'] for s in raw]);kept=[s for s in raw if not s['zero_feature']]
 np.testing.assert_array_equal(rawf['features'][~zero],f['features']);np.testing.assert_array_equal(rawf['logits'][~zero],f['logits'])
 assert np.array_equal(cl['segment_id'],[s['segment_id'] for s in kept]);assert np.array_equal(cl['labels'],disc['cluster_labels']);assert not cl['outlier_mask'].any()
 assert [s['segment_id'] for s in assign]==[s['segment_id'] for s in kept]
 assert np.array_equal(f['concept_activations'].argmax(1),f['assignments']);assert np.array_equal(f['assignments'],[s['assignment'] for s in assign])
 assert np.array_equal(full['image_id'],[s['image_id'] for s in source])
 random.seed(43) # released mask-region IDs consume Python RNG in fixed image order
 total_sam=0;empty_images=[];zero_only=[];area_zero=[];area_nonzero=[]
 for row in source:
  iid=row['image_id'];inp=BASE/'collection-28883445/derm7pt_r101_presam'/Path(row['input_path']).name
  assert sha256(inp)==row['input_sha256']
  with Image.open(inp) as im:assert list(im.size)==row['input_size_wh'] and im.mode=='RGB'
  masks=np.load(P/f'cache/training/{iid}.png_sam.npy',allow_pickle=False)
  original=np.load(P/f'cache/training/{iid}_sam_raw.npz')['masks'];meta=read(f'cache/training/{iid}_sam_metadata.json');assert len(original)==len(meta);total_sam+=len(original)
  sorted_masks=[original[i] for i in sorted(range(len(meta)),key=lambda i:meta[i]['predicted_iou'],reverse=True)]
  replay=np.asarray(get_unique_concept_masks(sorted_masks,.01),dtype=np.float32);np.testing.assert_array_equal(replay,masks)
  records=[s for s in raw if s['image_id']==iid];assert len(records)==len(masks)
  eff=np.load(P/f'cache/training/{iid}_effective_masks.npz')['masks'];assert len(eff)==len(records)
  for j,s in enumerate(records):
   assert s['raw_index']==j and s['raw_mask_sha256']==bh(masks[j]) and s['effective_mask_sha256']==bh(eff[j])
   assert int(masks[j].sum())==s['raw_pixels'] and int(eff[j].sum())==s['effective_pixels']
   (area_zero if s['zero_feature'] else area_nonzero).append(float(eff[j].mean()))
  if not records:empty_images.append(iid)
  if records and all(s['zero_feature'] for s in records):zero_only.append(iid)
 for a in [rawf['features'],rawf['logits'],f['concept_activations'],full['features'],full['logits']]:assert np.isfinite(a).all()
 for feats,logits in [(rawf['features'],rawf['logits']),(full['features'],full['logits'])]:np.testing.assert_allclose(feats@disc['fc_weight'].T+disc['fc_bias'],logits,rtol=1e-4,atol=1e-4)
 checks=read('scientific/training_checks.json');assert checks['status']=='PASS'
 for z in [f,np.load(P/'training_full_reconstruction.npz')]:
  rel=np.linalg.norm(z['sample_concept_components'].sum(1)-z['sample_features'],axis=1)/np.maximum(np.linalg.norm(z['sample_features'],axis=1),1e-12);assert rel.max()<1e-4
 # Saved same-batch checks, not another forward pass.
 aa=np.load(P/'adapter_check_arrays.npz')
 for check in read('adapter_checks.json')['checks']:
  assert check['pass_check'];np.testing.assert_allclose(aa[check['a']],aa[check['b']],rtol=check['rtol'],atol=check['atol'])
 ct=Counter(cl['labels'].tolist());cross=np.zeros((5,4),int)
 for l,a in zip(cl['labels'],f['assignments']):cross[int(l),int(a)]+=1
 summary=[]
 for i,c in enumerate(concepts):
  ids=set(cl['segment_id'][cl['labels']==c['cluster_id']].tolist());assert ids==set(c['initial_sorted_segments'])==set(c['pca_fitting_segments']);assert len(ids)==c['initial_members']==ct[c['cluster_id']]
  b=disc[f'concept_basis_{i:03d}'];assert b.shape==(c['subspace_dimension'],2048) and np.isfinite(b).all()
  summary.append({k:v for k,v in c.items() if not isinstance(v,list)}|dict(final_members=int((f['assignments']==i).sum())))
 assert [x['size'] for x in filters]==[ct[i] for i in range(5)]
 for e in read('training_example_index.json'):
  ci=next((i for i,c in enumerate(concepts) if c['concept_id']==e['concept_id']),3)
  allowed={s['segment_id'] for s in assign if s['assignment']==ci}
  assert set(e['top_prototypes'])<=allowed and set(e['random_final_members'])<=allowed
 counts=np.bincount(f['assignments'],minlength=4).tolist();pred=full['logits'].argmax(1)
 acc=dict(status='ACCEPTED_ENGINEERING_DISCOVERY',job_id='28889071',files_verified=len(index['files']),execution_commit=index['execution_commit'],images=101,held_out=0,sam_generated_masks=total_sam,postfilter_regions=len(raw),zero_features=int(zero.sum()),zero_fraction=float(zero.mean()),retained_features=len(kept),empty_effective_masks=sum(s['empty_effective'] for s in raw),images_without_regions=empty_images,images_without_nonzero_regions=zero_only,initial_clusters=len(filters),filtered_clusters=[x for x in filters if not x['retained']],concepts=summary,complement_dimension=int(disc['complement_basis'].shape[0]),final_assignment_counts=counts,cluster_to_final=cross.tolist(),completeness=result['completeness'],full_image_predicted_class_counts=np.bincount(pred,minlength=7).tolist(),mel_predicted=int((pred==4).sum()),zero_area_median=float(np.median(area_zero)),nonzero_area_median=float(np.median(area_nonzero)),segment_checks=checks,full_image_checks=read('training_full_reconstruction.json'),limitations=['101 fitting cases, not held-out concept evaluation','No calibrated clinical validity,patient independence or human semantic agreement','Concept and complement reconstruction tested on8saved samples each; FC reconstruction all423regions and101images','Raw-resolution image export not approved by automatic reviewer; display existing pre-SAM input only'])
 (BASE/'discovery-acceptance.json').write_text(json.dumps(acc,indent=2)+'\n')
 with (BASE/'discovery-concepts.csv').open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 np.savetxt(BASE/'cluster-to-final.csv',cross,delimiter=',',fmt='%d',header='C001,C002,C003,COMPLEMENT',comments='')
 print(json.dumps(acc,indent=2))
if __name__=='__main__':main()
