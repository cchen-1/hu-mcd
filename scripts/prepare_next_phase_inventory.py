"""Local accepted-cache inventory for protocol decisions; never runs a model.

Candidate geometric previews are NOT sensitivity experiments or approval.
Writes only a new output directory; keeps all source caches read-only.
"""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from classes import ImageClass,SegmentClass,ConceptDatasetClass
from hpc.final_mask_intervention import intervene

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def csvwrite(p,rows):
 with p.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();O=args.output;O.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(2);T=ROOT/'artifacts/bunya/ten-class-review';audit=read(T/'audit-final.json');assert audit['status']=='PASS'
 classes=[];regions=[];conditions=[('identity',0),('erosion',1),('erosion',2),('dilation',1),('dilation',2)];bindings=[]
 for c in audit['classes']:
  start=time.time();f=T/'evidence'/c['source_job'];cfg=read(f/'run/resolved_config.json');dc=read(f/'run/scientific/discovery.json');s=read(f/'run/summary.json');assert sha(f/'run/resolved_config.json')==c['config_sha256'];assert cfg['batch_size']==8
  cache=f/'cache/activations_validation';fp=next(cache.glob('*_acts.npy'));entry=next(e for e in c['cache_inventory'] if e['path'].endswith(fp.name));assert sha(fp)==entry['sha256'];features=np.load(fp,allow_pickle=False)
  assert features.shape==(c['splits']['validation']['raw_segments'],2048) and np.isfinite(features).all()
  mapping=read(f/'validation_row_mapping.json');retmap={(m['image_index'],m['raw_mask_index']):m for m in mapping}
  with np.load(f/'run/scientific/validation.npz',allow_pickle=False) as z:retained=z['features'];scores=z['concept_activations'];assignment=z['assignments']
  rawcursor=0;empty={f'{op}{r}':0 for op,r in conditions};batches=[];no_valid=0;valid_per_image=[]
  inputs=[x for x in c['images'] if x['split']=='validation'];assert len(inputs)==50
  inv={x['path']:x for x in c['cache_inventory']}
  for image_index,row in enumerate(inputs):
   assert row['image_index']==image_index
   imagefile=f/'images/validation'/row['input_name'];assert sha(imagefile)==row['actual_sha256']
   maskpath=f/'cache/segments_validation'/(row['input_name']+'_sam.npy');assert sha(maskpath)==inv['segments_validation/'+maskpath.name]['sha256']
   masks=np.load(maskpath,allow_pickle=False);assert len(masks)==row['raw_segments'];im=ImageClass(str(imagefile),cfg['max_shortest_side']);im.segments=[SegmentClass(mask,im) for mask in masks]
   ds=ConceptDatasetClass([im],dc['model_default_cfg'],0,True,-1,.25);valid=0
   for j,rawmask in enumerate(masks):
    # Original dataset performs the full accepted preprocessing; no CNN forward.
    effective=ds[j][1].numpy()[0].astype(bool);assert effective.shape==tuple(dc['model_default_cfg']['input_size'][1:])
    zero=bool(np.all(features[rawcursor]==0));key=(image_index,j);m=retmap.get(key);assert (m is None)==zero
    if m is not None:
     assert sha_bytes(rawmask)==m['mask_sha256'];np.testing.assert_array_equal(features[rawcursor],retained[m['flat_index']]);valid+=int(effective.any())
    rec=dict(class_name=c['class_name'],source_job=c['source_job'],region_id=f"HU-MCD-{c['class_name']}-validation-I{image_index:03d}-M{j:03d}",image_id=row['source_id'],image_index=image_index,raw_mask_index=j,raw_feature_row=rawcursor,input_sha256=row['actual_sha256'],raw_mask_sha256=sha_bytes(rawmask),effective_mask_sha256=sha_bytes(effective),feature_sha256=sha_bytes(features[rawcursor]),baseline_zero_feature=zero,baseline_effective_pixels=int(effective.sum()),retained_row=None if m is None else m['flat_index'],baseline_assignment=None if m is None else int(assignment[m['flat_index']]),baseline_margin=None if m is None else float(np.sort(scores[m['flat_index']])[-1]-np.sort(scores[m['flat_index']])[-2]))
    for op,r in conditions:
     changed=intervene(effective,op,r);name=f'{op}{r}';area=int(changed.sum());rec[name+'_pixels']=area;empty[name]+=int(area==0)
    regions.append(rec);rawcursor+=1
   valid_per_image.append(valid);no_valid+=int(valid==0)
  assert rawcursor==len(features)
  batches=[list(range(i,min(i+8,rawcursor))) for i in range(0,rawcursor,8)]
  z=sum(r['baseline_zero_feature'] for r in regions if r['class_name']==c['class_name']);assert z==c['splits']['validation']['zero_features']
  info=dict(class_name=c['class_name'],source_job=c['source_job'],images=50,raw_regions=rawcursor,zero_features=z,valid_baseline_regions=sum(valid_per_image),images_without_valid_baseline=no_valid,grid=list(dc['model_default_cfg']['input_size'][1:]),conditions=5,records=rawcursor*5,geometry_only_empty_preview=empty,maximum_cnn_nonempty_rows=sum(rawcursor-n for n in empty.values()),prior_validation_feature_seconds=s['stage_elapsed_seconds']['validation_activations'],prior_validation_assignment_plus_display_seconds=s['stage_elapsed_seconds']['validation_assignment_and_prototypes'],local_preparation_seconds=time.time()-start)
  classes.append(info);binding=dict(class_name=c['class_name'],source_job=c['source_job'],source_cache=c['source_cache'],source_commit=c['code_commit'],config_sha256=c['config_sha256'],dataset_sha256=c['dataset_sha256'],basis_sha256=sha(f/'run/scientific/discovery.npz'),raw_features_sha256=sha(fp),model_identity=c['model_identity'],inference_batches=batches,grid=info['grid'],norm_batch=False,reference_scope='Accepted discovery validation score path; zero rows excluded from valid analysis, separately retained');bindings.append(binding)
  print(c['class_name'],rawcursor,z,empty,flush=True)
 csvwrite(O/'A-region-ledger.csv',regions);(O/'A-bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
 (O/'A-inventory.json').write_text(json.dumps(dict(status='PASS',scope='Input/cache identity and proposed geometry-only workload; no model or perturbation result',protocol_approved=False,classes=classes,total_images=500,total_regions=len(regions),total_region_conditions=len(regions)*5,zero_features=sum(x['zero_features'] for x in classes),total_maximum_cnn_nonempty_rows=sum(x['maximum_cnn_nonempty_rows'] for x in classes),region_ledger_sha256=sha(O/'A-region-ledger.csv'),bindings_sha256=sha(O/'A-bindings.json')),indent=2)+'\n')
def sha_bytes(a):return hashlib.sha256(a.tobytes()).hexdigest()
if __name__=='__main__':main()
