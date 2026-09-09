"""Slurm-only, read-only audit/export of existing discoveries for a private visual atlas."""
import os,sys,json,re,socket,hashlib,tarfile,shutil,traceback,warnings,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/scratch/user/uqcche38/hu-mcd')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def save(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def execute(plan):
 job=os.environ.get('SLURM_JOB_ID','');assert job.isdigit() and re.fullmatch('bun[0-9]{3}',socket.gethostname().split('.')[0])
 out=ROOT/'launches'/job/'visual-audit';out.mkdir(parents=True,exist_ok=False)
 save(out/'plan.json',plan);results=[];alerts=[]
 def warn(message,category,filename,lineno,file=None,line=None):
  alerts.append(dict(stage='audit/export',category=category.__name__,message=str(message),file=filename,line=lineno));save(out/'warnings.json',alerts)
 warnings.showwarning=warn;warnings.simplefilter('always')
 for item in plan['runs']:
  ident=str(item['job_id']);run=ROOT/'outputs/runs'/ident;cache=ROOT/'cache/runs'/ident
  row=dict(source_job=ident,expected_commit=item['commit'],status='RUNNING',images=[],splits={});stage=out/ident;stage.mkdir();save(out/'progress.json',dict(status='RUNNING',classes=results,current=ident))
  try:
   config=json.loads((run/'resolved_config.json').read_text());dataset=json.loads(Path(config['dataset_manifest']).read_text());summary=json.loads((run/'summary.json').read_text());tracked=json.loads((run/'run_manifest.json').read_text());release=Path(tracked['repo_root'])
   assert summary['status']=='PASS' and str(summary['run_id'])==ident and str(tracked['run_id'])==ident and tracked['git_commit']==item['commit'] and not tracked['git_dirty']
   assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=release,text=True).strip()==item['commit']
   assert sha(run/'resolved_config.json')==summary['resolved_config_sha256']==tracked['resolved_config_sha256']
   assert sha(config['dataset_manifest'])==config['dataset_manifest_sha256']
   core={n:sha(release/n) for n in plan['core_files']};assert core==plan['core_sha256']
   row.update(class_name=config['class_name'],code_commit=item['commit'],core_sha256=core,config_sha256=sha(run/'resolved_config.json'),dataset_sha256=sha(config['dataset_manifest']),source_cache=str(cache),model_identity={'resnet_sha256':config['resnet_checkpoint_sha256'],'sam_sha256':config['sam_checkpoint_sha256'],'model':config['model_name'],'precision':config['precision']})
   d=np.load(run/'scientific/discovery.npz',allow_pickle=False);K=summary['concepts'];bases=[d[k] for k in sorted(d.files) if k.startswith('concept_basis_')];labels=d['cluster_labels'];retained=d['retained_cluster_labels'];assert len(bases)==len(retained)==K
   assert all(np.isfinite(x).all() for x in bases+[d['complement_basis']]);assert len(set(retained.tolist()))==K
   row['cluster_sizes']={str(int(k)):int((labels==k).sum()) for k in np.unique(labels)};row['retained_cluster_labels']=retained.tolist();row['basis_shapes']=[list(x.shape) for x in bases];row['complement_shape']=list(d['complement_basis'].shape);row['outlier_count']=int(d['outlier_mask'].sum())
   (stage/'inputs').mkdir();(stage/'images').mkdir()
   for p in Path(config['source_dir']).iterdir():
    if p.is_file() and p.suffix in ['.json','.txt']:shutil.copy2(p,stage/'inputs'/p.name)
   seen=set()
   for split,short,n in [('training','train',400),('validation','validation',50)]:
    entries=dataset[split];assert len(entries)==n and len(tracked['input_files'][split])==n
    a=np.load(run/'scientific'/f'{split}.npz',allow_pickle=False);mapping=json.loads((run/'scientific'/f'{split}_segments.json').read_text());checks=json.loads((run/'scientific'/f'{split}_checks.json').read_text());assert checks['status']=='PASS'
    found=list((cache/f'activations_{short}').glob('*_acts.npy'));assert len(found)==1
    raw=np.load(found[0],allow_pickle=False);logits=np.load(str(found[0]).replace('_acts.npy','_logits.npy'),allow_pickle=False);keep=np.any(raw!=0,axis=1)
    assert np.isfinite(raw).all() and np.isfinite(logits).all() and np.array_equal(raw[keep],a['features']) and np.array_equal(logits[keep],a['logits'])
    scores=a['concept_activations'];assign=a['assignments'];assert scores.shape==(int(keep.sum()),K+1) and np.isfinite(scores).all() and np.array_equal(assign,scores.argmax(1))
    if split=='training':assert len(labels)==len(assign) and all((labels==v).sum()>=config['clustering']['min_cluster_size'] for v in retained)
    cursor=rr=0;rows=[];sizes=[];(stage/'images'/split).mkdir()
    for i,e in enumerate(entries):
     actual=Path(e['input_path']);original=Path(e['source']);assert sha(actual)==e['input_sha256'] and sha(original)==e['sha256'];assert e['input_sha256'] not in seen;seen.add(e['input_sha256'])
     assert tracked['input_files'][split][i]['path']==str(actual.resolve()) and tracked['input_files'][split][i]['sha256']==e['input_sha256']
     with Image.open(actual) as im:
      im.load();assert im.mode=='RGB';w,h=im.size;scale=min(1,config['max_shortest_side']/min(w,h));size=(int(w*scale),int(h*scale)) if scale<1 else (w,h)
      # Match upstream integer dimension construction exactly.
      if min(w,h)>config['max_shortest_side']:
       q=config['max_shortest_side'];size=(q,int(q*(h/w))) if h>w else (int(q*(w/h)),q)
      if e.get('compatibility_adjustment'):
       with Image.open(original) as ori:ori.load();assert ori.mode=='L' and ori.size==im.size and all(c.tobytes()==ori.tobytes() for c in im.split())
     mp=cache/f'segments_{short}'/(actual.name+'_sam.npy');m=np.load(mp,allow_pickle=False);assert m.ndim==3 and m.shape[1:]==(size[1],size[0]) and np.logical_or(m==0,m==1).all()
     pos=np.flatnonzero(keep[cursor:cursor+len(m)]);sizes.append(len(pos));shutil.copy2(actual,stage/'images'/split/actual.name)
     for local,rawindex in enumerate(pos):
      saved=mapping[rr];assert saved['image_index']==i and saved['segment_index']==local and saved['image_path']==str(actual) and saved['mask_sha256']==hashlib.sha256(m[rawindex].tobytes()).hexdigest()
      rows.append(dict(flat_index=rr,image_index=i,segment_index=local,raw_mask_index=int(rawindex),source_id=original.name,input_name=actual.name,mask_sha256=saved['mask_sha256']));rr+=1
     row['images'].append(dict(split=split,image_index=i,source_id=original.name,input_name=actual.name,original_size=[w,h],model_image_size=list(size),raw_segments=len(m),retained_segments=len(pos),zero_features=len(m)-len(pos),actual_sha256=e['input_sha256']))
     cursor+=len(m)
    assert cursor==len(raw) and rr==len(mapping) and sizes==summary['train_segments_per_image' if split=='training' else 'validation_segments_per_image']
    assert len(list((cache/f'segments_{short}').glob('*.npy')))==n
    save(stage/(split+'_row_mapping.json'),rows)
    row['splits'][split]=dict(images=n,raw_segments=len(raw),retained_segments=int(keep.sum()),zero_features=int((~keep).sum()),assignment_counts=np.bincount(assign,minlength=K+1).tolist(),masks_features_mapping='PASS',saved_reconstruction_checks=checks,reconstruction_scope='All FC logits plus 8 stored oblique-decomposition samples; original run checks reused')
   row['cache_inventory']=[dict(path=str(p.relative_to(cache)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(cache.rglob('*')) if p.is_file()]
   row['status']='PASS';save(stage/'audit.json',row)
   # Archive immutable results, raw caches and private images; never include model checkpoints.
   archive=out/(ident+'.tar.gz')
   with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
    tar.add(stage,arcname=ident)
    for p in run.iterdir():
     if p.is_file() or p.name=='scientific':tar.add(p,arcname=ident+'/run/'+p.name)
    tar.add(cache,arcname=ident+'/cache')
    for p in (ROOT/'outputs/workstreams'/ident).glob('*'):
     if p.is_file():tar.add(p,arcname=ident+'/worker/'+p.name)
    for p in [ROOT/'logs'/('reference-'+ident+'.out'),ROOT/'logs'/('reference-'+ident+'.err'),ROOT/'launches'/ident/'launch_manifest.json']:
     if p.exists():tar.add(p,arcname=ident+'/logs/'+p.name)
   exports=[]
   if archive.stat().st_size>192*1024**2:
    with archive.open('rb') as f:
     index=0
     while True:
      chunk=f.read(192*1024**2)
      if not chunk:break
      p=out/(archive.name+'.part%03d'%index);p.write_bytes(chunk);exports.append(dict(file=p.name,sha256=sha(p),bytes=len(chunk)));index+=1
   else:exports=[dict(file=archive.name,sha256=sha(archive),bytes=archive.stat().st_size)]
   row['exports']=exports
  except Exception as e:
   row.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());save(stage/'audit.json',row)
  results.append(row);save(out/'progress.json',dict(status='RUNNING',classes=results))
 report=dict(status='PASS' if all(x['status']=='PASS' for x in results) else 'COMPLETED_WITH_FAILURES',audit_job=job,host=socket.gethostname(),analysis_commit=plan['analysis_commit'],worker_sha256=plan['worker_sha256'],classes=results,warnings=alerts)
 save(out/'audit.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['classes','warnings']}))
if __name__=='__main__':execute(json.loads(Path(sys.argv[1]).read_text()))
