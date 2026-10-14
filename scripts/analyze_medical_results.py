"""Local-only acceptance of jobs28781914/15 and offline review of saved evidence.
No inference, fitting, remote processing, exclusions or new sample selection.
Run with humcd-upstream Python; inputs are immutable collected artifacts.
"""
import csv,gzip,hashlib,html,io,json,math,shutil,zipfile
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.special import softmax,logsumexp
from sklearn.metrics import confusion_matrix,roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/bunya/medical-results-20260922'
B=OUT/'bundle'; B.mkdir(exist_ok=True)
for d in ['figures','images','pairs']: (B/d).mkdir(exist_ok=True)
J=lambda p:json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def table(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def ck(ok,label):
 if not ok:raise ValueError(label)
 checks.append(label)
checks=[];commit='00f06dc549439db92e4842cb5ecd761f9b38bc43'
collections={j:ROOT/J(OUT/f'collection-{j}.json')['folder'] for j in ['28781914','28781915']}
H,M=collections.values();bindings=J(ROOT/'artifacts/bunya/medical-approved-20260922/config-bindings.json')
for job,p in collections.items():
 r=J(p/'collection.json');a=J(p/'artifacts.json');c=J(p/'actual_config.json');l=J(p/'launch_manifest.json')
 ck(r['status']=='COMPLETE' and not r['errors'] and a['status']==l['status']=='PASS',job+' collection and launcher PASS')
 ck(a['commit']==l['commit']==c['execution_commit']==commit,job+' full execution commit')
 for name,e in a['files'].items():ck((p/name).stat().st_size==e['bytes'] and sha(p/name)==e['sha256'],job+' artifact '+name)
 ck(sha(p/'actual_config.json')==bindings[job]['expected_actual_config_file_sha256'],job+' actual config file hash')
 ck(c==l['plan']['config'],job+' runtime config equals submitted plan')
 for name,e in J(p/'input_binding.json').items():ck(e['sha256']==e['expected_sha256']==c['protocol']['inputs'][name]['sha256'],job+' input '+name)
 ck(not (p/'job.err').read_text() and not (p/'anomalies.jsonl').read_text(),job+' no recorded stderr/worker warnings')
source=rows(M/'input_manifest.csv');by_split={s:[r for r in source if r['split']==s] for s in ['train','val','test']}
derm_collection=ROOT/'artifacts/bunya/next-phase-20260921/derm7pt-resumption/collections/28734478/20260921T051458.316863Z'
cases=rows(derm_collection/'cases.csv');caseby={int(r['row_index']):r for r in cases}
inventory={int(r['row_index']):r for r in J(derm_collection/'image_inventory.json') if r['role']=='derm'}
mel_names={'melanoma','melanoma (in situ)','melanoma (less than 0.76 mm)','melanoma (0.76 to 1.5 mm)','melanoma (more than 1.5 mm)','melanoma metastasis'}
r101={int(r['row_index']) for r in cases if r['official_split']=='test' and r['diagnosis'] in mel_names};ck(len(r101)==101,'R101 explicit author MEL grouping')
table(B/'R101-metadata.csv',[r for r in cases if int(r['row_index']) in r101])
with gzip.open(H/'all_candidates.csv.gz','rt',newline='') as f:pairs=list(csv.DictReader(f))
with np.load(H/'source_hashes.npz') as z:hp=z['phash'];hd=z['dhash'];ids=z['image_id']
si=J(H/'source_index.json');dh=J(H/'derm_hashes.json')
ck(si==sorted(source,key=lambda r:r['image_id']),'H source index exact manifest join')
ck(list(ids)==[r['image_id'] for r in si] and len(si)==10015 and len(dh)==1011,'H cache shapes and IDs')
index={r['image_id']:i for i,r in enumerate(si)};groups=defaultdict(list)
for n,r in enumerate(pairs):
 d=int(r['derm_row_index']);s=si[index[r['source_image_id']]];v=inventory[d]
 ck(all(r['source_'+k]==s[k] for k in ['split','array_row','lesion_id','label']),f'H source mapping {n}')
 ck(r['derm_case_num']==str(v['case_num']) and r['derm_path'].endswith('/images/'+v['relative_path']),f'H derm mapping {n}')
 r.update(pair_id=f"H-D{v['case_num']}-{s['image_id']}",R101=d in r101,original_pair_row=n+1,first200=n<200)
 groups[d].append(r)
ck(rows(H/'first_review_queue.csv')==[{k:r[k] for k in rows(H/'first_review_queue.csv')[0]} for r in pairs[:200]],'H first200 exact original order')
# Independent cached-distance replay covers every planned comparison; no new hashes or predictions.
pop=np.array([bin(i).count('1') for i in range(256)],dtype=np.uint8)
names=['rot0','rot90','rot180','rot270','mirror_rot0','mirror_rot90','mirror_rot180','mirror_rot270']
for d in dh:
 k=int(d['row_index']);pp=np.array(d['phash'],dtype=np.uint8);dd=np.array(d['dhash'],dtype=np.uint8)
 dp=pop[pp[:,None,:]^hp[None,:,:]].sum(axis=2);ds=pop[dd[:,None,:]^hd[None,:,:]].sum(axis=2)
 pm,dm=dp.min(0),ds.min(0)
 chosen=sorted(set(np.flatnonzero((pm<=6)|(dm<=6)))|{int(pm.argmin()),int(dm.argmin())})
 rr=groups[k];ck([index[r['source_image_id']] for r in rr]==chosen,f'H complete candidate union row{k}')
 for r in rr:
  i=index[r['source_image_id']]
  ck(int(r['phash_distance'])==pm[i] and int(r['dhash_distance'])==dm[i] and r['phash_orientation']==names[dp[:,i].argmin()] and r['dhash_orientation']==names[ds[:,i].argmin()],f'H distance/orientation {r["pair_id"]}')
  ck((r['threshold_candidate']=='True')==(min(pm[i],dm[i])<=6) and (r['nearest_phash']=='True')==(i==pm.argmin()) and (r['nearest_dhash']=='True')==(i==dm.argmin()),f'H flags {r["pair_id"]}')
hr=J(H/'result.json');ck(hr['candidates']==len(pairs)==5913 and hr['pair_count']==10015*1011 and J(H/'medical_progress.json')['derm_complete']==1011,'H final counts/stage')
# Local archives already byte-verified this session; reuse their identity receipts.
npz=Path('/mnt/c/Users/uqcche38/Downloads/dermamnist_corrected_224.npz');zp=Path('/mnt/c/Users/uqcche38/Downloads/release_v0.zip')
# Independently check exact-match keys and export source arrays losslessly, without new inference.
def pixelkey(a):return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()
keys=set();wanted={r['source_image_id'] for r in pairs};assets=[]
with np.load(npz,allow_pickle=False) as z:
 for split,rr in by_split.items():
  arr=z[split+'_images'];ck(np.array_equal(z[split+'_labels'].reshape(-1),[int(r['label']) for r in rr]),'Local source labels '+split)
  for r,a in zip(rr,arr):
   keys.add(pixelkey(a))
   if r['image_id'] in wanted:
    dest=B/'images'/('S-'+r['image_id']+'.png')
    if not dest.exists():Image.fromarray(a).save(dest)
    assets.append(dict(asset=str(dest.relative_to(B)),role='source',image_id=r['image_id'],row=r['array_row'],split=split,sha256=sha(dest),pixel_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
for d in dh:
 ck(d['native_key'] not in keys and d['canonical_key'] not in keys,'H independently verified zero exact matches '+str(d['row_index']))
ck(all(r['native_decoded_equal']==r['canonical_224_equal']=='False' for r in pairs),'H exact flags consistent')
with zipfile.ZipFile(zp) as z:
 for d,v in inventory.items():
  raw=z.read('release_v0/images/'+v['relative_path']);ck(hashlib.sha256(raw).hexdigest()==v['sha256'],'Local derm archive member '+str(d))
  dest=B/'images'/('D-'+str(d)+Path(v['relative_path']).suffix);dest.write_bytes(raw)
  assets.append(dict(asset=str(dest.relative_to(B)),role='derm',image_id=v['case_num'],row=d,split=v['split'],sha256=sha(dest),pixel_sha256=v.get('pixel_sha256','')))
table(B/'image-assets.csv',assets)
htable=[]
for cohort,rr in [('ALL1011',pairs),('R101',[r for r in pairs if r['R101']])]:
 for kind in ['all_candidates','threshold_candidate','nearest_only','exact']:
  ss=[r for r in rr if kind=='all_candidates' or (kind=='threshold_candidate' and r['threshold_candidate']=='True') or (kind=='nearest_only' and r['threshold_candidate']=='False') or (kind=='exact' and (r['native_decoded_equal']=='True' or r['canonical_224_equal']=='True'))]
  for split in ['ALL','train','val','test']:
   tt=[r for r in ss if split=='ALL' or r['source_split']==split]
   htable.append(dict(cohort=cohort,type=kind,source_split=split,pairs=len(tt),derm_cases=len({r['derm_row_index'] for r in tt}),source_images=len({r['source_image_id'] for r in tt}),source_lesions=len({r['source_lesion_id'] for r in tt})))
table(B/'H-counts.csv',htable);table(B/'H-all-pairs.csv',pairs)
table(B/'H-review-template.csv',[dict(pair_id=r['pair_id'],R101=r['R101'],source_split=r['source_split'],status='UNREVIEWED',visual_relation='',reviewer='',review_date='',evidence='',suggested_action='NO_AUTOMATIC_CHANGE') for r in pairs])
# M exact update/order/selection and independently recomputed saved-prediction metrics.
epochs=[json.loads(s) for s in (M/'epochs.jsonl').read_text().splitlines()];fr=J(M/'frozen_classifier.json');mp=J(M/'actual_config.json')['protocol'];mr=J(M/'result.json')
ck([r['epoch'] for r in epochs]==list(range(1,51)),'M 50 completed epochs')
best=-np.inf;be=None
for e in epochs:
 n=e['epoch'];order=np.load(M/f'train_order_epoch_{n:03d}.npy')
 ck(np.array_equal(np.sort(order),np.arange(8215)),f'M epoch{n} exact permutation')
 ck(e['optimizer_steps']==129*n and e['image_visits']==8215*n,f'M epoch{n} cumulative workload')
 improved=e['validation']['macro_ovr_auc']>best
 if improved:best=e['validation']['macro_ovr_auc'];be=n
 ck(e['selected']==improved and e['best_epoch']==be and e['best_score']==best,f'M epoch{n} selection history')
ck(fr['selected_epoch']==be==10 and sha(M/'best/model.pt')==fr['checkpoint_sha256'],'M frozen epoch10 checkpoint identity')
ck(J(M/'last/identity.json')['epoch']==50 and J(M/'best/identity.json')['epoch']==10 and not fr['test_outcomes_used_for_selection'],'M last50/best10/test isolation record')
classes=['AKIEC','BCC','BKL','DF','MEL','NV','VASC'];metrics={};cmaps={};classrows=[];predrows=[]
for split,file in [('val','best_validation.npz'),('test','final_test.npz')]:
 with np.load(M/file) as z:a={k:z[k] for k in z.files}
 rr=by_split[split];y=a['labels'].reshape(-1);logits=a['logits'];prob=softmax(logits,axis=1);pred=logits.argmax(1);cm=confusion_matrix(y,pred,labels=range(7));cmaps[split]=cm
 ck(np.isfinite(logits).all() and logits.shape==(len(rr),7) and np.array_equal(a['array_row'],np.arange(len(rr))) and list(a['image_id'])==[r['image_id'] for r in rr] and list(a['lesion_id'])==[r['lesion_id'] for r in rr] and np.array_equal(y,[int(r['label']) for r in rr]),'M '+split+' complete identity and finite predictions')
 auc=[float(roc_auc_score(y==i,prob[:,i])) for i in range(7)];rec=np.diag(cm)/cm.sum(1)
 metrics[split]=dict(n=len(rr),accuracy=float(np.trace(cm)/len(rr)),balanced_accuracy=float(rec.mean()),macro_ovr_auc=float(np.mean(auc)),cross_entropy=float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y])),confusion_matrix=cm.tolist())
 recorded=epochs[9]['validation'] if split=='val' else mr['test']
 for k in ['accuracy','balanced_accuracy','macro_ovr_auc','cross_entropy']:ck(np.isclose(metrics[split][k],recorded[k],rtol=1e-5,atol=1e-6),'M independent metric '+split+' '+k)
 for i,name in enumerate(classes):
  tp=int(cm[i,i]);fn=int(cm[i].sum()-tp);fp=int(cm[:,i].sum()-tp);tn=len(rr)-tp-fn-fp
  classrows.append(dict(split=split,label=i,class_name=name,images=int(cm[i].sum()),lesions=len({r['lesion_id'] for r in rr if int(r['label'])==i}),predicted=int(cm[:,i].sum()),TP=tp,FN=fn,FP=fp,TN=tn,recall=tp/(tp+fn),precision=tp/(tp+fp) if tp+fp else 'NOT_APPLICABLE_NO_PREDICTIONS',f1=2*tp/(2*tp+fp+fn),specificity=tn/(tn+fp),ovr_auc=auc[i]))
 for i,r in enumerate(rr):predrows.append(dict(split=split,image_id=r['image_id'],lesion_id=r['lesion_id'],array_row=i,label=int(y[i]),prediction=int(pred[i]),**{f'p_{n}':float(prob[i,j]) for j,n in enumerate(classes)}))
 table(B/f'M-confusion-{split}.csv',[dict(true_class=classes[i],**{n:int(cm[i,j]) for j,n in enumerate(classes)}) for i in range(7)])
with np.load(M/'adapter_check_arrays.npz') as z:
 ck(list(z['image_id'])==[r['image_id'] for r in by_split['val'][:8]],'M adapter exact first8 fixed validation IDs')
 for c in J(M/'adapter_checks.json')['checks']:ck(c['pass_check'] and np.isfinite(z[c['a']]).all() and np.allclose(z[c['a']],z[c['b']],rtol=c['rtol'],atol=c['atol']),'M cached adapter '+c['a']+'/'+c['b'])
 import torch
 state=torch.load(M/'best/model.pt',map_location='cpu',weights_only=True)
 ck(state['fc.weight'].shape==(7,2048) and state['fc.bias'].shape==(7,) and all(torch.isfinite(t).all() for t in state.values()),'M checkpoint head dimensions/all finite')
 calc=z['features']@state['fc.weight'].numpy().T+state['fc.bias'].numpy()
 ck(np.allclose(calc,z['head_algebra'],rtol=1e-5,atol=1e-5),'M independent cached head algebra')
 algebra_max=float(np.max(np.abs(calc-z['head_algebra'])))
table(B/'M-class-metrics.csv',classrows);table(B/'M-image-predictions.csv',predrows)
table(B/'M-epoch-metrics.csv',[dict(epoch=e['epoch'],lr=e['lr'],train_CE=e['training_cross_entropy'],val_CE=e['validation']['cross_entropy'],val_macro_AUC=e['validation']['macro_ovr_auc'],val_accuracy=e['validation']['accuracy'],val_balanced_accuracy=e['validation']['balanced_accuracy'],selected=e['selected'],updates=e['optimizer_steps'],visits=e['image_visits']) for e in epochs])
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
fig,ax=plt.subplots(1,2,figsize=(13,4.6),constrained_layout=True);x=[e['epoch'] for e in epochs]
ax[0].plot(x,[e['training_cross_entropy'] for e in epochs],label='Training CE (augmented)');ax[0].plot(x,[e['validation']['cross_entropy'] for e in epochs],label='Validation CE')
for k,label in [('macro_ovr_auc','Macro OVR AUC'),('accuracy','Accuracy'),('balanced_accuracy','Balanced accuracy')]:ax[1].plot(x,[e['validation'][k] for e in epochs],label='Validation '+label)
for a in ax:a.axvline(10,color='black',linestyle='--',label='Selected epoch 10');a.set_xlabel('Completed epoch');a.legend(fontsize=8);a.grid(alpha=.2)
ax[0].set_ylabel('Cross entropy');ax[1].set_ylabel('Score');fig.suptitle('M | fixed seed 43, 50 epochs, batch 64 | test used only after checkpoint freeze')
for ext in ['png','pdf']:fig.savefig(B/'figures'/('M-training-curves.'+ext),dpi=240)
plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
for a,s in zip(ax,['val','test']):
 cm=cmaps[s];a.imshow(cm/cm.sum(1)[:,None],vmin=0,vmax=1,cmap='Blues');a.set_xticks(range(7),classes,rotation=45);a.set_yticks(range(7),classes);a.set_xlabel('Predicted class');a.set_ylabel('True class');a.set_title(s.upper()+f' | n={cm.sum()} | row-normalized color; count (% of row)')
 for i in range(7):
  for j in range(7):a.text(j,i,f'{cm[i,j]}\n({100*cm[i,j]/cm[i].sum():.1f}%)',ha='center',va='center',fontsize=8,color='white' if cm[i,j]/cm[i].sum()>.6 else 'black')
for ext in ['png','pdf']:fig.savefig(B/'figures'/('M-confusion-matrices.'+ext),dpi=240)
plt.close(fig)
accept=dict(generated_at=datetime.now(timezone.utc).isoformat(),execution_commit=commit,status='PASS_ENGINEERING_WITH_RESEARCH_LIMITATIONS',H=dict(status='PASS_BOUNDED_RETRIEVAL_MANUAL_REVIEW_PENDING',pair_comparisons=10125165,orientations_per_hash=81001320,candidate_pairs=5913,exact_native=0,exact_canonical224=0,threshold_pairs=4291,threshold_cases=366,R101_threshold_pairs=214,R101_threshold_cases=20),M=dict(status='PASS_ENGINEERING_NOT_MEDICAL_CERTIFICATION',epochs=50,updates=6450,image_visits=410750,selected_epoch=10,checkpoint_sha256=fr['checkpoint_sha256'],metrics=metrics,independent_head_max_abs=algebra_max),checks=len(checks),new_predictions=0,new_jobs=0,medical_discovery_authorized=False,external_inference_authorized=False)
dump(OUT/'acceptance.json',accept);dump(OUT/'acceptance-checks.json',checks);dump(B/'acceptance.json',accept)
print(json.dumps(accept,indent=2))
# Offline static pair pages: every retained pair, plus explicit R101 threshold-priority view.
CSS='body{font:17px system-ui,sans-serif;color:#142c3b;background:#f5f7fa;max-width:1300px;margin:auto;padding:28px}a{color:#006a80}header,article,section{background:white;padding:22px;margin:20px 0;border:1px solid #d7e0e6;border-radius:10px}h1{font-size:32px}h2{font-size:23px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}.pair img{width:100%;height:340px;object-fit:contain;background:#edf1f3}small{overflow-wrap:anywhere}textarea{width:98%;min-height:62px}button,select,input{font:inherit;margin:8px;padding:6px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ccd8df;text-align:left;padding:8px}figure{margin:0}img.chart{width:100%}@media(max-width:750px){.pair{grid-template-columns:1fr}.pair img{height:280px}}'
JS="""const key='MED26-H-overlap-review-v1'; let saved={}; try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){};document.querySelectorAll('article[data-pair]').forEach(a=>{const id=a.dataset.pair,s=saved[id];if(s){a.querySelector('select').value=s.verdict;a.querySelector('textarea').value=s.note;}a.querySelector('button').onclick=()=>{saved[id]={pair_id:id,verdict:a.querySelector('select').value,note:a.querySelector('textarea').value,recorded_at:new Date().toISOString(),not_patient_identity:true};try{localStorage.setItem(key,JSON.stringify(saved));a.querySelector('.saved').textContent='已保存本机 / Saved locally'}catch(e){a.querySelector('.saved').textContent='浏览器存储失败，请导出 / Storage failed: export now'}}});document.querySelector('#export').onclick=()=>{let b=new Blob([JSON.stringify(saved,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='H-manual-review.json';a.click();};"""
(B/'style.css').write_text(CSS);(B/'review.js').write_text(JS)
def page(title,body,up=''):
 return '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><link rel="stylesheet" href="'+up+'style.css"><body data-dataset="medical-hm-results-20260922"><script id="concept-ids" type="application/json">[]</script>'+body+'</body></html>'
def card(r):
 d=int(r['derm_row_index']);v=inventory[d];pid=r['pair_id'];src='../images/S-'+r['source_image_id']+'.png';der='../images/D-'+str(d)+Path(v['relative_path']).suffix
 return f'''<article id="{pid}" data-pair="{pid}"><h2>{pid}</h2><p>候选 / Candidate · R101: {r['R101']} · pHash {r['phash_distance']} ({r['phash_orientation']}) / dHash {r['dhash_distance']} ({r['dhash_orientation']}) · ≤6: {r['threshold_candidate']}</p><div class="pair"><figure><a href="{src}" target="_blank"><img loading="lazy" src="{src}" alt="Source {r['source_image_id']}"></a><figcaption>源域 / Source {r['source_split']} · {r['source_image_id']}<br>array row {r['source_array_row']} · lesion {r['source_lesion_id']} · label {classes[int(r['source_label'])]}</figcaption></figure><figure><a href="{der}" target="_blank"><img loading="lazy" src="{der}" alt="Derm7pt case {v['case_num']}"></a><figcaption>Derm7pt {r['derm_split']} · case {v['case_num']} · metadata row {d}<br>{html.escape(caseby[d]['diagnosis'])}<br>{html.escape(v['relative_path'])}</figcaption></figure></div><p>原始方向；点击查看原尺寸。旋转/镜像仅用于检索，未改变模型输入。<br>Native orientation; click for native size. D4 orientations were retrieval-only. Exact native/canonical match: False / False.</p><small>原始候选行 / Original candidate row {r['original_pair_row']} · first200: {r['first200']} · 患者身份未知 / patient identity unknown</small><p>人工审阅 / Manual review</p><select><option>UNREVIEWED / 未审阅</option><option>VISUALLY_DISTINCT / 视觉明显不同</option><option>POSSIBLE_SAME_IMAGE / 疑似同源图片</option><option>UNCERTAIN / 无法判断</option></select><textarea placeholder="Reviewer, date, visual evidence, confidence and suggested handling; no inferred patient ID / 审阅人、日期、依据、置信及建议，不推断患者身份"></textarea><button>保存草稿 / Save draft</button><span class="saved"></span></article>'''
links=[]
for prefix,rr,title in [('R101',[r for r in pairs if r['R101'] and r['threshold_candidate']=='True'],'R101 阈值候选 / threshold candidates'),('all',pairs,'全部候选 / all retained candidates')]:
 pages=math.ceil(len(rr)/50)
 for i in range(pages):
  name=f'{prefix}-{i+1:03d}.html';nav=' · '.join(f'<a href="{prefix}-{k+1:03d}.html">{k+1}</a>' for k in range(pages))
  body=f'<header><a href="../index.html">返回结果 / Results</a><h1>{title}</h1><p>{len(rr)} pairs · page {i+1}/{pages}. 原始 metadata 行 / source ID 顺序，不按“最像”筛选。Original metadata-row/source-ID order; no visual cherry-picking. 未审阅不等于重复或独立 / unreviewed means neither duplicate nor independent.</p><button id="export">导出全部草稿 / Export saved reviews</button><nav>{nav}</nav></header>'
  body+=''.join(card(r) for r in rr[i*50:(i+1)*50])+f'<nav>{nav}</nav><script src="../review.js"></script>'
  (B/'pairs'/name).write_text(page(title,body,'../'))
  if i==0:links.append((title,'pairs/'+name))
# Human-readable summary stays bilingual, static and meaningful without JavaScript.
def csvhtml(rr):
 return '<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in rr[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r.values())+'</tr>' for r in rr)+'</tbody></table>'
body='''<header><h1>医学 H/M 结果与验收<br>Medical H/M results & acceptance</h1><p>2026-09-22 · Private / 私有 · H28781914 / M28781915 · 完整执行版本 / full execution commit<br><code>'''+commit+'''</code></p><p>工程验收通过，研究限制保留；无新实验、无自动排除或重训。<br>Engineering acceptance passed with research limitations; no new experiments, exclusions or retraining.</p></header>'''
body+='''<section><h2>H：完成计划检索，身份判断仍待人工 / Retrieval complete, identity review pending</h2><p>10,015 × 1,011 = 10,125,165 图片对，每种hash检查8个方向，共81,001,320次方向比较。原尺寸解码像素与224 bicubic规范化精确匹配均为0；原始JPEG字节比较不适用NPZ。5,913候选含4,291阈值候选及1,622仅最近邻补充，不能将候选数解释为重复数。<br>All planned pairs and eight orientations per hash were completed and independently replayed from cache. Zero native/canonical exact matches; raw encoded-byte comparison is inapplicable to NPZ. 5,913 candidates comprise4,291 threshold hits and1,622 nearest-only supplements; these are not duplicate counts.</p><p>R101有214阈值候选对，涉及20/101病例；源train/val/test分别180/12/22对。尚未人工确认，不改动R101；不能宣称患者独立或无数据重叠。<br>R101:214 threshold pairs across20/101cases;180/12/22 involve source train/val/test. No human confirmation or R101 changes; neither patient independence nor absence of overlap is established.</p>'''
body+=csvhtml([r for r in htable if r['type']=='threshold_candidate'])
body+='<p>'+ ' · '.join(f'<a href="{p}">{t}</a>' for t,p in links)+'</p><p><a href="H-all-pairs.csv">全部可追溯候选 / All candidate pairs</a> · <a href="H-review-template.csv">审阅模板 / Review template</a> · <a href="image-assets.csv">图片哈希与原始身份 / Image identity</a> · <a href="H-counts.csv">所有计数 / Counts</a></p><p>所有5913对均提供并排原图；优先视图包含R101全部214阈值候选，其他最近邻仍保留在全集。阈值是预定操作规则，未校准敏感性/特异性。<br>All5913pairs have native side-by-side images. Priority view includes all214R101threshold pairs; nearest-only pairs remain in the full set. Threshold is prespecified operational retrieval, not calibrated sensitivity/specificity.</p></section>'
body+='''<section><h2>M：解释一个固定分类器 / One fixed classifier</h2><p>8215训练 / 573验证 / 1227测试。50轮、6450更新、410750图片访问；训练seed43，非跨seed稳定性实验。validation macro OVR AUC选中第10轮，随后冻结权重并进行test；末轮checkpoint另存。FP32、无AMP、cuDNN TF32开、matmul TF32关。<br>All training completed; validation macro OVR AUC selected epoch10. The checkpoint was frozen before final test; epoch50 is separately retained. FP32/no AMP, cuDNN TF32 enabled, matmul TF32 disabled; one training seed43.</p><p>测试accuracy89.00%，balanced accuracy59.70%，macro AUC0.96867。MEL为50/70检出（61病灶），recall71.43%、precision56.18%、AUC0.94410；20张MEL误分，39张非MEL误报。DF验证0/5、测试0/8，无一张被最终预测为DF，尽管DF AUC0.95939。<br>Test accuracy89.00%, balanced accuracy59.70%, macro AUC0.96867. MEL:50/70 recalled across61lesions,71.43%recall,56.18%precision,0.94410AUC;20false negatives and39false positives. DF recall0/5validation and0/8test; no DF argmax predictions despite test OVR AUC0.95939. Ranking and multiclass decisions measure different behavior.</p><p>高总体准确率不能代表小类别可靠；数值/适配通过不等于医学适用性。没有患者ID，不声称患者独立。<br>Overall accuracy does not establish minority-class reliability. Numerical/adapter checks do not certify medical utility; patient independence is unknown.</p><img class="chart" src="figures/M-training-curves.png" alt="Training loss and validation curves"><p>第10→50轮train CE 0.30282→0.01897，val CE 0.30058→0.43641，val AUC 0.98160→0.97182；与后期泛化退化相容，不证明唯一原因。训练accuracy未保存，不虚构曲线。<br>Epoch10→50: train CE0.30282→0.01897, val CE0.30058→0.43641 and val AUC0.98160→0.97182. Consistent with later overfitting/generalization degradation, not a causal proof. Training accuracy was not recorded.</p><img class="chart" src="figures/M-confusion-matrices.png" alt="Validation and test confusion matrices">'''
body+=csvhtml([r for r in classrows if r['split']=='test'])
body+='''<p><a href="M-class-metrics.csv">七类完整指标与分母 / Seven-class metrics</a> · <a href="M-image-predictions.csv">逐图已保存预测 / Saved image predictions</a> · <a href="M-epoch-metrics.csv">50轮轨迹 / 50epoch trace</a> · <a href="figures/M-training-curves.pdf">曲线PDF</a> · <a href="figures/M-confusion-matrices.pdf">矩阵PDF</a></p><p>普通与masked适配只验证固定validation前8图的同batch全1mask及线性head代数，不能代替未来SAM局部mask检查。无新预测。<br>Adapter evidence covers the first8fixed validation images, same-batch all-ones masks and saved linear-head algebra; it does not validate future local SAM masks. No new predictions.</p></section><section><h2>范围、缺失与下一步 / Scope and next step</h2><p>28781904/14/15均COMPLETED0:0；耗时6秒/88秒/782秒。M主机MaxRSS约3.20GiB；Slurm未记录GPU利用率/显存峰值，不能由主机RSS推断。两个作业stderr与worker异常文件为空，研究限制由验收另外登记。<br>All three jobs COMPLETED0:0 in6/88/782s. M host MaxRSS≈3.20GiB; GPU utilization/peak VRAM unavailable in scheduler output. Both job stderr and worker warning files are empty; research limitations are separately logged.</p><p>医学概念发现和Derm7pt外部推理尚未执行、未获授权；现有R101/S仅是已批准评价协议。H候选人工审阅不能被自动转为排除规则。<br>Medical concept discovery and external inference remain unexecuted and unauthorized; R101/S approve protocols only. Manual overlap review does not automatically authorize exclusions.</p><p><a href="acceptance.json">机器可读验收 / Acceptance receipt</a> · <a href="provenance.json">来源和生成说明 / Provenance</a></p></section>'''
(B/'index.html').write_text(page('Medical H/M acceptance',body))
dump(B/'provenance.json',dict(generated_at=accept['generated_at'],script=str(Path(__file__).relative_to(ROOT)),script_sha256=sha(__file__),execution_commit=commit,collections={j:str(p.relative_to(ROOT)) for j,p in collections.items()},selection='All5913 retained H pairs, original derm-row/source-ID order; priority view all214 R101 threshold candidates; no randomness or exclusions',source_images='Exact NPZ224 RGB arrays exported lossless PNG; no upscale saved',derm_images='Original JPEG members byte-matched accepted inventory; no transformation',local_archives_receipts=['dermamnist_corrected_224.npz.local-reuse.json','release_v0.zip.local-reuse.json'],limitations=['Unreviewed similarity is not identity','Image IDs/lesion IDs are not patient IDs','No new model predictions','All comparisons image-level, no confidence intervals or seed-generalization claim']))
dump(B/'bundle-manifest.json',dict(id='medical-hm-results-20260922',dataset_id='medical-hm-results-20260922',entry='index.html',created_at=accept['generated_at'],files={str(p.relative_to(B)):sha(p) for p in sorted(B.rglob('*')) if p.is_file() and p.name!='bundle-manifest.json'}))
print('BUNDLE_COMPLETE',len(pairs),'pairs',len(assets),'assets')
