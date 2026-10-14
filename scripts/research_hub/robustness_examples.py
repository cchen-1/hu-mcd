"""Compact paired mask examples from accepted caches only. No model execution."""
from pathlib import Path
import csv,json,hashlib,re,os,html
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[2];T=R/'artifacts/bunya/ten-class-review';A=R/'artifacts/bunya/robustness-completion-20260924';O=R/'artifacts/report-robustness-visual-20260924'
CONDS=['identity0','erosion1','erosion2','dilation1','dilation2'];LABELS=['Original mask','Erode 1 px','Erode 2 px','Dilate 1 px','Dilate 2 px'];SEED=20260921
J=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def zload(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvout(p,rr):
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def bi(z,e):return '<span data-l="zh">'+html.escape(z)+'</span><span data-l="en" hidden>'+html.escape(e)+'</span>'
def para(z,e):return '<p>'+bi(z,e)+'</p>'
def fig(rel,z,e):return '<figure><a href="'+rel+'" target="_blank"><img loading="lazy" src="'+rel+'" alt="'+html.escape(e)+'"></a><figcaption>'+bi(z,e)+'</figcaption></figure>'
def short(a):return a.rsplit('-',1)[-1] if a else 'INVALID'
def atomic_image(im,p):
 p=Path(p);tmp=p.with_name(p.name+'.new');im.save(tmp,format='PNG',dpi=(300,300));os.replace(tmp,p)
def overlay(rgb,m,b):
 v=rgb.astype(float);v[~m]=v[~m]*.30+205*.70;v[m]=v[m]*.72+np.array([0,195,208])*.28
 add=m&~b;remove=b&~m;v[add]=rgb[add]*.25+np.array([245,177,36])*.75;v[remove]=rgb[remove]*.25+np.array([221,60,153])*.75
 return Image.fromarray(np.rint(v).clip(0,255).astype('uint8'))
def selected(rgb,m):
 yy,xx=np.indices(m.shape);bg=np.where(((xx//8+yy//8)%2)[...,None],215,242)*np.ones((1,1,3));return Image.fromarray(np.where(m[...,None],rgb,bg).astype('uint8'))
def make_sheet(examples,title,p):
 W=2760;H=210+len(examples)*420+80;im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im)
 d.text((25,16),title,font=font(42),fill='#173441');d.text((25,72),'Same image / region across columns | left: overlay + changes; right: SELECTED ONLY',font=font(27),fill='#375563')
 d.text((25,115),'Cyan selected | grey hidden | gold ADDED | magenta REMOVED | checkerboard = display only, not model input',font=font(25),fill='#375563')
 d.text((25,169),'Full image @224',font=font(26),fill='#173441')
 for j,label in enumerate(LABELS):d.text((335+j*480,169),label,font=font(28),fill='#173441')
 for n,e in enumerate(examples):
  y=210+n*420;d.rectangle((12,y,W-12,y+407),fill='#f0f4f5' if n%2==0 else '#ffffff');rgb=np.array(Image.open(e['rgb_path']));im.paste(Image.fromarray(rgb),(25,y+42));d.text((25,y+9),e['case_id'],font=font(18),fill='#173441')
  d.text((25,y+275),e['class_name'].replace('_',' '),font=font(24),fill='#173441');d.text((25,y+310),e['selection_label'],font=font(23),fill='#173441');d.text((25,y+347),'I%03d / M%03d'%(e['image_index'],e['raw_mask_index']),font=font(23),fill='#375563')
  masks=np.load(e['masks_path'])['masks'];base=masks[0]
  for j,c in enumerate(e['conditions']):
   x=330+j*480;im.paste(overlay(rgb,masks[j],base),(x,y+42));im.paste(selected(rgb,masks[j]),(x+234,y+42))
   label=short(c['concept_id']) if c['valid'] else ('EMPTY' if c['empty'] else 'ZERO FEATURE');status='BASE' if j==0 else ('SAME' if c['concept_id']==e['baseline_concept'] and c['valid'] else 'SWITCH' if c['valid'] else 'INVALID')
   d.text((x+3,y+6),label+' | '+status,font=font(26),fill='#173441' if status in ['BASE','SAME'] else '#953451')
   d.text((x+3,y+276),f"Area {c['area']/50176:.2%} | IoU {c['iou']:.3f}",font=font(24),fill='#375563')
   d.text((x+3,y+312),f"Added {c['added']} / removed {c['removed']} px",font=font(22),fill='#375563')
   score='NA' if c['original_score'] is None else f"{c['original_score']:.3f}";margin='NA' if c['original_margin'] is None else f"{c['original_margin']:+.3f}"
   d.text((x+3,y+348),'s0 '+score+' | m0 '+margin,font=font(23),fill='#375563')
 d.text((25,H-57),'s0 = original concept score; m0 = original score minus strongest competitor. OC = complement. No new inference; colors/checkerboard are display only.',font=font(24),fill='#375563');atomic_image(im,p)

def build(report):
 from classes import ImageClass,SegmentClass,ConceptDatasetClass
 from hpc.final_mask_intervention import intervene
 report=Path(report);O.mkdir(exist_ok=True);dest=report/'hub/robustness-examples';dest.mkdir(exist_ok=True);assets=dest/'inputs';assets.mkdir(exist_ok=True)
 atlas=J(T/'atlas_final.json');accepted=J(A/'acceptance.json');assert accepted['status']=='ACCEPTED';new={e['class_name']:Path(R/e['collection']) for e in accepted['completed_classes']};old=J(R/'artifacts/bunya/batch-review-20260922/acceptance-A.json')['robustness']['statuses']
 old={s['class_name']:next((R/'artifacts/bunya/batch-review-20260922/collections'/str(s['job'])).glob('*/collection.json')).parent for s in old}
 records=[];examples=[];classstats=[];checks=[];figures=[];sourcefiles={};refs=[]
 rules={'R':'Uniform draw from all baseline-valid rows; RandomState(20260921 + atlas class index).','S':'Uniform draw from baseline-valid rows valid and unchanged under ALL four interventions; same RNG continued.','D':'Among valid label-switch pairs, highest mask IoU; ties condition order then raw row. Diagnostic, not representative.','E':'If any baseline-valid row becomes empty: first in condition/row order. Else first becomes zero. Else learned/complement switch with lowest baseline margin; otherwise first valid switch. Diagnostic.'}
 for ci,cl in enumerate(atlas['classes']):
  name=cl['name'];home=new.get(name,old[name]);ledger=J(home/'region_ledger.json');inputs=J(home/'input_manifest.json')['validation'];model=J(home/'model_identity.json');basis={k['basis_index']:k['id'] for k in cl['concepts']};concepts={k['id']:k for k in cl['concepts']};data={};masks={}
  for cond in CONDS:
   data[cond]=zload(home/(cond+'_science.npz'));zz=zload(home/(cond+'_masks.npz'));masks[cond]=np.unpackbits(zz['packed'],axis=2,count=int(zz['shape'][2])).astype(bool)
   for suffix in ['_science.npz','_masks.npz']:sourcefiles[str((home/(cond+suffix)).relative_to(R))]=sha(home/(cond+suffix))
  for fn in ['region_ledger.json','model_identity.json','input_manifest.json']:sourcefiles[str((home/fn).relative_to(R))]=sha(home/fn)
  base=data['identity0'];valid=base['valid'];bb=base['assignments'];oc=max(basis);rng=np.random.RandomState(SEED+ci);stable=valid.copy();pairs=[];invalid=[];ocpairs=[]
  for co,cond in enumerate(CONDS[1:],1):
   zz=data[cond];mm=masks[cond];same=valid & zz['valid'] & (bb==zz['assignments']);stable &= same
   inter=(mm&masks['identity0']).sum((1,2));union=(mm|masks['identity0']).sum((1,2));iou=np.divide(inter,union,out=np.ones(len(ledger),float),where=union>0)
   for i,r in enumerate(ledger):
    vv=bool(zz['valid'][i]);sw=bool(valid[i] and vv and bb[i]!=zz['assignments'][i]);cur=int(zz['assignments'][i]);orig=int(bb[i]);original_score=float(zz['scores'][i,orig]) if valid[i] and vv else None
    margin=float(original_score-np.max(np.delete(zz['scores'][i],orig))) if original_score is not None else None
    rec=dict(class_name=name,condition=cond,region_id=r['region_id'],raw_row=i,image_id=r['image_id'],baseline_valid=bool(valid[i]),valid=vv,empty=bool(zz['empty'][i]),zero=bool(zz['zero'][i]),baseline_concept=basis[orig] if valid[i] else None,current_concept=basis[cur] if vv else None,switch=sw,iou=float(iou[i]),baseline_area=int(base['area'][i]),area=int(zz['area'][i]),baseline_margin=float(base['margin'][i]) if valid[i] else None,original_score=original_score,original_margin=margin,transition='baseline_invalid' if not valid[i] else 'invalidated' if not vv else 'same' if not sw else 'to_OC' if cur==oc else 'from_OC' if orig==oc else 'learned_to_learned')
    records.append(rec)
    if sw:
     pairs.append((-float(iou[i]),co,i))
     if (orig==oc)!=(cur==oc):ocpairs.append((float(base['margin'][i]),co,i))
    if valid[i] and not vv:invalid.append((0 if zz['empty'][i] else 1,co,i))
  choices=[('R',int(rng.choice(np.flatnonzero(valid))),None,'Random'),('S',int(rng.choice(np.flatnonzero(stable))),None,'Stable control')]
  v=sorted(pairs)[0];choices.append(('D',v[2],CONDS[v[1]],'High-IoU switch'))
  if invalid:v=sorted(invalid)[0];choices.append(('E',v[2],CONDS[v[1]],'Empty / zero'))
  else:v=sorted(ocpairs)[0] if ocpairs else (0,sorted(pairs)[0][1],sorted(pairs)[0][2]);choices.append(('E',v[2],CONDS[v[1]],'OC transition'))
  cls=[]
  for kind,i,focus,label in choices:
   r=ledger[i];source=inputs[r['image_index']];evidence=T/'evidence'/str(cl['job_id']);img=evidence/'images/validation'/source['prepared_name'];assert sha(img)==r['input_sha256']==source['input_sha256'];raw=np.load(evidence/'cache/segments_validation'/(source['prepared_name']+'_sam.npy'),allow_pickle=False)[r['raw_mask_index']];assert ah(raw)==r['raw_mask_sha256']
   im=ImageClass(str(img),300);im.segments=[SegmentClass(raw,im)];ds=ConceptDatasetClass([im],model['default_cfg'],0,True,-1,.25);rgbtensor,masktensor,_=ds[0];m=masktensor.numpy()[0].astype(bool);assert ah(m)==r['effective_mask_sha256'] and np.array_equal(m,masks['identity0'][i])
   rgb=np.rint(ds.resize((ds.segments[0]*255).astype('uint8')).numpy().transpose(1,2,0)*255).astype('uint8')
   # Compare display RGB with the actual normalized input, before any overlays.
   recon=rgbtensor.numpy().transpose(1,2,0)*np.array(model['default_cfg']['std'])+np.array(model['default_cfg']['mean']);assert np.max(np.abs(recon-rgb/255))<1e-6
   case=f'ROB-{name}-{kind}';rgbpath=assets/(case+'-rgb.png');atomic_image(Image.fromarray(rgb),rgbpath);mp=assets/(case+'-masks.npz');np.savez_compressed(mp,masks=np.stack([masks[c][i] for c in CONDS]))
   conditions=[]
   for co,cond in enumerate(CONDS):
    zz=data[cond];m=masks[cond][i];b=masks['identity0'][i];assert np.array_equal(m,intervene(b,cond[:-1],int(cond[-1])));vv=bool(zz['valid'][i]);orig=int(bb[i]);ss=float(zz['scores'][i,orig]) if vv else None;mg=ss-float(np.max(np.delete(zz['scores'][i],orig))) if vv else None
    conditions.append(dict(condition=cond,valid=vv,empty=bool(zz['empty'][i]),zero=bool(zz['zero'][i]),assignment=int(zz['assignments'][i]),concept_id=basis[int(zz['assignments'][i])] if vv else None,area=int(m.sum()),iou=float((m&b).sum()/(m|b).sum()),added=int((m&~b).sum()),removed=int((b&~m).sum()),original_score=ss,original_margin=mg,mask_sha256=ah(m)))
   ex=dict(case_id=case,class_name=name,kind=kind,selection_label=label,rule=rules[kind],seed=SEED+ci,raw_row=i,region_id=r['region_id'],image_id=r['image_id'],image_index=r['image_index'],raw_mask_index=r['raw_mask_index'],baseline_concept=basis[int(bb[i])],baseline_margin=float(base['margin'][i]),focus_condition=focus,source_image=str(img.relative_to(R)),source_sha256=sha(img),rgb_path=str(rgbpath),masks_path=str(mp),rgb_sha256=sha(rgbpath),conditions=conditions,review_status='UNREVIEWED',scientific_job=home.parent.name)
   examples.append(ex);cls.append(ex);checks.append(case+' input hash, actual input RGB, raw/effective mask identity, all5 geometry')
   for id in dict.fromkeys(v['concept_id'] for v in conditions if v['concept_id']):
    k=concepts[id]
    refs.append(dict(case_id=case,concept_id=id,references=[dict(type=view,**ee) for view in ['validation_top','validation_random'] for ee in k['views'][view]['examples'][:1]]))
  path=dest/(name+'.png');make_sheet(cls,'HU-MCD | '+name.replace('_',' ')+' | fixed-mask sensitivity',path);figures.append(dict(path=str(path.relative_to(report)),width=2760,height=1970,sha256=sha(path),cases=[e['case_id'] for e in cls]))
  focus=cls[2];f=next(v for v in focus['conditions'] if v['condition']==focus['focus_condition']);classstats.append(dict(class_name=name,stable_all_four=int(stable.sum()),baseline_valid=int(valid.sum()),high_iou_example=focus['case_id'],focus_condition=focus['focus_condition'],focus_iou=f['iou'],focus_from=focus['baseline_concept'],focus_to=f['concept_id'],focus_baseline_margin=focus['baseline_margin'],focus_margin=f['original_margin']))
  print('Rendered',name,flush=True)
 # Front-page representative of the diagnostic selection RULE, not population.
 for pg in range(2):
  selected_ex=[e for e in examples if e['kind']=='D'][pg*5:(pg+1)*5];path=dest/f'overview-{pg+1}.png';make_sheet(selected_ex,f'HU-MCD | boundary sensitivity | diagnostic examples {pg*5+1}-{pg*5+5}',path);figures.append(dict(path=str(path.relative_to(report)),width=2760,height=2390,sha256=sha(path),cases=[e['case_id'] for e in selected_ex]))
 csvout(dest/'region-diagnostics.csv',records);csvout(dest/'selection.csv',[{k:v for k,v in e.items() if k!='conditions'} for e in examples]);dump(dest/'examples.json',examples);dump(dest/'concept-references.json',refs);dump(dest/'class-discussion.json',classstats);dump(dest/'figures.json',figures)
 if not (dest/'review-template.csv').exists():csvout(dest/'review-template.csv',[dict(case_id=e['case_id'],region_id=e['region_id'],review_status='UNREVIEWED',visible_change='',meaningful_content_changed='',assignment_plausibility='',reason='',reviewer='',reviewed_date='') for e in examples])
 stats=[]
 for cond in CONDS[1:]:
  rr=[r for r in records if r['condition']==cond and r['baseline_valid']];sw=[r for r in rr if r['switch']];same=[r for r in rr if r['transition']=='same'];stats.append(dict(condition=cond,baseline_valid=len(rr),counts={k:sum(r['transition']==k for r in rr) for k in ['same','learned_to_learned','to_OC','from_OC','invalidated']},switch_median_baseline_margin=float(np.median([r['baseline_margin'] for r in sw])),same_median_baseline_margin=float(np.median([r['baseline_margin'] for r in same])),switch_median_iou=float(np.median([r['iou'] for r in sw])),same_median_iou=float(np.median([r['iou'] for r in same]))))
 dump(dest/'population-summary.json',stats);dump(dest/'provenance.json',dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),source_acceptance_sha256=sha(A/'acceptance.json'),generator=str(Path(__file__).relative_to(R)),generator_sha256=sha(__file__),source_files=sourcefiles,selection_rules=rules,seed=SEED,class_order=[c['name'] for c in atlas['classes']],cases=len(examples),checks=checks,display='224x224 actual pre-normalization RGB; no content crop; original images linked separately; selected-only checkerboard and overlays are presentation only',inference_calls=0,remote_calls=0,notes='All4 perturbations shown for each case; duplicates across selection strata are permitted and identifiable; no human labels inferred.'))
 dump(O/'build-result.json',dict(status='RENDERED',examples=40,conditions=200,region_diagnostics=len(records),figures=figures,statistics=stats));return dest

def apply(report):
 report=Path(report);dest=report/'hub/robustness-examples'
 if not (dest/'provenance.json').exists():build(report)
 examples=J(dest/'examples.json');cs=J(dest/'class-discussion.json');stats=J(dest/'population-summary.json');refs=J(dest/'concept-references.json');by={e['case_id']:e for e in examples}
 body='<section id="robustness-visual-comparison"><h2>'+bi('边界变化：紧凑视觉对照','Boundary changes: compact visual comparison')+'</h2>'
 body+=para('每行同一图片、同一片段，五列依次为不扰动、腐蚀1/2、膨胀1/2像素。每格左边为叠加与变化、右边为仅选中区域；青色选中、灰色遮蔽、金色新增、品红移除。棋盘格与着色只用于展示，不是模型填充输入。','Each row follows one image/region: identity, erode1/2, dilate1/2. Each cell pairs overlay/change with selected-only. Cyan selected, grey hidden, gold added, magenta removed. Checkerboard and colors are display aids, not model filling inputs.')
 body+=para('所有图片均为实际224×224输入几何，没有为突出局部而裁剪。短标签Cxx只在本类别内有效；OC为补空间。s0是原概念投影分量范数除以特征范数，不是概率；m0是原概念得分减去最大竞争者，负值表示原概念失去领先。','Images use the actual224×224 input geometry, with no region crop. Cxx labels are class-specific; OC is complement. s0 is the original concept component norm divided by feature norm, not probability. m0 subtracts the strongest competitor from the original concept score; negative means the original concept lost its lead.')
 body+=para('总览每类选一个“掩码IoU最高但发生有效分配切换”的诊断例，明确不是随机总体代表。逐类图各四行：R固定随机、S所有扰动均未切换的随机控制、D最高IoU切换、E空/零区域或补空间切换。抽样种子20260921加固定类别序号，不改变实验样本。','Overview: one highest-mask-IoU valid switch per class, a diagnostic selection rather than a random population sample. Each class has four rows: R fixed random, S random stable-under-all-conditions control, D highest-IoU switch, E empty/zero or complement transition. Display RNG seed20260921 plus fixed class index; experimental samples unchanged.')
 for pg in [1,2]:
  body+=fig(f'hub/robustness-examples/overview-{pg}.png',f'诊断总览{pg}/2；点击打开高清图。','Diagnostic overview '+str(pg)+'/2; click for full-resolution export.')
  subset=cs[(pg-1)*5:pg*5];lo=min(x['focus_iou'] for x in subset);hi=max(x['focus_iou'] for x in subset)
  body+=para(f'图下分析：这五个特意筛选的例子在发生切换的条件下，掩码IoU为{lo:.3f}–{hi:.3f}。因此，高掩码重合并不保证最高分概念不变。它们用于定位边界敏感案例，不能用于估算发生率；请同时查看下方随机和稳定控制。',f'Analysis: in the switching condition, mask IoU ranges from {lo:.3f} to {hi:.3f} for these five deliberately selected cases. High mask overlap does not guarantee an unchanged top-scoring concept. These cases locate sensitivity, not its prevalence; compare the random and stable controls below.')
 body+=para('选择限制：十类最高IoU切换例恰好都来自膨胀1像素；按最高IoU筛选也可能偏向较大区域。因此不能用这十张诊断图比较腐蚀与膨胀的总体风险，或代替十类全量结果。','Selection limit: all ten maximum-IoU switch examples happen to come from dilation1; choosing maximum IoU may also favor larger regions. These diagnostic images cannot compare population-level erosion/dilation risks or replace the full ten-class results.')
 body+='<h3>'+bi('全量缓存支持的讨论','Discussion supported by all cached regions')+'</h3>'
 for s in stats:
  co=s['condition'];counts=s['counts'];switch=sum(counts[k] for k in ['learned_to_learned','to_OC','from_OC'])
  body+=para(f"{co}：5274个基准有效片段中，学习概念间切换{counts['learned_to_learned']}、转入补空间{counts['to_OC']}、离开补空间{counts['from_OC']}、失效{counts['invalidated']}。仍有效但切换者（n={switch}）原分配间隔中位数{s['switch_median_baseline_margin']:.3f}，保持者（n={counts['same']}）为{s['same_median_baseline_margin']:.3f}。",f"{co}: among5274 baseline-valid regions, {counts['learned_to_learned']} switch between learned concepts, {counts['to_OC']} enter complement, {counts['from_OC']} leave complement, and {counts['invalidated']} become invalid. Median baseline margin is {s['switch_median_baseline_margin']:.3f} for valid switches (n={switch}) versus {s['same_median_baseline_margin']:.3f} for retained assignments (n={counts['same']}).")
 body+=para('推断与限制：低原始得分间隔可能使小变化更容易改变赢家；以上是片段汇总的描述性关联，包含同图相关片段，不是独立样本检验，也不能单独证明机制。区域大小和实际变化比例仍可能影响结果。稳定不等于语义正确，切换也不等于解释错误；人工审阅尚未完成。','Inference and limits: a small initial score margin may make winner changes easier. These are pooled-region descriptive associations with correlated regions within images, not independent-sample tests or proof of mechanism. Region size and relative area change may also matter. Stability is not semantic correctness, and switching is not necessarily an explanation error; human review remains pending.')
 body+='<h3>'+bi('逐类四种选择对照','Four selection views for each class')+'</h3><nav>'+' · '.join('<a href="#rob-vis-'+r['class_name']+'">'+r['class_name'].replace('_',' ')+'</a>' for r in cs)+'</nav>'
 for r in cs:
  name=r['class_name'];body+='<section id="rob-vis-'+name+'"><h3>'+name.replace('_',' ')+'</h3>'+fig('hub/robustness-examples/'+name+'.png','R随机／S稳定控制／D高IoU切换／E失效或补空间；四种选择不得混作随机样本。','R random / S stable control / D high-IoU switch / E invalid or complement; do not treat the four selection strata as a random sample.')
  body+=para(f"本类所有四种扰动均保持的片段为{r['stable_all_four']}/{r['baseline_valid']}。诊断例{r['high_iou_example']}在{r['focus_condition']}下，从{short(r['focus_from'])}到{short(r['focus_to'])}，IoU={r['focus_iou']:.3f}；原概念相对最大竞争者的间隔由{r['focus_baseline_margin']:+.3f}变成{r['focus_margin']:+.3f}。这直接显示了几何变化和赢家变化的对应关系，不确定其语义是否改变。",f"{r['stable_all_four']}/{r['baseline_valid']} baseline-valid regions remain assigned under all four changes. Diagnostic case {r['high_iou_example']} moves {short(r['focus_from'])}→{short(r['focus_to'])} under {r['focus_condition']}, IoU={r['focus_iou']:.3f}; its original-concept margin changes {r['focus_baseline_margin']:+.3f}→{r['focus_margin']:+.3f}. This links geometric change to a winner change without establishing a semantic change.")
  if name=='airliner':
   body+=para('AI视觉观察（待人工确认）：D行选中的是飞机外侧的天空背景，飞机主体在selected-only中被遮蔽；因此不能把这行切换说成“飞机部件识别失败”。R行机头／机身前部区域在四种扰动下保持C07，是不同内容、不同选择规则的稳定对照，不是对同一片段的反证。','AI visual observation (awaiting human confirmation): row D selects sky around the aircraft; the aircraft itself is hidden in selected-only. Its switch cannot be described as failure to recognize an aircraft part. Row R retains C07 for a nose/front-fuselage region under all four changes; it is a stable comparator with different content and selection, not contrary evidence about the same region.')
  if name=='golden_retriever':
   body+=para('AI视觉观察（待人工确认）：D行五列仍能看到同一狗头／颈部主体，细节边界有变化，分配却由C10转为腐蚀后的C09或膨胀后的C02。原始间隔只有0.015，支持“竞争概念接近时赢家可能改变”的具体案例；还不能说这些概念语义相同。R/E行是狭小区域，非空但零特征与空掩码不同，不能从棋盘格或缩略图误判。','AI visual observation (awaiting human confirmation): row D retains the same visible head/neck subject across all five columns with boundary-detail changes, while C10 becomes C09 after erosion or C02 after dilation. The initial margin is only0.015, illustrating a winner change among close competitors; semantic equivalence is not established. R/E contain narrow regions; nonempty zero features differ from empty masks and must not be inferred from checkerboards or thumbnails.')
  body+='<details><summary>'+bi('追溯图片、完整概念ID与固定概念参照','Trace images, full concept IDs and fixed concept references')+'</summary>'
  for e in [e for e in examples if e['class_name']==name]:
   src=Path(e['source_image']);input_name=src.name;image_rel='assets/images/'+name+'/validation/'+input_name
   # Some report atlases omit unselected original images: copy only already local authorized input.
   outimg=report/image_rel
   if not outimg.exists():outimg.parent.mkdir(parents=True,exist_ok=True);outimg.write_bytes((R/src).read_bytes())
   assert sha(outimg)==e['source_sha256']
   body+='<p><b>'+e['case_id']+'</b> · '+e['region_id']+' · <a href="'+image_rel+'">'+e['image_id']+'</a><br>'+html.escape(' → '.join(c['concept_id'] or ('EMPTY' if c['empty'] else 'ZERO') for c in e['conditions']))+'</p>'
   for rr in [v for v in refs if v['case_id']==e['case_id']]:
    body+='<div><b>'+rr['concept_id']+'</b> '+bi('原已保存验证成员，仅作参照：','Previously saved validation members, reference only:')
    for ref in rr['references']:
     body+='<a href="'+ref['overlay']+'" target="_blank"><img loading="lazy" style="width:130px;height:130px;object-fit:contain" src="'+ref['overlay']+'" alt="'+ref['id']+'"></a>'+html.escape(('P: ' if ref['type']=='validation_top' else 'R: ')+ref['id'])+' '
    body+='</div>'
  body+='</details></section>'
 body+=para('未命名概念，不补写语义真值。诊断案例不代表总体；参考原型不代表整个概念。保存原始区域ID、掩码与图片哈希，以及空/零状态；本轮没有模型调用或远端操作。','No concept names or semantic ground truth are added. Diagnostic cases are not population representatives; prototypes do not represent whole concepts. Region IDs, image/mask hashes and empty/zero states are retained. No model calls or remote operations.')
 for n in ['selection.csv','examples.json','region-diagnostics.csv','population-summary.json','concept-references.json','review-template.csv','provenance.json','figures.json']+(['events.json'] if (dest/'events.json').exists() else []):body+='<a href="hub/robustness-examples/'+n+'">'+n+'</a> · '
 body+='</section>';path=report/'robustness.html';s=path.read_text();s=re.sub(r'<!-- ROB-VIS START -->.*?<!-- ROB-VIS END -->','',s,flags=re.S);anchor='<h2><span data-l="zh">读图与讨论</span>';block='<!-- ROB-VIS START -->'+body+'<!-- ROB-VIS END -->';s=s.replace(anchor,block+anchor,1) if anchor in s else s.replace('</main>',block+'</main>');path.write_text(s)
 # Update the scoped file manifest. Preserve all unrelated rows.
 mf=report/'FILES.sha256.csv'
 if mf.exists():
  rr={v['path']:v for v in csv.DictReader(mf.open())}
  original_images=[report/'assets/images'/e['class_name']/'validation'/Path(e['source_image']).name for e in examples]
  for f in [path]+list(dest.rglob('*'))+original_images:
   if f.is_file():key=f.relative_to(report).as_posix();rr[key]=dict(path=key,sha256=sha(f),bytes=f.stat().st_size)
  with mf.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['path','sha256','bytes']);w.writeheader();w.writerows(rr.values())
 dump(O/'publish.json',dict(at=datetime.now(timezone.utc).isoformat(),report=str(path),status='PUBLISHED_PENDING_QA',generator_sha256=sha(__file__)))

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--report',default=str(T/'report'));ap.add_argument('--build',action='store_true');args=ap.parse_args()
 if args.build:build(args.report)
 apply(args.report)
