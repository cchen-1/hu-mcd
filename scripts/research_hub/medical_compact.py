"""Compact medical evidence from accepted caches; no models, sampling or remote reads."""
from pathlib import Path
import json,csv,hashlib,re,html,shutil,math
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .medical_comparison import bi,p,h
R=Path(__file__).resolve().parents[2];B=R/'artifacts/bunya'
HP=B/'ham-derm-discovery-20260924/collection-28889071'
DP=B/'batch-review-20260922/collections/28792059/20260922T063256.936418Z'
EP=B/'medical-external-results-20260923/collection'
DB=B/'batch-review-20260922/bundle';EB=B/'medical-external-results-20260923/bundle'
def j(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open()))
def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
def txt(im,xy,s,n=25,color='#193842'):ImageDraw.Draw(im).text(xy,str(s),font=font(n),fill=color)
def fit(canvas,im,box):
 x,y,w,h=box;scale=min(w/im.width,h/im.height);im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS);canvas.paste(im,(int(x+(w-im.width)/2),int(y+(h-im.height)/2)))
def overlay(im,mask,hidden=False):
 a=np.array(im.convert('RGB'));m=np.asarray(Image.fromarray(mask.astype(bool)).resize(im.size,Image.Resampling.NEAREST),bool)
 v=a.astype(float);v[m]=.68*v[m]+.32*np.array([0,210,225])
 if hidden:v[~m]=.4*v[~m]+.6*190
 edge=m & ~(np.roll(m,1,0)&np.roll(m,-1,0)&np.roll(m,1,1)&np.roll(m,-1,1));v[edge]=[0,160,180]
 return Image.fromarray(np.rint(v).astype('uint8'))
def selected(im,mask):
 a=np.array(im);m=np.asarray(Image.fromarray(mask.astype(bool)).resize(im.size,Image.Resampling.NEAREST),bool);a[~m]=220;return Image.fromarray(a)
def fighref(rel,z,e):return '<figure><a href="'+rel+'" target="_blank"><img loading="lazy" src="'+rel+'" alt="'+html.escape(e)+'"></a><figcaption>'+bi(z,e)+'</figcaption></figure>'
def apply(S):
 S=Path(S);out=S/'hub/medical-compact';out.mkdir(parents=True,exist_ok=True)
 evidence=[];figures=[];memo={}
 extmeta={x['image_id']:x for x in j(EP/'actual_input_mapping.json')};hammeta={x['image_id']:x for x in j(HP/'fit_input_binding.json')['training']}
 external={x['segment_id']:x for x in j(EP/'segment_mapping.json')}
 sources={};assignments={};initial={}
 for route,base,roles in [('ham',HP,['training']),('derm',DP,['training','held_out'])]:
  for role in roles:
   sources[route,role]={x['segment_id']:x for x in j(base/(role+'_raw_segments.json'))};assignments[route,role]={x['segment_id']:x for x in j(base/(role+'_assignments.json'))}
  z=np.load(base/'initial_clusters.npz');print(route,'cluster arrays',z.files)
  initial[route]=dict(zip(z['segment_id'].astype(str),z['labels'].astype(int)))
 sources['derm','external']=external
 def region(route,role,sid):
  key=(route,role,sid)
  if key in memo:return memo[key]
  r=sources[route,role][sid];iid=r['image_id'];k=r['raw_index']
  if route=='ham':base=HP;cache=base/'cache/training';image=B/'ham-derm-discovery-20260924/collection-28883445/derm7pt_r101_presam'/f'{iid}.png'
  elif role=='external':base=EP;cache=base/'cache';image=EP/'inputs'/f'{iid}.png'
  else:base=DP;cache=base/'cache'/role;image=DB/'images'/f'{iid}.png'
  assert image.exists(),image
  im=Image.open(image).convert('RGB');raw=np.load(cache/(iid+'.png_sam.npy'))[k];eff=np.load(cache/(iid+'_effective_masks.npz'))['masks'][k]
  assert hashlib.sha256(raw.tobytes()).hexdigest()==r['raw_mask_sha256'],sid
  assert hashlib.sha256(eff.tobytes()).hexdigest()==r['effective_mask_sha256'],sid
  assert int(eff.sum())==r['effective_pixels'];assert raw.shape==np.asarray(im).shape[:2]
  if route=='ham':assert sha(image)==hammeta[iid]['input_sha256']
  if role=='external':assert sha(image)==extmeta[iid]['input_sha256']
  assigned=(r.get('concept_id') if role=='external' else assignments[route,role].get(sid,{}).get('concept_id')) or 'ZERO_EXCLUDED'
  name=route+'-'+role+'-'+iid+'-S'+str(k).zfill(4)
  for suffix,v in [('overlay',overlay(im,eff,True)),('only',selected(im,eff)),('input',im)]:v.save(out/(name+'-'+suffix+'.png'))
  rec=dict(route=route,role=role,segment_id=sid,image_id=iid,raw_index=k,final_concept=assigned,initial_cluster=initial[route].get(sid) if role=='training' else None,effective_fraction=float(eff.mean()),raw_mask_sha256=r['raw_mask_sha256'],effective_mask_sha256=r['effective_mask_sha256'],image_sha256=sha(image),image_source=str(image.relative_to(R)),mask_source=str(cache.relative_to(R)),asset=name+'-overlay.png')
  if route=='ham' or role=='external':
   meta=hammeta[iid] if route=='ham' else extmeta[iid]
   orig=EB/'originals'/(iid+Path(meta['inventory']['relative_path']).suffix)
   assert sha(orig)==meta['inventory']['sha256']
   original_name='original-'+iid+orig.suffix
   shutil.copy2(orig,out/original_name);rec.update(original_asset=original_name,original_sha256=sha(orig))
  memo[key]=(im,raw,eff,rec);return memo[key]
 def save(im,name,kind):
  im.save(out/name,dpi=(300,300));figures.append(dict(file=name,width=im.width,height=im.height,kind=kind,sha256=sha(out/name)));return name
 def pipeline(route,iid):
  meta=hammeta[iid] if route=='ham' else extmeta[iid];base=HP if route=='ham' else EP;cache=base/'cache/training' if route=='ham' else base/'cache';role='training' if route=='ham' else 'external'
  rr=[r for r in sources[route,role].values() if r['image_id']==iid];original=EB/'originals'/(iid+Path(meta['inventory']['relative_path']).suffix)
  assert sha(original)==meta['inventory']['sha256'];ori=Image.open(original).convert('RGB');inp=region(route,role,rr[0]['segment_id'])[0]
  # Verify that the stored pre-SAM image is exactly the documented transform of the already-local original.
  expected=ori.resize(inp.size,Image.Resampling.LANCZOS if route=='ham' else Image.Resampling.BICUBIC)
  assert np.array_equal(np.array(expected),np.array(inp)),(route,iid,'preprocess replay')
  rawsam=np.load(cache/(iid+'_sam_raw.npz'))['masks'];sm=j(cache/(iid+'_sam_metadata.json'));order=sorted(range(len(sm)),key=lambda i:sm[i]['predicted_iou'],reverse=True)[:3]
  W=3000;cell=(W-80)/max(len(rr),4);height=1550
  c=Image.new('RGB',(W,height),'white');title=('HAM | Derm7pt concept fitting' if route=='ham' else 'DermaMNIST-C | frozen concepts on external Derm7pt')
  txt(c,(35,20),title+' | '+iid,40);txt(c,(35,77),'Read top row left to right; then inspect ALL postprocessed regions and their final assignments below.',28)
  tops=[(ori,'1  Original',f'{ori.width} x {ori.height}; verified original hash'),(inp,'2  Actual pre-SAM input',f'{inp.width} x {inp.height}; '+('short-side 300; aspect preserved' if route=='ham' else 'E224 bicubic square (historic protocol)'))]
  tops += [(overlay(inp,rawsam[k]),f'3  SAM candidate {k:03d}',f'Rank {n+1} of {len(rawsam)} by saved score') for n,k in enumerate(order)]
  for i,(im,title,sub) in enumerate(tops):
   x=35+i*590;txt(c,(x,130),title,28);fit(c,im,(x,174,555,370));txt(c,(x,555),sub,22)
  txt(c,(35,610),f'4  HU-MCD postprocessing: {len(rawsam)} overlapping SAM candidates -> {len(rr)} regions. No one-to-one candidate mapping.',30)
  txt(c,(35,660),'Upper tile = postprocessed region; lower tile = effective selected area -> saved concept assignment.',27)
  for i,r in enumerate(rr):
   im,rm,em,rec=region(route,role,r['segment_id']);x=35+i*cell;dest=rec['final_concept'].split('-')[-1]
   if dest=='COMPLEMENT':dest='OC (not learned)'
   if dest=='ZERO_EXCLUDED':dest='ZERO (excluded)'
   txt(c,(x,717),f'S{r["raw_index"]:04d} -> {dest}',26);fit(c,overlay(im,rm),(x,762,cell-22,285));fit(c,selected(im,em),(x,1080,cell-22,285))
   txt(c,(x,1380),f'Effective area: {em.mean():.1%}',23)
   evidence.append(rec|dict(view='pipeline_all_regions',selection='case578 first cohort case; case752 previously recorded frame diagnostic',pipeline=iid,original_sha256=sha(original),original_source=str(original.relative_to(R)),candidate_display=order))
  txt(c,(35,1460),'Cyan = SELECTED. Grey = HIDDEN (display only; not a grey-background model input). Raw masks are not clinical ground truth.',24)
  txt(c,(35,1500),'Top 3 SAM candidates are a display subset; ALL downstream regions shown. Concept labels describe assignments, not medical names.',24)
  return save(c,route+'-'+iid+'-pipeline.png','pipeline')
 ham_examples=j(HP/'training_example_index.json');derm_examples={role:j(DP/(role+'_example_index.json')) for role in ['training','held_out']}
 hs=j(S/'medical-ham-derm/data/example-selection.json');ds=rows(DB/'display-selection.csv');es=rows(EB/'selection-manifest.csv')
 hacc=j(B/'ham-derm-discovery-20260924/discovery-acceptance.json');dresult=j(DP/'result.json')
 hconcepts=sorted(hacc['concepts'],key=lambda c:c['global_rank'])+[dict(concept_id='HUMCD-DERM7PT-R101-MEL-COMPLEMENT',cluster_id=None,initial_members=None,subspace_dimension=1986,final_members=138,global_importance=.383381373,global_rank=None)]
 dconcepts=[dict(concept_id='HUMCD-MEL-C001',cluster_id=0,initial_members=807,subspace_dimension=32,global_importance=dresult['concept_scores'][0],global_rank=1),dict(concept_id='HUMCD-MEL-COMPLEMENT',cluster_id=None,initial_members=None,subspace_dimension=2016,global_importance=dresult['concept_scores'][1],global_rank=None)]
 def picks(route,cid,role,view):
  if route=='ham':
   e=next(x for x in ham_examples if x['concept_id']==cid)
   if view in ['top','random']:return e['top_prototypes' if view=='top' else 'random_final_members']
   label='Initial cluster / 初始聚类' if view=='initial' else 'Low-margin diagnostics / 低间隔诊断'
   return [x['segment_id'] for x in hs if x['concept_id']==cid and x['view']==label]
  if role=='external':return [x['segment_id'] for x in es if x['concept_id']==cid and x['selection']=={'top':'released_top10_one_per_image','random':'random10_seed4303','diagnostic':'diagnostic_low_margin3'}.get(view)]
  if view in ['top','random']:return next(x for x in derm_examples[role] if x['concept_id']==cid)['top_prototypes' if view=='top' else 'random_final_members']
  return [x['segment_id'] for x in ds if x['concept_id']==cid and x['role']==role and x['selection_rule']==('initial_cluster_random8_seed4302' if view=='initial' else 'diagnostic_low_margin3')]
 def rolelabel(route,role):return {'training':('Derm7pt FIT (101; no held-out)' if route=='ham' else 'Source FIT (400)'), 'held_out':'Source HELD-OUT (70)', 'external':'Derm7pt EXTERNAL (101; frozen concepts)'}[role]
 def sheet(route,cid,roles):
  W=2600;c=Image.new('RGB',(W,160+len(roles)*600+65),'white');txt(c,(30,15),cid,38);txt(c,(30,75),'P1/P2 = first saved prototypes     R1/R2 = first saved random members     Cyan SELECTED; grey HIDDEN',27)
  for ri,role in enumerate(roles):
   y=140+ri*600;txt(c,(35,y),rolelabel(route,role),29)
   for col,(view,idx) in enumerate([('top',0),('top',1),('random',0),('random',1)]):
    ids=picks(route,cid,role,view);x=35+col*645
    if idx>=len(ids):txt(c,(x,y+210),'No saved member',26);continue
    im,rm,em,rec=region(route,role,ids[idx]);txt(c,(x,y+45),('P' if view=='top' else 'R')+str(idx+1),24);fit(c,overlay(im,em,True),(x,y+77,610,400));txt(c,(x,y+491),rec['image_id']+' / S'+str(rec['raw_index']).zfill(4),23);txt(c,(x,y+527),f'Selected {em.mean():.1%}',23)
    assert rec['final_concept']==cid
    evidence.append(rec|dict(view=view,selection='first 2 in frozen saved selection',display_concept=cid))
  txt(c,(35,c.height-45),'Examples retain original order; no aesthetic selection. Full canvas, no crops. Pixel area is not importance or quality.',25)
  return save(c,route+'-'+cid+'-members.png','concept_members')
 def figure(rel,z,e):return fighref(rel,z,e)
 pages={}
 for route,name,concepts,roles in [('derm','medical-dermamnist.html',dconcepts,['training','held_out','external']),('ham','medical-ham-derm/index.html',hconcepts,['training'])]:
  prefix='../hub/medical-compact/' if route=='ham' else 'hub/medical-compact/'
  body='<section id="compact-medical">'+h('图像证据：先看过程，再比较成员','Visual evidence: trace the process, then compare members')
  body+=p('沿用 reproduction 的青色选区／灰色隐藏背景；显示完整画布，不裁掉难解释区域。点击图片可查看高分辨率图。图像放大不会增加原始细节。','Uses the reproduction atlas convention: cyan selected, grey hidden; full canvases without cropping away difficult regions. Click for high-resolution exports. Enlargement adds no source detail.')
  body+='<nav class="tools"><a href="#medical-process">'+bi('① 处理链','1 Process')+'</a><a href="#medical-concepts">'+bi('② 全部学习概念','2 All learned concepts')+'</a><a href="#medical-oc">'+bi('③ 补空间','3 Complement')+'</a><a href="#medical-diagnostics">'+bi('④ 诊断与限制','4 Diagnostics and limits')+'</a></nav>'
  body+='<section id="medical-process">'+h('一张图如何变成概念成员？','How does an image become concept members?')+p('两条路线都用固定病例578展示处理链；它是清单首例，不是“最佳分割”。病例752保留为此前记录的图框诊断例。上排三个SAM候选按保存分数排序；下排展示该图全部后处理区域及最终归属。','Both routes use fixed case578, the first cohort case, not a “best segmentation”. Case752 remains the previously recorded frame diagnostic. The upper row shows three SAM candidates ranked by saved score; the lower row shows every postprocessed region and its final assignment.')
  for i,iid in enumerate(['Derm7pt-case578','Derm7pt-case752']):
   filename=pipeline(route,iid)
   if i:body+='<details class="compact-details"><summary>'+bi('展开图框诊断例752：同一规则、全部区域','Expand frame diagnostic case752: same rule, all regions')+'</summary>'
   body+=figure(prefix+filename,'原图 → 实际SAM输入 → 原始SAM候选；下排：后处理区域 → 有效选区及概念归属。','Original → actual SAM input → original SAM candidates; lower row: postprocessed region → effective selected area and concept assignment.')
   if i:body+='</details>'
  if route=='ham':
   body+=p('这张图值得注意：S0000主要选择外围，S0002选择中央较深色区域，两者都被分配给C002。SAM确实提出了不同区域，但最终概念把它们归到一起。这是此例的可见事实，不证明整个概念错误，也不证明SAM医学分割准确。','Notice S0000 selects mainly the periphery while S0002 selects the central darker area; both are assigned to C002. SAM proposed distinct regions, but the final concept groups them together. This is an observation in this case, not proof that the whole concept is wrong or that SAM is medically accurate.')
  else:
   body+=p('这张图中，中央较深色区域S0001归C001，外围S0000及细边缘S0002归补空间。它提供一个可追踪的区域分配例，不足以把C001命名为“病灶”或证明其他图片也一致。','Here the central darker S0001 region maps to C001; peripheral S0000 and thin-edge S0002 map to the complement. This is a traceable example of region assignment, insufficient to name C001 “lesion” or infer consistency on other images.')
  body+=p('概念由整个拟合集的区域特征共同形成，不是从这单张图独立“生成”的。SAM候选可重叠，后处理会重组区域；上、下排不能逐列当成同一个掩码。最后的灰底选区仅供看图，不是网络的灰底输入。','Concepts are fitted from region features across the fitting cohort, not independently created from this image. SAM candidates can overlap and postprocessing recomposes regions; upper and lower columns are not one-to-one masks. Grey-backed selected regions are display aids, not grey-background network inputs.')+'</section>'
  body+='<p class=""><span data-l="zh">有效掩码保存在分类器的224×224网格上；这里回映到SAM输入画布以方便比较。这个显示步骤不改变模型输入，也不代表HAM路线在SAM之前被改成224方图。</span><span data-l="en" hidden>Effective masks are saved on the classifier’s 224×224 grid and mapped back to the SAM-input canvas here for comparison. This display step changes no model input and does not mean the HAM route used a square224 image before SAM.</span></p>'
  body+='<section id="medical-concepts">'+h('全部学习概念：原型与随机成员并列','All learned concepts: prototypes alongside random members')
  body+=p(('旧路线只有1个学习概念；三行分别是源域拟合、源域留出和Derm7pt外部成员。' if route=='derm' else '新路线3个学习概念，按保存的全局重要性排序。所有示例均为Derm7pt拟合成员；留出／外部概念验证未运行。')+' 每个角色默认显示原型前2例＋已保存随机成员前2例；其余原有示例仍保留。','The earlier route has one learned concept, shown across source fitting, source held-out and external Derm7pt members. ' if route=='derm' else 'The new route has three learned concepts in saved global-importance order. All examples are Derm7pt fitting members; held-out/external concept validation is not run. ')+p('初始聚类大小不是最终分配数量；重要性不是质量或概率。','Initial cluster size is not final assignment count; importance is not quality or probability.')
  body+='<div class="tools"><label>'+bi('范围 ','Scope ')+'<select id="med-scope"><option value="all" data-zh="全部学习概念" data-en="All learned concepts">All learned concepts</option><option value="top" data-zh="前2个学习概念（按保存重要性）" data-en="Top 2 learned (saved importance)">Top 2 learned</option></select></label></div>'
  for c in concepts:
   cid=c['concept_id'];oc=cid.endswith('COMPLEMENT')
   if oc:body+='</section><section id="medical-oc">'+h('补空间：单列，不计入学习概念','Complement: separate, not a learned concept')
   counts={role:sum(x.get('concept_id')==cid for x in (external.values() if role=='external' else assignments[route,role].values())) for role in roles}
   body+='<article class="med-card" id="compact-'+cid+'" data-med-rank="'+str(c['global_rank'] or 0)+'" data-med-learned="'+str(not oc).lower()+'"><h3>'+cid+'</h3>'
   body+=p('初始cluster '+str(c['cluster_id'] if c['cluster_id'] is not None else 'N/A')+' · 初始成员 '+str(c['initial_members'] if c['initial_members'] is not None else 'N/A')+' · 维数 '+str(c['subspace_dimension'])+' · 全局重要性 '+f"{c['global_importance']:.4f}"+' · 学习概念排名 '+str(c['global_rank'] if c['global_rank'] is not None else 'N/A'), 'Initial cluster '+str(c['cluster_id'] if c['cluster_id'] is not None else 'N/A')+' · initial members '+str(c['initial_members'] if c['initial_members'] is not None else 'N/A')+' · dimension '+str(c['subspace_dimension'])+' · global importance '+f"{c['global_importance']:.4f}"+' · learned rank '+str(c['global_rank'] if c['global_rank'] is not None else 'N/A'))
   body+=p('最终分配：'+'；'.join(rolelabel(route,r)+' '+str(counts[r]) for r in roles),'Final assignments: '+'; '.join(rolelabel(route,r)+' '+str(counts[r]) for r in roles))
   body+=figure(prefix+sheet(route,cid,roles),'P=保存的原型；R=保存的固定随机成员。青色=有效所选区域，灰色=隐藏。','P = saved prototype; R = saved fixed random member. Cyan = effective selected area; grey = hidden.')
   # Native, legible per-member view supports originals and selected-only inspection.
   for view in ['top','random','initial','diagnostic']:
    if view=='initial' and oc:continue
    labels={'top':('更多原型与原图','More prototypes and input images'),'random':('更多随机成员与原图','More random members and input images'),'initial':('初始聚类成员（不是最终分配成员）','Initial cluster members (not final assignments)'),'diagnostic':('低分配间隔诊断（非代表性抽样）','Low-margin diagnostics (not representative)')}
    body+='<details class="compact-details"><summary>'+bi(*labels[view])+'</summary>'
    for role in (['training'] if view=='initial' else roles):
     ids=picks(route,cid,role,view)[:4 if view=='initial' else (3 if view=='diagnostic' else 10)]
     body+=h(rolelabel(route,role),rolelabel(route,role),4)+'<div class="med-member-grid">'
     for sid in ids:
      im,rm,em,rec=region(route,role,sid);assert view=='initial' or rec['final_concept']==cid
      fn=rec['asset'];stem=fn.removesuffix('-overlay.png');evidence.append(rec|dict(view=view,selection='first entries from frozen saved selections; max4 initial / max3 diagnostic / max10 other',display_concept=cid))
      cap=rec['image_id']+' / S'+str(rec['raw_index']).zfill(4)+' · '+f'{em.mean():.1%}'
      body+='<figure><a href="'+prefix+fn+'"><img loading="lazy" src="'+prefix+fn+'" alt="'+sid+' selected region"></a><figcaption>'+html.escape(cap)+'<br>'+bi('初始cluster ','Initial cluster ')+str(rec['initial_cluster'] if rec['initial_cluster'] is not None else 'N/A')+' → '+rec['final_concept'].split('-')[-1]+'<br>'+('<a href="'+prefix+rec['original_asset']+'">'+bi('原图','Original')+'</a> · ' if rec.get('original_asset') else '')+'<a href="'+prefix+stem+'-input.png">'+bi('实际输入','Actual input')+'</a> · <a href="'+prefix+stem+'-only.png">'+bi('仅选区','Selected only')+'</a></figcaption></figure>'
     body+='</div>'
    body+='</details>'
   body+='<details class="compact-details"><summary>'+bi('填写探索性审阅：默认未审阅','Exploratory review: unreviewed by default')+'</summary><form data-med-review="'+cid+'">'
   for key,z,e in [('description','稳定简短描述','Stable short description'),('content_type','主体／部件／背景／混合','Subject / part / background / mixed'),('random_support','随机成员是否支持描述','Do random members support it?'),('cohort_consistency','不同角色的成员是否一致；无留出时填不适用','Consistency across roles; N/A where no held-out set'),('unrelated','明显不相关成员','Clearly unrelated members'),('evidence','依据、示例ID及限制','Evidence, example IDs and limits')]:body+='<label>'+bi(z,e)+'<textarea name="'+key+'"></textarea></label>'
   body+='<label>'+bi('状态','Status')+'<select name="status"><option value="unreviewed">UNREVIEWED</option><option value="in_progress">IN PROGRESS</option><option value="reviewed">REVIEWED</option></select></label></form></details></article>'
  body+='</section><section id="medical-diagnostics">'+h('重点检查什么？','What deserves closer inspection?')
  body+=p('① 同一概念的P与R是否共享稳定内容？② 青色区域是否常是整图、外围皮肤或图框？这些内容本身不自动等于失败。③ 初始聚类与最终成员哪里变化？④ 零特征区域是否仍有可见图像内容？','1. Do prototypes and random members share stable content? 2. Are selected regions often whole images, surrounding skin or frames? None automatically implies failure. 3. Where do initial and final memberships differ? 4. Do zero-feature regions still contain visible image content?')
  if route=='ham':body+=p('C003对应cluster4的118个初始成员中，78个最终进入补空间；C001的既存随机例752含图框区域。它们是明确的检查入口，不足以单独解释失败原因。72/423个区域为零特征，掩码非空。','Of 118 initial cluster4 members for C003, 78 end in the complement; the existing C001 random example752 includes a frame region. These are specific review targets, not sufficient explanations of failure. 72/423 regions have zero features despite nonempty masks.')
  else:body+=p('源域训练68/917、留出14/157、外部37/296个区域为零特征。源域和外部示例必须分开看；同一个冻结概念ID不保证内容不变。','Zero features: source training 68/917, held-out 14/157, external 37/296 regions. Inspect source and external members separately; a shared frozen concept ID does not guarantee unchanged visual content.')
  body+=p('当前图像可帮助检查内容一致性，不能直接回答病灶边界或七点征象的分割精度。图像级标签不是像素真值；没有医生审阅时不赋予已确认的医学名称。','These images support inspection of content consistency, not lesion-boundary or seven-point-sign segmentation accuracy. Image-level labels are not pixel ground truth; medical names remain unconfirmed without clinician review.')
  body+=p('选择规则：沿用保存的原型排序；随机成员沿用已保存seed4301（源域/HAM）或4303（外部）；初始成员沿用已有seed4302清单，不新增抽样。重要性为MEL分类头权重的分解分量平方范数／原权重平方范数；补空间不参与排名。','Selection: saved prototype order; saved random seed4301 (source/HAM) or4303 (external); existing initial-member seed4302 selections, no new sampling. Importance is squared norm of the decomposed MEL head-weight component divided by original squared weight norm; complement excluded from ranking.')
  oldreview='../medical-ham-derm/data/review-template.csv' if route=='ham' else 'workstreams/medical-external-results-20260923/concepts.html'
  body+=p('<a href="'+prefix+'selection.csv">示例／掩码映射CSV</a> · <a href="'+prefix+'figures.json">全部高分辨率图清单</a> · <a href="'+oldreview+'">原有审阅入口</a>','<a href="'+prefix+'selection.csv">Example/mask mapping CSV</a> · <a href="'+prefix+'figures.json">High-resolution figure inventory</a> · <a href="'+oldreview+'">Existing review entry</a>')
  body+='<button id="med-export">'+bi('导出本页紧凑视图审阅','Export this compact-view review')+'</button><span id="med-review-status"></span>'+p('新表单单独保存，不覆盖原有审阅或CSV；均为探索性记录，不是医学人类实验。请导出备份。','New forms store separately without overwriting earlier reviews or CSVs. These are exploratory records, not a medical human study. Export a backup.')+'</section></section>'
  pages[name]=body
 for name,body in pages.items():
  f=S/name;s=f.read_text();s=re.sub(r'<!-- MEDICAL-COMPACT -->.*?<!-- /MEDICAL-COMPACT -->','',s,flags=re.S)
  at=s.index('<!-- /QUESTION-GUIDE -->')+len('<!-- /QUESTION-GUIDE -->') if '<!-- /QUESTION-GUIDE -->' in s else s.index('</h1>')+5
  s=s[:at]+'<!-- MEDICAL-COMPACT -->'+body+'<!-- /MEDICAL-COMPACT -->'+s[at:]
  # Keep original evidence intact and addressable, but prevent duplicate large galleries dominating the reading flow.
  if 'id="medical-legacy-evidence"' not in s:
   at=s.index('<!-- /MEDICAL-COMPACT -->')+len('<!-- /MEDICAL-COMPACT -->');end=s.index('</main>',at)
   s=s[:at]+'<details class="compact-details" id="medical-legacy-evidence"><summary>'+bi('原有完整报告、全部证据与历史记录','Original full report, all evidence and history')+'</summary>'+s[at:end]+'</details>'+s[end:]
  rel='../hub/' if name.startswith('medical-ham') else 'hub/'
  if 'medical-compact.css' not in s:s=s.replace('</head>','<link rel="stylesheet" href="'+rel+'medical-compact.css"></head>')
  if 'medical-compact.js' not in s:s=s.replace('</body>','<script src="'+rel+'medical-compact.js"></script></body>')
  f.write_text(s)
 parent=S/'medical.html'
 if parent.exists():
  text=parent.read_text();text=re.sub(r'<!-- MEDICAL-COMPACT-LINKS -->.*?<!-- /MEDICAL-COMPACT-LINKS -->','',text,flags=re.S)
  links=p('<a href="medical-dermamnist.html#compact-medical">DermaMNIST-C：处理链＋源域/留出/外部概念对照</a> · <a href="medical-ham-derm/index.html#compact-medical">HAM：处理链＋全部3个概念</a>','<a href="medical-dermamnist.html#compact-medical">DermaMNIST-C: process + source/held-out/external members</a> · <a href="medical-ham-derm/index.html#compact-medical">HAM: process + all three concepts</a>')
  text=text.replace('</h1>','</h1><!-- MEDICAL-COMPACT-LINKS -->'+links+'<!-- /MEDICAL-COMPACT-LINKS -->',1);parent.write_text(text)
 css='''.med-card{border-top:2px solid #9bbdc4;padding:1rem 0;margin:1rem 0}.med-member-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.med-member-grid figure{margin:0}.med-member-grid img{width:100%}.med-member-grid figcaption{font-size:12px}.med-card label{display:block;margin:.6rem 0}.med-card textarea{display:block;width:100%;box-sizing:border-box;min-height:65px;font:inherit}.compact-details{border:1px solid #b9ccd2;padding:12px;margin:12px 0;border-radius:6px}.compact-details>summary{cursor:pointer;font-weight:600}#compact-medical figure{margin:12px 0}#compact-medical img{max-width:100%;height:auto}#compact-medical{overflow-wrap:anywhere}@media(max-width:700px){.med-member-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}'''
 (S/'hub/medical-compact.css').write_text(css)
 js="""(()=>{const route=location.pathname.includes('medical-ham-derm')?'ham':'dermamnist';const key='humcd-medical-compact-review-v1:'+route;let records={};try{records=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}document.querySelectorAll('form[data-med-review]').forEach(f=>{const id=f.dataset.medReview;f.querySelectorAll('[name]').forEach(el=>{if(records[id]&&records[id][el.name]!==undefined)el.value=records[id][el.name];el.addEventListener('input',()=>{records[id]=Object.fromEntries([...f.querySelectorAll('[name]')].map(x=>[x.name,x.value]));records[id].updated_at=new Date().toISOString();try{localStorage.setItem(key,JSON.stringify(records));document.getElementById('med-review-status').textContent=document.documentElement.lang==='zh'?' 已保存；请导出备份。':' Saved; export a backup.'}catch(e){document.getElementById('med-review-status').textContent='Storage unavailable; export your review.'}})})});const b=document.getElementById('med-export');if(b)b.onclick=()=>{const ids=[...document.querySelectorAll('[data-med-review]')].map(f=>f.dataset.medReview);const data={schema_version:1,scope:'Exploratory compact-view review; not clinical validation',route,records:ids.map(id=>({concept_id:id,status:'unreviewed',...(records[id]||{})}))};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='medical-'+route+'-compact-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};const filter=document.getElementById('med-scope');if(filter)filter.onchange=()=>document.querySelectorAll('[data-med-learned=true]').forEach(c=>c.hidden=filter.value==='top'&&Number(c.dataset.medRank)>2);function openHash(){const e=document.getElementById(decodeURIComponent(location.hash.slice(1)));if(!e)return;for(let p=e;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;e.scrollIntoView()}addEventListener('hashchange',openHash);openHash();})();"""
 (S/'hub/medical-compact.js').write_text(js)
 fields=list(dict.fromkeys(k for r in evidence for k in r))
 with (out/'selection.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(evidence)
 (out/'figures.json').write_text(json.dumps(figures,indent=2)+'\n')
 receipt=dict(generated_at=datetime.now(timezone.utc).isoformat(),figures=len(figures),selection_records=len(evidence),unique_regions=len(memo),new_predictions=0,new_sampling=0,source_originals='Already-local accepted medical external report originals; checked against both input bindings',rules='saved order; P2/R2; all concepts; OC separate; all downstream regions for two previously fixed cases',generator_sha256=sha(Path(__file__)))
 (out/'generation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return receipt
if __name__=='__main__':
 import sys
 apply(Path(sys.argv[1]))
