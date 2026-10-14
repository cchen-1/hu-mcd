"""Explain cached segmentation stages without running a model or changing assignments."""
from pathlib import Path
import json,hashlib,re,html
import numpy as np
from PIL import Image,ImageDraw
from .medical_comparison import bi,p,h
R=Path(__file__).resolve().parents[2];B=R/'artifacts/bunya/ham-derm-discovery-20260924';P=B/'collection-28889071'
def load(n):return json.loads((P/n).read_text())
def apply(S):
 out=Path(S)/'medical-ham-derm';dest=out/'walkthrough';dest.mkdir(exist_ok=True)
 raw=load('training_raw_segments.json');assigned={x['segment_id']:x for x in load('training_assignments.json')};source=load('fit_input_binding.json')['training']
 # Fixed first cohort record plus the already-discussed frame example; no aesthetic selection.
 selected=[source[0]['image_id'],'Derm7pt-case752'];records=[]
 colors={'C001':(0,130,180),'C002':(180,70,150),'C003':(20,150,100),'COMPLEMENT':(180,120,20),'ZERO':(110,110,110)}
 section='<!-- MEDICAL-WALKTHROUGH -->'+h('先看一张图如何变成概念成员','Start with one image: from masks to concept membership')
 section+=p('先沿同一病例向下看，再进入跨图片概念。这里使用实际短边300工作输入，不是原始分辨率图；没有新增推理。','Follow one case down the page before comparing concepts across images. These are the actual short-side300 working inputs, not full-resolution originals. No new inference.','lead')
 section+='<div class="toc"><a href="#trace-Derm7pt-case578">'+bi('首例逐步追踪','First-case walkthrough')+'</a><a href="#trace-Derm7pt-case752">'+bi('图框诊断例','Frame diagnostic case')+'</a><a href="#HUMCD-DERM7PT-R101-MEL-C001">'+bi('直接查看概念示例','Jump to concept examples')+'</a></div>'
 section+=p('SAM候选 → HU-MCD区域 → 有效掩码 → 区域特征 → 概念归属。候选会重叠，后处理会组合区域并可能补充未覆盖区域，因此不是一个SAM候选对应一个最终概念。','SAM candidates → HU-MCD regions → effective masks → region features → concept assignments. Candidates overlap; postprocessing combines regions and may add uncovered areas. One SAM candidate does not correspond to one final concept.','note')
 section+=p('案例选择：固定清单首例；另加此前讨论图框问题的case752作为诊断例。诊断例不是总体代表。轮廓/颜色仅表示选区，不表示医学标签或重要性；灰底只为展示，并非网络实际接收的输入。','Selection: the first fixed cohort case, plus case752, the previously discussed frame example. The diagnostic case is not population-representative. Outlines and colours show selected regions, not medical labels or importance. Grey backgrounds are display aids, not literal network inputs.')
 def fig(name,z,e):return '<figure><a href="walkthrough/'+name+'"><img loading="lazy" src="walkthrough/'+name+'" alt="'+html.escape(e)+'"></a><figcaption>'+bi(z,e)+'</figcaption></figure>'
 for iid in selected:
  row=next(x for x in source if x['image_id']==iid);path=B/'collection-28883445/derm7pt_r101_presam'/Path(row['input_path']).name
  assert hashlib.sha256(path.read_bytes()).hexdigest()==row['input_sha256']
  im=Image.open(path).convert('RGB');a=np.array(im);w,hh=im.size
  sam=np.load(P/f'cache/training/{iid}_sam_raw.npz')['masks'];meta=load(f'cache/training/{iid}_sam_metadata.json')
  masks=np.load(P/f'cache/training/{iid}.png_sam.npy');eff=np.load(P/f'cache/training/{iid}_effective_masks.npz')['masks'];rr=[x for x in raw if x['image_id']==iid]
  def overlay(mask,color):
   mask=np.asarray(mask,dtype=bool);v=a.copy();v[mask]=(.68*a[mask]+.32*np.array(color)).astype(np.uint8)
   edge=mask & ~(np.roll(mask,1,0)&np.roll(mask,-1,0)&np.roll(mask,1,1)&np.roll(mask,-1,1));v[edge]=color
   return Image.fromarray(v)
  im.save(dest/(iid+'-input.png'))
  section+=f'<section id="trace-{iid}">'+h(iid+('：首例' if iid==selected[0] else '：图框诊断例'),iid+(': first cohort case' if iid==selected[0] else ': frame diagnostic case'),3)
  section+=h('1 · 实际 SAM 输入','1 · Actual SAM input',3)+fig(iid+'-input.png',f'{w}×{hh}；保留比例；所有候选都来自这张图。',f'{w}×{hh}; aspect preserved. Every candidate below comes from this image.')
  section+=h(f'2 · SAM 原始候选：{len(sam)}个',f'2 · Original SAM candidates: {len(sam)}',3)+p('每张图只高亮一个候选；先显示模型分数最高的4个，其余展开可见。编号保留SAM原始顺序。分数仅用于排序，不等于真实分割准确率。','Each image highlights one candidate. The four highest model scores appear first; expand to see all others. IDs retain original SAM order. Scores are for ranking, not measured segmentation accuracy.')
  order=sorted(range(len(sam)),key=lambda i:meta[i]['predicted_iou'],reverse=True)
  for group,ids in enumerate([order[:4],order[4:]]):
   if group:section+='<details><summary>'+bi('查看其余SAM候选','Show remaining SAM candidates')+'</summary>'
   section+='<div class="trace-grid">'
   for k in ids:
    name=f'{iid}-sam-{k:03d}.png';overlay(sam[k],(0,140,220)).save(dest/name)
    section+=fig(name,f'SAM-{k:03d} · 选区 {sam[k].mean():.1%} · 模型分数 {meta[k]["predicted_iou"]:.3f}',f'SAM-{k:03d} · selected {sam[k].mean():.1%} · model score {meta[k]["predicted_iou"]:.3f}')
   section+='</div>'
   if group:section+='</details>'
  section+=h(f'3–5 · 后处理留下{len(masks)}个区域：逐个追踪',f'3–5 · Follow each of the {len(masks)} postprocessed regions',3)
  section+=p('每行是同一个区域：左为HU-MCD后处理区域（不是原始SAM候选），中为网络使用的有效掩码，右为仅显示有效选区。中图掩码从224网格回映到本图，仅用于显示；归属来自保存的特征，不是看图命名。','Each row follows the same region: left, HU-MCD postprocessing output (not a raw SAM candidate); centre, the effective network mask; right, the effective selected area alone. The centre mask is mapped back from its224 grid for display only. Assignment comes from saved features, not visual naming.')
  for k,r in enumerate(rr):
   assert r['raw_index']==k and hashlib.sha256(masks[k].tobytes()).hexdigest()==r['raw_mask_sha256']
   assert hashlib.sha256(eff[k].tobytes()).hexdigest()==r['effective_mask_sha256']
   x=assigned.get(r['segment_id']);label=x['concept_id'].split('-')[-1] if x else 'ZERO';color=colors[label]
   em=np.asarray(Image.fromarray(eff[k]).resize((w,hh),Image.Resampling.NEAREST),bool)
   v=np.full_like(a,215);v[em]=a[em]
   name=f'{iid}-region-{k:02d}';overlay(masks[k],color).save(dest/(name+'-post.png'));overlay(em,color).save(dest/(name+'-effective.png'));Image.fromarray(v).save(dest/(name+'-only.png'))
   destlabel=bi('零特征：按原规则不参与拟合','Zero feature: excluded from fitting by the released rule') if not x else ('<a href="#'+('complement' if label=='COMPLEMENT' else x['concept_id'])+'">'+bi('补空间（非学习概念）','Complement (not learned)')+'</a>' if label=='COMPLEMENT' else '<a href="#'+x['concept_id']+'">'+label+'</a>')
   section+='<div class="panel trace-region"><h4>S'+str(k).zfill(4)+' → '+destlabel+'</h4><div class="trace-row">'
   section+=fig(name+'-post.png','3 · HU-MCD后处理区域','3 · HU-MCD postprocessed region')+fig(name+'-effective.png','4 · 有效选区（有颜色的部分）','4 · Effective selected area (coloured)')+fig(name+'-only.png','5 · 选区示意；灰色=隐藏','5 · Display only; grey=hidden')+'</div>'
   section+=p(f'后处理选区占图 {masks[k].mean():.2%}；有效224掩码占图 {eff[k].mean():.2%}。不是重要性分数。',f'Postprocessed area {masks[k].mean():.2%}; effective224 mask area {eff[k].mean():.2%}. These are areas, not importance scores.')+'</div>'
   records.append(dict(image_id=iid,segment_id=r['segment_id'],assignment=x['concept_id'] if x else 'ZERO_EXCLUDED',raw_mask_sha256=r['raw_mask_sha256'],effective_mask_sha256=r['effective_mask_sha256']))
  section+='</section>'
 section+=p('下方旧面板的 Raw selected region 指HU-MCD后处理区域，不是SAM最初候选；Effective region only 是展示示意，不是灰底推理输入。','In the existing panels below, Raw selected region means the HU-MCD postprocessed region, not an original SAM candidate. Effective region only is a display aid, not a grey-background inference input.','note')
 section+=h('再看跨图片概念：不要只看原型','Next compare concepts across images—not only prototypes')+p('下方完整概念索引与全部旧示例保持不变。建议在同一概念内并列检查原型、随机成员与初始成员：是否共享一种稳定内容，是否混入边框，以及分配变化发生在哪一阶段。','The complete index and all existing examples below are unchanged. Within each concept compare prototypes, random members and initial members: do they share stable content, include frame artifacts, or change membership at a later stage?')+'<!-- /MEDICAL-WALKTHROUGH -->'
 file=out/'index.html';s=file.read_text();s=re.sub(r'<!-- MEDICAL-WALKTHROUGH -->.*?<!-- /MEDICAL-WALKTHROUGH -->','',s,flags=re.S)
 # Place this before numerical summaries: the viewer sees the process first.
 anchor=re.search(r'<p class="note">',s);assert anchor;s=s[:anchor.start()]+section+s[anchor.start():]
 style='<style>.trace-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.trace-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.trace-row figure,.trace-grid figure{margin:0}.trace-region{margin:20px 0}#trace-'+selected[0]+' > figure{max-width:650px}#trace-'+selected[1]+' > figure{max-width:650px}@media(max-width:800px){.trace-row,.trace-grid{grid-template-columns:1fr}}</style>'
 s=s.replace('</head>',style+'</head>');file.write_text(s)
 (dest/'mapping.json').write_text(json.dumps(dict(selection_rule='First fixed cohort case plus previously discussed case752; not representative sampling',images=selected,records=records,new_predictions=0),indent=2)+'\n')
 return dict(images=selected,regions=len(records),assets=len(list(dest.glob('*.png'))))
if __name__=='__main__':
 import sys
 print(json.dumps(apply(Path(sys.argv[1]))))
