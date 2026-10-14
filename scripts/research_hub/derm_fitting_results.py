"""Incremental private report extension; fixed cached examples, no inference."""
from pathlib import Path
import json,csv,html,shutil,hashlib,re
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[2];B=R/'artifacts/bunya/ham-derm-discovery-20260924';P=B/'collection-28889071'
def j(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def apply(S):
 a=j(B/'discovery-acceptance.json');assert a['status']=='ACCEPTED_ENGINEERING_DISCOVERY'
 out=S/'medical-ham-derm';out.mkdir(exist_ok=True);assets=out/'assets';assets.mkdir(exist_ok=True);data=out/'data';data.mkdir(exist_ok=True)
 for src in [B/'discovery-acceptance.json',B/'classifier-acceptance.json',B/'discovery-concepts.csv',B/'classifier-class-metrics.csv',B/'cluster-to-final.csv',P/'concept_index.json',P/'training_assignments.json',P/'training_example_index.json',P/'cluster_filters.json',P/'anomalies.jsonl',P/'warnings.jsonl',P/'resolved_config.json']:
  shutil.copy2(src,data/src.name)
 shutil.copy2(B/'classifier-curves.png',assets/'classifier-curves.png')
 raw=j(P/'training_raw_segments.json');lookup={s['segment_id']:s for s in raw};examples=j(P/'training_example_index.json');concepts=j(P/'concept_index.json');ci={c['concept_id']:c for c in concepts}
 f=np.load(P/'scientific/training.npz');assign=j(P/'training_assignments.json');margin=np.sort(f['concept_activations'],axis=1)[:,-1]-np.sort(f['concept_activations'],axis=1)[:,-2];kept=[s for s in raw if not s['zero_feature']]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
 selections=[];generated={}
 def panel(sid):
  if sid in generated:return generated[sid]
  s=lookup[sid];iid=s['image_id'];path=B/'collection-28883445/derm7pt_r101_presam'/f'{iid}.png';im=Image.open(path).convert('RGB');arr=np.asarray(im);w,h=im.size
  rm=np.load(P/f'cache/training/{iid}.png_sam.npy')[s['raw_index']].astype(bool)
  em=np.load(P/f'cache/training/{iid}_effective_masks.npz')['masks'][s['raw_index']]
  em=np.asarray(Image.fromarray(em).resize((w,h),Image.Resampling.NEAREST),dtype=bool)
  def overlay(mask,color):
   v=arr.copy();v[mask]=(.58*arr[mask]+.42*np.array(color)).astype(np.uint8);return Image.fromarray(v)
  selected=np.full_like(arr,220);selected[em]=arr[em]
  views=[im,overlay(rm,(0,180,240)),overlay(em,(255,155,20)),Image.fromarray(selected)]
  canvas=Image.new('RGB',(w*2+36,h*2+136),'white');d=ImageDraw.Draw(canvas)
  labels=['Pre-SAM input (not full-resolution original)','Raw selected region: BLUE','Effective selected mask: ORANGE','Effective region only; grey = hidden']
  for k,(v,label) in enumerate(zip(views,labels)):
   x=12+(k%2)*(w+12);y=12+(k//2)*(h+46);d.text((x,y),label,fill='black',font=font);canvas.paste(v,(x,y+24))
  d.text((12,h*2+103),iid+' / S'+str(s['raw_index']).zfill(4),fill='black',font=font)
  # Larger export uses nearest-native display scaling only; it adds no image detail.
  name=iid+'-S'+str(s['raw_index']).zfill(4)+'.png';canvas.save(assets/name);generated[sid]=name;return name
 def card(sid,role,cid):
  s=lookup[sid];name=panel(sid);selections.append(dict(concept_id=cid,view=role,segment_id=sid,image_id=s['image_id'],raw_index=s['raw_index'],panel=name,raw_mask_sha256=s['raw_mask_sha256'],effective_mask_sha256=s['effective_mask_sha256']))
  return f'<figure><a href="assets/{name}"><img loading="lazy" src="assets/{name}" alt="{html.escape(sid)} selected-region panels"></a><figcaption>{html.escape(sid)}<br>原选区 / raw pixels: {s["raw_pixels"]}; 有效选区 / effective224 pixels: {s["effective_pixels"]}; zero={s["zero_feature"]}</figcaption></figure>'
 body='''<h1>HAM → Derm7pt：首次概念拟合 / First concept fitting</h1><p><a href="../index.html">← 统一报告入口 / Research report</a> · <a href="../medical.html">历史医学结果 / Prior medical results</a></p>
 <p class="notice">101例均用于拟合，0例留出；工程验收通过，不证明临床有效性或概念语义正确。<br>All101cases are fitting inputs, with no held-out set. Engineering acceptance does not establish clinical utility or semantic validity.</p>
 <p>5个初始聚类→3个学习概念，另有补空间。423片段中72个零特征排除出拟合；掩码并不为空，101图仍全部纳入。<br>Five initial clusters yield three learned concepts plus a complement.72/423regions have exact-zero features and are excluded by the released rule; their masks are nonempty. All101images remain represented.</p>
 <p>completeness=0.7853是权重子空间覆盖量，不是概念质量分数。最终分配：213/351归学习概念，138/351归补空间。<br>Completeness0.7853 measures classifier-weight subspace coverage, not interpretability.213/351retained regions map to learned concepts;138/351to the complement.</p>
 <p>所有图来自此前已收集的短边300工作输入。蓝色=原选区，橙色=模型有效选区，灰色=隐藏区域；有效224掩码仅为展示回映到输入尺寸。未取得原始分辨率图片的新传输许可。<br>Images use existing short-side300working inputs. Blue: raw selected region; orange: effective selected region; grey: hidden pixels. Effective224masks are resized only for display; no extra square input was introduced beforeSAM. Full-resolution originals were not newly exported.</p>'''
 body+='<h2>完整概念索引 / Complete concept index</h2><table><tr><th>ID</th><th>Cluster</th><th>初始 / Initial</th><th>维数 / Dim</th><th>最终 / Final</th><th>重要性 / Importance</th><th>Rank</th></tr>'
 for c in a['concepts']:body+=f'<tr><td><a href="#{c["concept_id"]}">{c["concept_id"]}</a></td><td>{c["cluster_id"]}</td><td>{c["initial_members"]}</td><td>{c["subspace_dimension"]}</td><td>{c["final_members"]}</td><td>{c["global_importance"]:.6f}</td><td>{c["global_rank"]}</td></tr>'
 body+='<tr><td><a href="#complement">COMPLEMENT（非学习概念）</a></td><td>不适用 / NA</td><td>NA</td><td>1986</td><td>138</td><td>0.383381</td><td>NA</td></tr></table><p>重要性=分解后的MEL分类头权重向量分量平方范数 / 原权重平方范数；不是概率，不代表语义，补空间不纳入学习概念排名。<br>Importance is the squared norm of a decomposed MEL head-weight component divided by the squared weight norm; not a probability or semantic rating. Complement excluded from learned ranking.</p>'
 body+='<h2>聚类→最终分配 / Cluster → final assignment</h2><table><tr><th>Initial cluster</th><th>C001</th><th>C002</th><th>C003</th><th>Complement</th></tr>'
 for i,row in enumerate(a['cluster_to_final']):body+='<tr><td>'+str(i)+(' (filtered)' if i in [2,3] else '')+'</td>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>'
 body+='</table><p>C003初始118成员中78转入补空间；最终45成员包括来自cluster1的5个。被过滤cluster2的27个全部进入补空间；cluster3有16个归C001。该对应关系不是分类准确率，也没有语义标准答案。<br>78/118originalC003members move to complement; its45final members include5fromcluster1. All27filteredcluster2members map to complement;16cluster3members map toC001. This is a membership transition, not classification accuracy or proof of assignment error.</p>'
 body+='<h2>概念展示 / Concept evidence</h2><p>每类视图前4例直接展开，其余折叠；无好坏筛选。top和随机成员用作业保存顺序（随机seed4301）；初始聚类成员用独立seed4302随机取最多10例；诊断取该概念分配间隔最低的3例（相同分数按原行序），不能代表总体。均为拟合例，不存在验证列。<br>First4examples per view are visible; remaining samples expandable. Saved prototypes/random members retain job order(seed4301); initial-cluster samples use independentseed4302,max10; diagnostics select3lowest assignment margins, ties in row order. No aesthetic selection. All are fitting examples; validation is not run.</p>'
 rng=np.random.RandomState(4302)
 for e in examples:
  cid=e['concept_id'];c=ci.get(cid);iscomp=c is None
  body+=f'<section id="{"complement" if iscomp else cid}"><h3>{cid}</h3><p>审阅状态：未审阅 / Review: UNREVIEWED</p>'
  groups=[]
  if c:groups.append(('Initial cluster / 初始聚类',rng.choice(c['initial_sorted_segments'],size=min(10,c['initial_members']),replace=False).tolist()))
  groups+=[('Top prototypes / 原规则原型',e['top_prototypes']),('Random final members / 随机最终成员',e['random_final_members'])]
  members=[i for i,s in enumerate(assign) if s['concept_id']==cid];low=sorted(members,key=lambda i:(margin[i],i))[:3]
  groups.append(('Low-margin diagnostics / 低间隔诊断',[kept[i]['segment_id'] for i in low]))
  for title,ids in groups:
   body+='<h4>'+title+'</h4><div class="grid">'+''.join(card(sid,title,cid) for sid in ids[:4])+'</div>'
   if len(ids)>4:body+='<details><summary>其余 / Remaining '+str(len(ids)-4)+'</summary><div class="grid">'+''.join(card(sid,title,cid) for sid in ids[4:])+'</div></details>'
  body+='</section>'
 body+='<h2>零特征诊断 / Zero-feature examples</h2><p>固定seed4303随机6例。有效掩码面积中位数：零特征1.59%，非零7.53%。这支持面积差异观察，不证明零特征由面积单独导致。<br>Six examples sampled with seed4303. Median effective-mask area:1.59%zero vs7.53%nonzero. This is an association, not proof of a causal explanation.</p><div class="grid">'
 for sid in np.random.RandomState(4303).choice([s['segment_id'] for s in raw if s['zero_feature']],6,replace=False):body+=card(sid,'zero_feature_random','ZERO_FEATURE')
 body+='</div><h2>有限AI视觉观察（待确认） / Limited AI visual inspection (provisional)</h2><p>仅查看每个学习概念及补空间的首个top、首个随机例，共8例，不是完整语义审阅。C001：case604-S0000为外围区域，case752-S0004几乎是图框边缘；C002：case793-S0002、case631-S0000选区较大，包含主要色素区域；C003：case726-S0001、case796-S0004多为外侧浅色皮肤；补空间：case752-S0003小角落与case812-S0004外围区域并存。C001至少这两例不支持直接赋予单一医学征象名称；C002两例视觉范围较一致，但并不证明全部成员或医学语义一致。<br>Only the first saved prototype and random member per learned concept/complement were viewed,8examples total. C001includes peripheral skin(case604-S0000)and an almost pure frame edge(case752-S0004); C002includes large pigmented areas(case793-S0002,case631-S0000); C003includes pale peripheral skin(case726-S0001,case796-S0004). Complement includes a corner fragment(case752-S0003)and surrounding tissue(case812-S0004). These limited observations do not justify medical sign names or population-wide semantic consistency. All human reviews remainUNREVIEWED.</p><h2>模型行为与范围 / Model behaviour and scope</h2><p>这101例MEL拟合病例，未遮挡预测34例MEL、38例NV、26例BKL、2例DF、1例AKIEC。解释的是固定MEL输出方向；错误预测也保留，不因结果差改样本。<br>Unmasked predictions on101MELfitting cases:34MEL,38NV,26BKL,2DF,1AKIEC. Concepts explain the fixed MEL output direction; misclassified cases remain. This is neither a multi-class external benchmark nor held-out concept validation.</p><p>源域分类器测试准确率89.32%，平衡准确率68.97%，MEL45/70；与本组数据和角色不同，不作受控因果比较。<br>Source classifier test accuracy89.32%,balanced68.97%,MEL45/70. Cohort and role differ; this is not a controlled causal comparison.</p><a href="assets/classifier-curves.png"><img class="wide" src="assets/classifier-curves.png" alt="Source classifier curves and confusion matrix"></a>'
 body+='<h2>验收、警告与待审阅 / Acceptance, warnings and review</h2><ul><li>433份文件哈希；101图顺序、掩码、特征、聚类及分配映射通过。全部423片段和101全图分类头代数核对；概念+补空间重构各8个保存样例通过，非所有样例重构。 /433files verified; all identities/mappings pass. FC algebra checked for423regions and101images; decomposition checked on8saved samples each.</li><li>本地首次回放遗漏Python随机状态失败；按seed43原序回放后逐像素一致，原失败保留。 /Initial local replay omitted Python RNG;seed43replay matches every pixel. Failed attempt retained.</li><li>SAM保存的predicted_iou有156/2886大于1（最大1.1276）；仅为原模型排序分数，不当作已校准真实IoU，不截断或改变排序。 /156/2886saved predicted-IoU scores exceed1(max1.1276); retained as model ranking scores,not calibrated measured IoU.</li><li>np.bool8弃用警告保留，无观测数值影响。 /Inheritednp.bool8deprecation retained; no observed numerical impact.</li><li>病例ID不等于患者ID；跨数据相似候选尚未由本次解决。 /Case identity is not patient identity; prior cross-dataset similarity questions remain.</li><li>尚无人类语义审阅、医学征象定位、外部概念验证或双向评价；不生成占位分数。 /Human semantics,sign localization,external concept evaluation and bidirectional evaluation not performed.</li></ul>'
 body+='<p>审阅模板：<a href="data/review-template.csv">CSV</a>；示例与掩码哈希：<a href="data/example-selection.json">selection.json</a>；<a href="data/discovery-acceptance.json">验收 / acceptance</a>；<a href="data/resolved_config.json">配置 / config</a>；<a href="data/concept_index.json">完整概念映射</a></p>'
 body+='<p>Execution649ce176c7d2e4e17b9e9be196fb0e7c33ef3e42 · Job28889071 · Classifier epoch27,bee957a2a13b894ea12a6f41cc9d5b38f206fc65b4a60fd1ebd81446099c0d37. Generated '+datetime.now(timezone.utc).isoformat()+'</p>'
 text='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HAM–Derm7pt | HU-MCD</title><style>body{font:16px/1.65 system-ui,sans-serif;color:#183040;background:#f7f8fa;margin:0;overflow-wrap:anywhere}main{max-width:1280px;margin:auto;padding:32px}h1,h2,h3{line-height:1.3}h2{margin-top:42px}a{color:#006c83}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px;border:1px solid #cbd5df;text-align:left;font-size:14px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}figure{margin:0;background:white;padding:8px}img{max-width:100%;height:auto}figcaption{font-size:12px;overflow-wrap:anywhere}.notice{padding:16px;border-left:5px solid #00798b;background:#eaf3f5}section{border-top:2px solid #b7c7d2;margin-top:36px}.wide{width:100%}summary{cursor:pointer;padding:12px}@media(max-width:800px){main{padding:16px}.grid{grid-template-columns:1fr}table{display:block;overflow:auto}}</style><main>'+body+'</main></html>'
 (out/'index.html').write_text(text)
 (data/'example-selection.json').write_text(json.dumps(selections,indent=2))
 # Never replace a human-edited template on regeneration.
 if not (data/'review-template.csv').exists():
  with (data/'review-template.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['concept_id','status','short_description','subject_part_background_mixed','random_supports_description','irrelevant_members','evidence_example_ids','reviewer','review_date','notes'])
   for e in examples:w.writerow([e['concept_id'],'UNREVIEWED']+['']*9)
 marker='<!-- HAM-DERM-FIT-CURRENT -->'
 note=marker+'<aside style="padding:18px;border:2px solid #00798b;margin:20px"><a href="medical-ham-derm/index.html">2026-09-24 新分支：HAM分类器 → Derm7pt101例概念拟合 / New branch: accepted HAM classifier → Derm7pt101case concept fitting</a><p>3个学习概念；72/423零特征；101例全量拟合，无留出验证。保留下方历史实验。 /3learned concepts;72/423zero features;all101cases fitted,no held-out validation. Earlier experiments below are retained.</p></aside><!-- /HAM-DERM-FIT-CURRENT -->'
 for n in ['index.html','medical.html']:
  f=S/n;s=f.read_text();s=re.sub(re.escape(marker)+r'.*?<!-- /HAM-DERM-FIT-CURRENT -->','',s,flags=re.S)
  pos=s.find('>',s.find('<main'))+1;assert pos>0;f.write_text(s[:pos]+note+s[pos:])
 (data/'generation.json').write_text(json.dumps(dict(script='scripts/research_hub/derm_fitting_results.py',script_sha256=sha(Path(__file__)),source_index_sha256=sha(P/'artifacts.json'),acceptance_sha256=sha(B/'discovery-acceptance.json'),panels=len(generated),selections=len(selections),new_predictions=0,private=True),indent=2))
 return dict(panels=len(generated),selections=len(selections),path=str(out/'index.html'))
if __name__=='__main__':
 import sys
 print(json.dumps(apply(Path(sys.argv[1])),indent=2))
