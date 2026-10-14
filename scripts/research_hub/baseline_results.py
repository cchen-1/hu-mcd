"""Incremental private baseline report extension, sourced from accepted local caches.
Called after build_research_hub or directly on a copy of the current live report.
Preserves the existing HU atlas/review schema and unrelated research pages.
"""
from pathlib import Path
import csv,json,html,re,shutil,hashlib,os
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];D=R/'artifacts/bunya/nine-class-baselines-20260923/final-20260924';A=D/'analysis'
def load(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def E(x):return html.escape(str(x),quote=True)
def bi(z,e):return f'<span data-l="zh">{E(z)}</span><span data-l="en" hidden>{E(e)}</span>'
def p(z,e,cls=''):return f'<p class="{cls}">{bi(z,e)}</p>'
def h(z,e,n=2):return f'<h{n}>{bi(z,e)}</h{n}>'
def link(url,z,e=None):return f'<a href="{E(url)}">{bi(z,e or z)}</a>'
def table(headers,data):return '<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+bi(*v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in rr)+'</tr>' for rr in data)+'</tbody></table></div>'
def fig(url,z,e):return f'<figure><a href="{url}" target="_blank"><img src="{url}" alt="{E(e)}" loading="lazy"></a><figcaption>{bi(z,e)}</figcaption></figure>'
def copytree(src,dst):
 dst.mkdir(parents=True,exist_ok=True)
 for f in src.rglob('*'):
  if not f.is_file():continue
  q=dst/f.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True)
  if q.exists():q.unlink()
  if f.suffix in ['.png','.npz']:os.link(f,q)
  else:shutil.copy2(f,q)

def apply(stage):
 S=Path(stage);summary=load(A/'summary.json');assert summary['status']=='PASS' and summary['accepted_stages']==63
 baseline=rows(A/'baseline_results.csv');points=rows(A/'class_points.csv');pooled=rows(A/'pooled_points.csv');stages=rows(A/'stage_status.csv');candidates=rows(A/'ACE_all_candidates.csv');ci=rows(A/'concept_index.csv');classes=list(dict.fromkeys(r['class_name'] for r in baseline));T=R/'artifacts/bunya/ten-class-review';atlas=load(T/'atlas_final.json');hu={c['name']:c for c in atlas['classes']}
 data=S/'hub/data/baselines-20260924';copytree(A,data)
 observations=load(D/'visual-observations.json');shutil.copy2(D/'visual-observations.json',data/'visual-observations.json')
 smpath=S/'hub/data/source-manifest.json';sm=load(smpath)
 if not (S/'hub/data/before-baselines-20260924-source-manifest.json').exists():shutil.copy2(smpath,S/'hub/data/before-baselines-20260924-source-manifest.json')
 for entry in sm:
  src=Path(entry['path'])
  if src.name in ['workstream_status.json','stage_status.csv'] and src.is_file():
   dst=S/entry['copy'];previous=dst.with_name('before-baselines-20260924-'+dst.name)
   if not previous.exists():shutil.copy2(dst,previous)
   shutil.copy2(src,dst);entry['sha256']=hashlib.sha256(dst.read_bytes()).hexdigest()
 for src in [A/'summary.json',R/'docs/NINE_CLASS_BASELINE_RESULTS_20260924.zh-en.md']:
  dst=data/src.name;shutil.copy2(src,dst)
  sm=[e for e in sm if e['path']!=str(src)];sm.append(dict(path=str(src),copy=str(dst.relative_to(S)),sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
 smpath.write_text(json.dumps(sm,indent=2)+'\n')
 obsmap={r['concept_id']:r for r in observations['records']}
 for name in ['scheduler.json','accounting.psv']:shutil.copy2(D/name,data/name)
 for f in list(D.glob('*.acceptance.json'))+list(D.glob('*-semantic-replay.json')):shutil.copy2(f,data/f.name)
 # This template preserves the current header, navigation, language switch and review IDs.
 template=(S/'baseline-progress.html').read_text()
 def page(filename,title,body):
  main='<main id="main"><div class="eyebrow">'+bi('私有研究报告 · 基线验收更新 2026-09-24','Private research report · Baseline acceptance updated 24 Sep 2026')+'</div>'+h(*title,1)+body+'</main>'
  text=re.sub(r'<main\b[^>]*>.*?</main>',lambda m:main,template,flags=re.S);text=re.sub(r'<title>.*?</title>','<title>'+E(title[0])+' | HU-MCD</title>',text)
  (S/filename).write_text(text)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 colors={'HU-MCD':'#087f8c','MCD':'#8055a2','ACE':'#ba6225','Random':'#68737d'};styles={'HU-MCD':('o','-'),'MCD':('s','--'),'ACE':('^','-.'),'Random':('. ',':')}
 out=S/'hub/figures/baselines-20260924';out.mkdir(exist_ok=True)
 def chart(name,rr,title,cap):
  f,axs=plt.subplots(2,2,figsize=(12.5,8),layout='constrained',height_ratios=[2,1])
  for col,direction in enumerate(['C-Deletion','C-Insertion']):
   for method,color in colors.items():
    q=[r for r in rr if r['method']==method and r['direction']==direction]
    xx=[float(r['deleted_pixel_percent' if col==0 else 'visible_pixel_percent']) for r in q];yy=[float(r['accuracy'])*100 for r in q];nn=[int(r['contributing_images']) for r in q]
    marker,ls=styles[method];marker=marker.strip()
    label='ACE released-code reference' if method=='ACE' else method
    axs[0,col].plot(xx,yy,marker=marker,ls=ls,color=color,label=label,ms=4,lw=1.4)
    axs[1,col].plot(xx,nn,marker=marker,ls=ls,color=color,ms=3,lw=1.2)
   axs[0,col].set(title=direction,ylabel='Target-class accuracy (%)',xlim=(-2,102),ylim=(-2,103));axs[0,col].legend(fontsize=9)
   axs[1,col].set(xlabel='Deleted pixels (%)' if col==0 else 'Visible pixels (%)',ylabel='Contributing images',ylim=(0,cap*1.05),xlim=(-2,102));axs[1,col].axhline(cap*.75,color='#555',ls=':',lw=.8)
   for ax in axs[:,col]:ax.grid(alpha=.15)
  f.suptitle(title+'\nNative computed points; changing denominators and pixel budgets',fontsize=14)
  for ext in ['png','svg']:f.savefig(out/(name+'.'+ext),dpi=300,bbox_inches='tight')
  plt.close(f)
 chart('ten-class-four-method',pooled,'Ten classes | 500 fixed validation images | All four methods',500)
 for name in classes:chart(name,[r for r in points if r['class_name']==name],name.replace('_',' ')+' | 50 fixed validation images',50)
 f,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained');yy=np.arange(10)
 for offset,method in [(-.23,'HU-MCD'),(0,'MCD'),(.23,'ACE')]:
  vals=[hu[n]['metrics']['learned_concepts'] if method=='HU-MCD' else int(next(r for r in baseline if r['class_name']==n and r['method']==method)['retained_concepts']) for n in classes]
  axs[0].barh(yy+offset,vals,.21,label='ACE nominal-retained' if method=='ACE' else method,color=colors[method])
 above=[int(next(r for r in baseline if r['class_name']==n and r['method']=='ACE')['nominal_retained_above_control']) for n in classes];below=[int(next(r for r in baseline if r['class_name']==n and r['method']=='ACE')['nominal_retained_below_control']) for n in classes]
 axs[1].barh(yy,above,label='Mean above random control',color='#8055a2');axs[1].barh(yy,below,left=above,label='Mean below random control',color='#ba6225',hatch='//')
 for ax in axs:ax.set(yticks=yy,yticklabels=[n.replace('_',' ') for n in classes],xlabel='Retained concept count');ax.invert_yaxis();ax.legend(fontsize=9)
 axs[0].set_title('Different methods, discovery budgets and retention rules');axs[1].set_title('ACE: passing nominal p < .01 is not a direction test')
 for ext in ['png','svg']:f.savefig(out/('counts-and-ACE-direction.'+ext),dpi=300,bbox_inches='tight')
 plt.close(f)
 transitions=rows(A/'initial_to_final.csv')
 for n in classes:
  if n=='golden_retriever':continue
  figt,axes=plt.subplots(2,1,figsize=(14,10),layout='constrained')
  for ax,method in zip(axes,['MCD','ACE']):
   rr=[r for r in transitions if r['class_name']==n and r['method']==method];clusters=sorted({int(r['initial_cluster_id']) for r in rr});targets=sorted({r['final_concept_id'] for r in rr});m=np.zeros((len(clusters),len(targets)),dtype=int)
   for row in rr:m[clusters.index(int(row['initial_cluster_id'])),targets.index(row['final_concept_id'])]+=int(row['count'])
   fractions=np.divide(m,m.sum(1,keepdims=True),out=np.zeros_like(m,dtype=float),where=m.sum(1,keepdims=True)>0)
   im=ax.imshow(fractions,aspect='auto',vmin=0,vmax=1,cmap='Blues');figt.colorbar(im,ax=ax,label='Fraction within original row')
   ax.set(title=method+' | Row-normalized identity transitions (not accuracy or purity)',ylabel='Initial cluster',xlabel='Final assignment / filtered candidate',yticks=range(len(clusters)),yticklabels=[str(k)+' (n='+str(m[i].sum())+')' for i,k in enumerate(clusters)],xticks=range(len(targets)),xticklabels=[t.replace('MCD-'+n+'-','').replace('ACE-R-'+n+'-','') for t in targets]);ax.tick_params(axis='x',rotation=45,labelsize=8);ax.tick_params(axis='y',labelsize=7)
  figt.suptitle(n+' | Initial cluster to final concept; cluster -1 = initial SSC outlier',fontsize=14)
  for ext in ['png','svg']:figt.savefig(out/(n+'-transitions.'+ext),dpi=300,bbox_inches='tight')
  plt.close(figt)
 prefix='hub/data/baselines-20260924/';figroot='hub/figures/baselines-20260924/'
 body=p('本批 63 个阶段均为 COMPLETED / 0:0，已收集并通过本地独立验收。Golden 既有结果直接复用；十类、四种方法、两个方向共 80 个类别×方法×评价单元现已验收。','All 63 batch stages are COMPLETED / 0:0, collected and independently accepted locally. Existing Golden results are reused. All 80 class × method × direction cells across ten classes and four methods are now accepted.','lead')
 body+=p('本轮没有新增模型实验、预测、调参、重新选样或重提作业。验收覆盖输入与版本链、掩码／特征对应、MCD 重构、ACE 梯度与检验，以及实际逐图掩码序列；不等于概念语义已通过人工验证。','This update adds no model experiments, predictions, tuning, reselection or job retries. Acceptance covers input/version chains, mask–feature identity, MCD reconstruction, ACE gradients/tests and actual per-image mask trajectories; semantic human validation remains separate.')
 body+=fig(figroot+'ten-class-four-method.png','从 500 张逐图轨迹按原生步数汇总，仅保留贡献数 >375 的点。连线只连接已有点；不是相同像素预算比较，未补端点。','Pool all 500 image trajectories by native step, keeping n>375. Lines connect existing points only; not an equal-pixel-budget comparison, with no padded endpoints.')
 body+=h('如何阅读首个实际扰动点','Reading the first actual perturbation point')
 first=[]
 for method in ['HU-MCD','MCD','ACE','Random']:
  de=next(r for r in pooled if r['method']==method and r['direction']=='C-Deletion' and int(r['native_step'])==1)
  ins=next(r for r in pooled if r['method']==method and r['direction']=='C-Insertion' and int(r['native_step'])==1)
  first.append([method,f"{float(de['deleted_pixel_percent']):.2f}%",f"{float(de['accuracy'])*100:.2f}%",f"{float(ins['accuracy'])*100:.2f}%",de['contributing_images']])
 body+=table([('方法','Method'),('删除／插入像素','Deleted / inserted pixels'),('删除后准确率','Accuracy after deletion'),('插入后准确率','Accuracy after insertion'),('贡献图片','Contributors')],first)
 body+=p('该表仅描述各方法第一个已计算步骤：HU-MCD 的下降较大、插入准确率较高，但各图片的像素预算、排序规则和 ACE 分母并不相同，不能据此给出受控方法排名或显著性结论。Random 首步只覆盖约1%像素，尤其不能直接与约23%的概念步骤比较。完整曲线及逐类结果在下方。','This table describes each method’s first computed step: HU-MCD has a larger accuracy drop and higher insertion accuracy here. Per-image pixel budgets, ordering rules and the ACE denominator differ, so this is not a controlled ranking or significance result. Random covers only about1% at its first step and cannot be directly compared with concept steps covering about23%. Full curves and class results follow.')
 body+=h('十类发现与评价','Ten-class discovery and evaluation')
 rr=[]
 for n in classes:
  m=next(r for r in baseline if r['class_name']==n and r['method']=='MCD');a=next(r for r in baseline if r['class_name']==n and r['method']=='ACE');dest='baseline-details.html' if n=='golden_retriever' else 'baseline-'+n+'.html'
  rr.append([link(dest,n),hu[n]['metrics']['learned_concepts'],m['retained_concepts'],f"{float(m['completeness']):.6f}",a['retained_concepts'],a['nominal_retained_above_control']+' / '+a['nominal_retained_below_control'],bi('四方法、双向已验收','Four methods, both directions accepted')])
 body+=table([('类别 / 详情','Class / details'),('HU 学习概念','HU learned'),('MCD 概念','MCD concepts'),('MCD completeness','MCD completeness'),('ACE 名义保留','ACE nominal-retained'),('ACE 高于 / 低于控制','ACE above / below control'),('评价','Evaluation')],rr)
 body+=fig(figroot+'counts-and-ACE-direction.png','补空间不计入学习概念数量；ACE 仅表示参考代码的筛选结果。','Complements excluded from learned counts; ACE counts reflect released-code filtering only.')
 body+=h('比较范围与主要限制','Comparison scope and main limitations')+p(f"ACE 共保留 {summary['ACE_nominal_retained']} 个候选，其中 {summary['ACE_retained_above_control']} 个均值高于、{summary['ACE_retained_below_control']} 个低于随机控制。双侧名义 p 值参与筛选；重复负样本和身份交集使其不能解释为已校准的独立重复证据。",f"ACE retains {summary['ACE_nominal_retained']} candidates: {summary['ACE_retained_above_control']} have means above and {summary['ACE_retained_below_control']} below random control. Two-sided nominal p values govern retention; repeated negatives and identity overlaps prevent calibrated independent-replicate interpretation.",'note')
 body+=p('HU-MCD/MCD 使用每类 400 张发现图片，ACE 使用固定 400 的前 50 张。所有方法共用每类固定 50 张验证、权重与精度，但分割、排名和贡献人数不同；不能把曲线差异单独归因于概念易理解程度。','HU-MCD/MCD use 400 discovery images per class; ACE uses the first 50 of the frozen 400. All methods share each class’s 50 validation inputs, weights and precision, while segmentation, ranking and contributors differ. Curve differences alone do not establish human understandability.')
 body+=p('ACE 多个平均 TCAV 相同的候选存在排序并列。本地默认 argsort 顺序可能不同；报告与掩码重放沿用实际记录的合法降序顺序，不重新排序结果。具体差异见重放记录。','ACE candidates can tie in mean TCAV. Local default argsort can produce a different tie order. The report and mask replay preserve the recorded valid descending order; results are not reranked. Replay receipts document the differences.')
 body+=p('全图初始预测的正确／错误身份在各类四种方法间完全一致。原规则的缺失端点、变化分母、Random 网格边缘及全零特征行为仍保留。名义显著、预测影响和可理解性是三个不同结论。','Initial full-image correctness identities agree across all four methods in each class. Original omitted endpoints, changing denominators, Random grid borders and zero-feature behavior are retained. Nominal significance, prediction impact and understandability are separate conclusions.')
 body+=h('配置、精度与来源','Configuration, precision and provenance')+p('九类执行提交 1e094cd7fa9f3e6990e6aad1294e85b96b82905e。MCD：layer4 7×7 空间特征，原 SSC 两次求解、q=.75，k=3…19 首次 completeness>.5；ACE：25 候选、覆盖与大小过滤、20 次 CAV／TCAV、原双侧名义检验。两方向复用同一次拟合。','Nine-class execution commit: 1e094cd7fa9f3e6990e6aad1294e85b96b82905e. MCD: layer4 7×7 spatial features, two original SSC passes, q=.75, first completeness>.5 over k=3…19. ACE: 25 candidates, coverage/size filters, 20 CAV/TCAV runs and original two-sided nominal test. Each fit serves both evaluation directions.')
 body+=p('保留 FP32、batch8、matmul TF32=false、cuDNN TF32=true、matmul precision=highest；实际 resolved config 为准。Golden 与九类封装代码不同，核心原方法源码哈希与协议逐项核对，差异文件保留。','FP32, batch8, matmul TF32=false, cuDNN TF32=true and matmul precision=highest are retained; resolved configs are authoritative. Golden and nine-class wrapper versions differ; core method hashes and protocols are checked, with code diffs preserved.')
 body+=h('完整阶段、异常与可下载证据','All stages, anomalies and downloadable evidence')
 body+=table([('类别','Class'),('方法','Method'),('阶段','Stage'),('作业','Job'),('State / ExitCode','State / ExitCode'),('耗时','Elapsed'),('验收','Acceptance')],[[r['class_name'],r['method'],r['stage'],r['job_id'],r['state']+' / '+r['exit_code'],r['elapsed'],link(prefix+f"BASE26-{r['method']}-{r['class_name']}-{r['stage']}.acceptance.json",'已验收','Accepted')] for r in stages])
 for f in ['summary.json','baseline_results.csv','class_points.csv','first_change_summary.csv','pooled_points.csv','pooled_contributors.csv','concept_index.csv','review_template.csv','ACE_all_candidates.csv','ACE_round_scores.csv','ACE_role_intersections.csv','ACE_CAV_round_intersections.csv','transition_summary.csv','initial_to_final.csv','warning_inventory.json','visual-observations.json','bindings.json','scheduler.json','MCD-Golden-to-multiclass.diff','ACE-Golden-to-multiclass.diff']:
  body+='<p>'+link(prefix+f,f)+'</p>'
 body+=p('完整 warning_inventory 保留逐条原始警告和日志位置，不能把匹配到的“error”字样自动判为作业失败。统一异常记录区分观察、影响、处理与剩余不确定性。','The complete warning inventory preserves individual warnings and log locations; an “error” text match is not automatically a failed job. The unified registry distinguishes observations, impact, action and remaining uncertainty.')+link('issues.html','统一疑点与限制','Unified issues and limitations')
 page('baseline-progress.html',('十类四方法：最终结果与九类基线验收','Ten classes, four methods: final results and baseline acceptance'),body)
 for n in classes:
  if n=='golden_retriever':continue
  v=load(D/'visuals'/n/'index.json');assert v['status']=='PASS';copytree(D/'visuals'/n,S/'hub/baseline-concepts'/n);lookup={e['id']:e for e in v['examples']};vp='hub/baseline-concepts/'+n+'/'
  b=link('baseline-progress.html','← 十类比较与验收','← Ten-class comparison and acceptance')+p('所有概念按固定 ID 纳入，未作语义质量筛选。以下为待人工审阅的缓存示例；不对概念命名或宣称纯度。','All concepts are included by stable ID without semantic-quality selection. These cached examples await human review; no semantic names or purity claims are assigned.','lead')+fig(figroot+n+'.png','同时阅读预测准确率、像素比例与贡献图片数。','Read prediction accuracy, pixel fraction and contributors together.')
  b+=fig(figroot+n+'-transitions.png','初始聚类→最终分配的逐行比例。MCD含最初SSC离群行；ACE包含后来被过滤的候选。不同拟合／分配规则会改变成员，这不是分类准确率、纯度或错误率。','Row-normalized initial-cluster→final-assignment transitions. MCD includes initial SSC outliers; ACE includes later-filtered candidates. Different fitting/assignment rules change membership; this is not accuracy, purity or error rate.')
  b+=h('完整概念索引','Complete concept index')+table([('稳定 ID','Stable ID'),('方法','Method'),('初始成员','Initial members'),('最终训练 / 验证','Final train / validation'),('排名','Rank'),('审阅','Review')],[[link('#'+c['id'],c['id']),c['method'],c['initial_members'] if c['initial_members'] is not None else bi('不适用','Not applicable'),str(c['final_training_members'])+' / '+str(c['final_validation_members']),c.get('global_rank') or bi('补空间单列','Complement separate'),bi('未审阅','Unreviewed')] for c in v['cards']])
  b+=h('各方法前五个学习概念','Top five learned concepts per method')
  for m in ['MCD','ACE — HU-MCD released-code reference']:
   selected=sorted([c for c in v['cards'] if c['method']==m and not c['is_complement']],key=lambda c:c['global_rank'])[:5]
   b+='<p>'+E(m)+': '+(' · '.join(link('#'+c['id'],c['id']) for c in selected) or bi('没有保留概念','No retained concepts'))+'</p>'
  b+=p('MCD 排名来自全局权重重要性；ACE 排名来自原 TCAV 均值及实际并列顺序；均不是语义质量排序。MCD 成员数单位为 7×7 空间单元，ACE 为 SLIC 片段。','MCD ranks use global weight-space importance; ACE ranks use TCAV means and recorded tie order. Neither ranks semantic quality. MCD counts are 7×7 spatial cells; ACE counts are SLIC segments.')
  b+=p('MCD 三联图第一幅为 224×224 画布，另提供实际输入原尺寸链接；并非原图像素分辨率。MCD top 展示使用原 >.4 阈值，随机最终视图使用 benchmark resize→argmax，两者不能混淆。初始 MCD 视图为原 7×7 聚类单元的展示放大。ACE 原型使用初始成员；验证最近成员是展示扩展。ACE 特征来自紧裁剪，图中全图上下文仅用于审阅。','MCD triplets use a 224×224 canvas; links provide actual inputs at original resolution. MCD top display uses the original >.4 threshold; random final views use benchmark resize→argmax. Initial MCD views enlarge the original 7×7 cluster cells for display only. ACE prototypes are initial members; nearest validation members are a display extension. ACE features use tight crops; full context here is for review only.','note')
  for c in v['cards']:
   b+='<section id="'+E(c['id'])+'">'+h(c['id'],c['id'])+p('当前审阅：未审阅。每组三图从左到右为：输入画布、掩码叠加、仅保留所选区域。青色为所选区域，灰色为被遮蔽区域。点击打开原尺寸图。','Review status: unreviewed. Each triplet shows, left to right: input canvas, mask overlay, selected region only. Cyan marks selected regions; grey marks hidden regions. Click to view full-size images.')
   if c['id'] in obsmap:
    observation=obsmap[c['id']];assert hashlib.sha256((D/'visuals'/n/c['comparison']).read_bytes()).hexdigest()==observation['figure_sha256']
    b+=p('AI 有限样本观察（待人工确认）：'+observation['observations']['zh'],'AI limited-sample observation (awaiting human confirmation): '+observation['observations']['en'],'note')
   if c['is_complement']:b+=p('补空间单独展示，不计入学习概念，不等同背景。','Complement shown separately; excluded from learned counts and not synonymous with background.')
   meta={k:x for k,x in c.items() if k not in ['slots','comparison','id','review_status']};b+='<details><summary>'+bi('数值、原始索引与身份','Values, original indices and identity')+'</summary><pre>'+E(json.dumps(meta,indent=2))+'</pre></details>'
   if c['method'].startswith('ACE'):b+='<p>'+link('issues.html#SCI-ACE','ACE参考协议限制','ACE reference-protocol limits')+' · '+link('issues.html#BASE26-006','并列顺序与重放','Tie order and replay')+'</p>'
   if c['method'].startswith('ACE'):b+=p(f"名义 p={c['nominal_p']:.4g}；TCAV均值差={c['mean_difference']:+.4f}。名义保留不表示高于控制。",f"Nominal p={c['nominal_p']:.4g}; mean TCAV difference={c['mean_difference']:+.4f}. Nominal retention does not imply an advantage over control.")
   b+='<p>'+link(vp+c['comparison'],'高分辨率六视图导出','High-resolution six-view export')+'</p><div class="baseline-example-grid">'
   for slot in c['slots']:
    e=lookup.get(slot['example']);title=slot['title'];tz=title.replace('Training: initial cluster random','训练：初始聚类随机').replace('Training discovery','训练发现').replace('Training','训练').replace('Validation','验证').replace('source top display (> .4)','原 top 展示（>.4）').replace('final-assignment random image','最终分配随机图片').replace('top','原型／最近成员').replace('random','最终分配随机片段')
    b+='<article>'+h(tz,title,3)
    if e:b+=fig(vp+e['image'],e['input']['prepared_name']+'；所选 '+f"{e['selected_pixels']/e['canvas_pixels']:.2%}",e['input']['prepared_name']+'; selected '+f"{e['selected_pixels']/e['canvas_pixels']:.2%}")+ '<p class="source">'+E(e.get('segment_id',e['id']))+'</p><p>'+link(vp+e['original_file'],'打开未经展示缩放的实际输入原图','Open actual input at original resolution')+'</p><details><summary>'+bi('选择规则与证据','Selection rule and evidence')+'</summary><p>'+E(e['rule'])+'</p><p>'+E(e['mask_sha256'])+'</p>'+link(vp+e['mask'],'精确展示掩码','Exact display mask')+'</details>'
    else:b+=p('不适用，或没有合格成员；未替换样本。','Not applicable or no eligible member; no replacement was selected.')
    b+='</article>'
   b+='</div></section>'
  b+='<p>'+link(vp+'index.json','完整示例身份、选择规则与哈希','All example identities, selection rules and hashes')+' · '+link(prefix+'review_template.csv','可填写审阅模板','Fillable review template')+'</p>'
  page('baseline-'+n+'.html',(n+'：MCD 与 ACE 概念及评价',n+': MCD and ACE concepts and evaluation'),b)
 # Incremental wording updates on current hub pages only; legacy evidence remains historical.
 replacements={
 'HU-MCD＋Random 十类；MCD、ACE 仅 Golden。':'十类四方法、固定 500 张验证，双向评价已验收。',
 'HU-MCD + Random for ten classes; MCD and ACE only Golden.':'All four methods, ten classes and 500 fixed validation images accepted in both directions.',
 '九类 MCD／ACE 的 63 个阶段已提交，结果尚未验收，因此四方法十类对照未齐。本批已获授权，无需重复申请；ACE 协议限制保留。':'九类 63 个阶段已验收，四方法十类对照已齐。仍非作者原始清单逐张复现；ACE 参考协议限制、方法间发现预算差异及人类实验缺口保留。',
 'The 63 stages for nine-class MCD/ACE are submitted but results are unaccepted, so the full comparison remains incomplete. This batch is already authorized; ACE limitations remain.':'All 63 nine-class stages are accepted and the four-method ten-class comparison is complete. Exact author input identity is not claimed; ACE protocol limits, differing discovery budgets and the missing human study remain.',
 '重复负样本、配对双侧检验；保留三概念均值低于对照。另有 SLIC 末标签遗漏和交叉熵梯度行为。':'Golden 保留三概念均值低于控制；十类高于／低于方向分开统计，见基线详情。重复负样本、双侧检验、SLIC 末标签遗漏及交叉熵梯度行为均保留。',
 'Repeated negatives and paired two-sided testing retain three concepts below control mean. Released code also omits the final SLIC label and uses cross-entropy gradients.':'Golden retains three concepts below control; ten-class above/below directions are separately reported in baseline details. Repeated negatives, two-sided tests, SLIC final-label omission and CE gradients remain.',
  '十类发现和 HU-MCD＋Random 双向评价已验收；MCD、ACE 仅 Golden 完成。':'十类 HU-MCD、MCD、ACE 发现及四方法双向评价均已验收。',
 'Discovery and HU-MCD + Random evaluation accepted for ten classes; MCD and ACE completed only for Golden.':'Discovery and four-method bidirectional evaluation accepted across all ten classes.',
 '九类 MCD／ACE 的 63 个阶段作业已提交，等待结果验收；':'九类 MCD／ACE 的 63 个阶段已验收；',
 'The 63 stages for nine-class MCD/ACE have been submitted and await result acceptance;':'All 63 stages for nine-class MCD/ACE are accepted;',
 '九类基线：已提交，尚未验收':'九类基线：已完成并验收',
 'Nine-class baselines: submitted, not accepted':'Nine-class baselines: completed and accepted',
 '当前只有提交后状态快照；未收集结果，不把排队或完成状态当成科研验收。':'已刷新最终调度状态，并完成输入、数组、掩码序列与评价验收。',
 'Only the post-submission snapshot is available; results are not collected, and scheduling/completion is not scientific acceptance.':'Final scheduler states, input identities, scientific arrays, mask trajectories and evaluations are now verified.',
 'MCD、ACE 已提交，未验收':'MCD、ACE 已验收',
 'MCD, ACE submitted, not accepted':'MCD, ACE accepted',
 '九类已提交阶段（无新结果）':'十类四方法最终结果与新增概念',
 'Nine-class submitted stages (no new results)':'Ten-class four-method results and new concepts',
 '等待已提交九类 MCD／ACE 完成后统一验收；':'九类 MCD／ACE 已验收；',
 'Accept the already-submitted nine-class MCD/ACE results after completion;':'Nine-class MCD/ACE results are accepted;',
 '保全检查发现新提交版本及63阶段回执；已提交，结果未收集验收。':'63阶段最终 COMPLETED/0:0；结果及缓存重放已验收，历史提交回执保留。',
 '43/50 图至少有一个缺失学习概念掩码；原 NaN 得分／跳过路径存在，最终预测有限。':'Golden 的43/50图至少缺一个学习掩码；其他类别见本批重放记录。保留原NaN得分／跳过路径，预测有限。',
 '43/50 images have an absent learned-concept mask; original NaN-score/skipping paths occur, with finite predictions.':'In Golden,43/50 images have an absent learned mask; other classes have their own replay records. Original NaN-score/skipping paths remain; predictions are finite.',
 'Preservation checks found a new committed version and63stage receipts: submitted, but results uncollected/unaccepted.':'All 63 stages completed with exit0:0 and passed result/cache-replay acceptance; original submission receipts retained.'}
 for f in [S/'index.html',S/'reproduction.html',S/'baseline-details.html',S/'paper-comparison.html',S/'issues.html',*list((S/'classes').glob('*.html'))]:
  text=f.read_text()
  for old,new in replacements.items():text=text.replace(E(old),E(new))
  if f.name=='issues.html':
   text=re.sub(r'<section id="BASE26-006">.*?</section>','',text,flags=re.S)
   text=text.replace('</main>','<section id="BASE26-006">'+h('BASE26-006：并列排序与实际重放','BASE26-006: rank ties and actual replay')+p('ACE相同TCAV的并列候选可能有不同本地排序。核对实际顺序为合法降序后，全部实际掩码重放通过；保留原顺序，未改方法。','Equal-TCAV ACE candidates can have different local tie orders. Recorded valid descending orders reproduce all actual masks; actual ordering is preserved without changing the method.')+link('baseline-progress.html','完整结果、警告及重放回执','Full results, warnings and replay receipts')+'</section></main>')
  if f.name=='reproduction.html':
   text=re.sub(r'<section id="baseline-four-method-figure">.*?</section>','',text,flags=re.S)
   anchor=h('Golden Retriever 基线','Golden Retriever baselines')
   if anchor in text:text=text.replace(anchor,'<section id="baseline-four-method-figure">'+fig(figroot+'ten-class-four-method.png','十类四方法完整评价；原生点、变化分母，详见下方结果入口。','Complete ten-class four-method evaluation: native points and changing denominators; details at the result link.')+'</section>'+anchor)
  if f.name in ['index.html','reproduction.html','paper-comparison.html']:
   text=re.sub(r'<section id="baseline-update-20260924">.*?</section>','',text,flags=re.S)
   text=text.replace('</main>','<section id="baseline-update-20260924">'+p('2026-09-24 基线更新：十类完整比较与新概念展示已纳入。','24 Sep 2026 baseline update: complete ten-class comparisons and new concept views are included.')+link('baseline-progress.html','查看本轮最终结果与限制','Review final results and limitations')+'</section></main>')
  f.write_text(text)
 # Keep the downloadable issue summary synchronized with its current HTML view.
 issue_csv=S/'hub/data/report-issues.csv'
 if issue_csv.exists():
  rr=rows(issue_csv);fields=list(rr[0])
  for row in rr:
   for key,value in row.items():
    for old,new in replacements.items():value=value.replace(old,new)
    row[key]=value
  with issue_csv.open('w',newline='',encoding='utf-8-sig') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
 css=S/'hub/style.css';style='\n/* Accepted baseline extension 20260924 */\n.baseline-example-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1.6rem}.baseline-example-grid article{min-width:0}.baseline-example-grid figure img{width:100%;height:auto}.baseline-example-grid .source{overflow-wrap:anywhere}.baseline-example-grid details p{overflow-wrap:anywhere} @media(max-width:800px){.baseline-example-grid{grid-template-columns:1fr}}\n'
 if 'Accepted baseline extension 20260924' not in css.read_text():css.write_text(css.read_text()+style)
 validation=load(S/'hub/data/report-validation.json');validation.update(baseline_update=summary['generated_at'],baseline_stages_accepted=63,MCD_direction_class_cells=20,ACE_direction_class_cells=20,all_method_direction_class_cells=80,new_baseline_concept_cards=sum(len(load(D/'visuals'/n/'index.json')['cards']) for n in classes if n!='golden_retriever'))
 (S/'hub/data/report-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
 shutil.copy2(R/'docs/ANOMALIES.jsonl',S/'hub/data/ANOMALIES.jsonl')
 print(json.dumps(dict(status='PASS',stage=str(S),summary=summary['generated_at'],concept_cards=validation['new_baseline_concept_cards'])))
 # Preserve the unified class comparisons and interpretations in future rebuilds.
 if (R/'artifacts/report-interpretation-20260924/facts.json').exists():
  if __package__:
   from .interpretation import apply as apply_interpretation
  else:
   from interpretation import apply as apply_interpretation
  apply_interpretation(S)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('stage',type=Path);args=parser.parse_args();apply(args.stage)
