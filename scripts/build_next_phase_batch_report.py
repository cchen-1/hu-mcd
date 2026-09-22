"""Private offline addition to the existing report, from accepted cached results.
No model calls, experiment reruns or changed research samples. Deterministic views.
"""
import csv,hashlib,html,io,json,shutil,sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/bunya/batch-review-20260922';B=OUT/'bundle'
for d in ['figures','examples','images']: (B/d).mkdir(parents=True,exist_ok=True)
J=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def table(p,rr):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def esc(x):return html.escape(str(x))
def render_table(rows,keys):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+esc(k)+'</th>' for k in keys)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc('未运行 / Not run' if r.get(k) is None else r[k])+'</td>' for k in keys)+'</tr>' for r in rows)+'</tbody></table></div>'
STYLE='''body{font:17px/1.55 system-ui,sans-serif;color:#172b3a;background:#f8fafb;max-width:1440px;margin:auto;padding:32px}h1{font-size:32px}h2{margin-top:44px;border-bottom:2px solid #1d7777;padding-bottom:8px}.en{color:#4b606b}a{color:#075f84}p{max-width:1120px}.table{overflow:auto}table{border-collapse:collapse;background:white;font-size:14px}th,td{padding:9px 12px;border:1px solid #ccd8df;text-align:left}th{background:#e8f0f3}figure{margin:12px 0;background:white;border:1px solid #ccd8df;padding:12px}figure img{max-width:100%;height:auto}figcaption{font-size:14px;overflow-wrap:anywhere}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:20px}.notice{padding:16px;background:#fff2dd;border-left:5px solid #b96618}.good{background:#e6f4ef;padding:16px}textarea{width:95%;min-height:75px}label{display:block;margin:12px 0}select,input,textarea,button{font:inherit;padding:6px}nav{display:flex;gap:16px;flex-wrap:wrap}.pill{padding:3px 7px;background:#e8f0f3;border-radius:4px} @media(max-width:900px){body{padding:16px}.twocol{grid-template-columns:1fr}}'''
def page(title,body):return '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+esc(title)+'</title><style>'+STYLE+'</style><body data-dataset="medical-and-robustness-batch-20260922"><script id="concept-ids" type="application/json">[]</script><nav><a href="index.html">本批总览 / Batch overview</a><a href="medical.html">医学概念 / Medical concepts</a><a href="robustness.html">掩码敏感性 / Mask sensitivity</a></nav><h1>'+esc(title)+'</h1>'+body+'</body></html>'
a=J(OUT/'acceptance-both.json');A=a['robustness'];M=a['medical'];d1=ROOT/M['collection'];d0=J(ROOT/'artifacts/bunya/medical-discovery-20260922/completed-prerequisite-evidence/dataset_manifest.json')
for f in OUT.glob('*.csv'):shutil.copy2(f,B/f.name)
shutil.copy2(OUT/'acceptance-both.json',B/'acceptance.json');shutil.copy2(OUT/'final-sacct.txt',B/'final-sacct.txt')
source=Path('/mnt/c/Users/uqcche38/Downloads/dermamnist_corrected_224.npz')
images={}
with np.load(source,allow_pickle=False) as z:
    for role,split in [('training','train'),('held_out','test')]:
        pixels=z[split+'_images']
        for row in d0[role]:
            rgb=pixels[int(row['array_row'])];iid=row['image_id']
            if hashlib.sha256(rgb.tobytes()).hexdigest()!=row['input_pixel_sha256']:raise ValueError('Local image identity '+iid)
            im=Image.fromarray(rgb);path=B/'images'/(iid+'.png');im.save(path)
            if sha(path)!=row['input_sha256']:raise ValueError('D0 lossless input identity '+iid)
            images[iid]=rgb.copy()
raw={};masks={};assign={};rawrows={};sci={};examples={}
for role in ['training','held_out']:
    raw[role]=J(d1/(role+'_raw_segments.json'));assign[role]={r['segment_id']:r for r in J(d1/(role+'_assignments.json'))};rawrows[role]={r['segment_id']:r for r in raw[role]}
    with np.load(d1/'scientific'/(role+'.npz')) as z:sci[role]={k:z[k] for k in z.files}
    examples[role]=J(d1/(role+'_example_index.json'))
    for row in d0[role]:
        with np.load(d1/'cache'/role/(row['image_id']+'_effective_masks.npz')) as z:masks[row['image_id']]=z['masks']

selection=[]
def triptych(role,sid,selection_rule):
    r=rawrows[role][sid];rgb=images[r['image_id']];mask=masks[r['image_id']][r['raw_index']]
    if hashlib.sha256(mask.tobytes()).hexdigest()!=r['effective_mask_sha256']:raise ValueError('Display mask identity')
    overlay=rgb.copy();overlay[mask]=(.6*rgb[mask]+.4*np.array([0,195,175])).astype(np.uint8)
    yy,xx=np.indices(mask.shape);gray=np.where((xx//12+yy//12)%2,210,235).astype(np.uint8)
    selected=np.repeat(gray[:,:,None],3,axis=2);selected[mask]=rgb[mask]
    im=Image.new('RGB',(224*3+24,258),'white');dr=ImageDraw.Draw(im)
    for x,ar,title in [(0,rgb,'Original (224 x 224)'),(236,overlay,'Teal = SELECTED'),(472,selected,'Grey = HIDDEN')]:
        im.paste(Image.fromarray(ar),(x,28));dr.text((x+4,6),title,fill='#152833')
    f=B/'examples'/(sid+'.png')
    if not f.exists():im.save(f)
    ar=assign[role].get(sid);cid=ar['concept_id'] if ar else 'INVALID_ZERO_FEATURE'
    selection.append(dict(role=role,segment_id=sid,image_id=r['image_id'],concept_id=cid,selection_rule=selection_rule,effective_area_fraction=r['effective_pixels']/50176,mask_sha256=r['effective_mask_sha256']))
    return '<figure><a href="examples/'+esc(f.name)+'"><img loading="lazy" width="696" height="258" src="examples/'+esc(f.name)+'" alt="Original, selected-region overlay, selected-only view"></a><figcaption>'+esc(sid)+'<br>'+esc(cid)+' · area '+f'{r["effective_pixels"]/50176:.1%}'+' · <a href="images/'+esc(r['image_id'])+'.png">原尺寸输入 / Native input</a></figcaption></figure>'

fig,ax=plt.subplots(figsize=(12,6));classes=[r['class_name'] for r in A['statuses']];conditions=['erosion1','erosion2','dilation1','dilation2'];mat=np.full((10,4),np.nan)
for r in A['rows']:
    if r['condition'] in conditions:mat[classes.index(r['class_name']),conditions.index(r['condition'])]=r['image_mean_retention']
cm=plt.colormaps['YlGnBu'].copy();cm.set_bad('#dddddd');im=ax.imshow(mat,vmin=0,vmax=1,cmap=cm,aspect='auto');ax.set_yticks(range(10),classes);ax.set_xticks(range(4),conditions)
for i in range(10):
    for j in range(4):ax.text(j,i,'Not run' if np.isnan(mat[i,j]) else f'{mat[i,j]:.1%}',ha='center',va='center',color='white' if mat[i,j]>.75 else 'black')
ax.set_title('Fixed masks / fixed concepts: image-mean assignment retention');fig.colorbar(im,ax=ax,label='Valid baseline retained / valid baseline');fig.tight_layout();fig.savefig(B/'figures/robustness.png',dpi=260);fig.savefig(B/'figures/robustness.pdf');plt.close(fig)
xt=np.zeros((3,2),int)
for r in M['cluster_to_assignment']:xt[r['initial_cluster'],r['final_assignment']]=r['count']
fig,(ax,ax2)=plt.subplots(1,2,figsize=(12,4.6));im=ax.imshow(xt,cmap='Blues');ax.set_yticks(range(3),['Cluster 0 (retained)','Cluster 1 (filtered)','Cluster 2 (filtered)']);ax.set_xticks([0,1],['HUMCD-MEL-C001','Complement'])
for i in range(3):
    for j in range(2):ax.text(j,i,str(xt[i,j]),ha='center',va='center',color='white' if xt[i,j]>300 else 'black')
ax.set_title('Initial cluster to final assignment\nThis is not classification accuracy')
roles=['Training','Held-out'];lear=[s['learned_assignments']/s['retained_regions'] for s in M['splits']];oc=[1-x for x in lear]
ax2.bar(roles,lear,label='Learned C001',color='#197e86');ax2.bar(roles,oc,bottom=lear,label='Complement',color='#bac3ce');ax2.set_ylim(0,1);ax2.set_ylabel('Share of valid, nonzero regions');ax2.legend(loc='upper center',bbox_to_anchor=(.5,-.1),ncol=2)
for i,s in enumerate(M['splits']):ax2.text(i,.3,f"{s['learned_assignments']}/{s['retained_regions']}\n{lear[i]:.1%}",ha='center',color='white');ax2.text(i,.86,f"{s['complement_assignments']}/{s['retained_regions']}",ha='center')
fig.tight_layout();fig.savefig(B/'figures/medical-summary.png',dpi=260);fig.savefig(B/'figures/medical-summary.pdf');plt.close(fig)

medical='<p>400个不同病灶的训练图与全部70张留出图（61病灶）已通过工程验收。3个初始cluster中仅807片段的cluster保留；31和11片段的cluster按固定50阈值过滤。不是将补空间计为第二个学习概念。</p><p class="en">Engineering acceptance covers all400 discovery images/distinct lesions and70 held-out images/61 lesions. Only the807-member initial cluster survives the fixed50-member threshold; clusters of31 and11 are filtered. The complement is not a second learned concept.</p>'
medical+='<p>模型为固定epoch10医学ResNet50；采样seed43与分类器训练seed43是不同RNG用途。Completeness=0.662911，表示目标权重方向被学习子空间覆盖的范数比例，不是语义质量。C001维数32，全局重要性0.439451＝目标权重在该子空间分量的平方范数／权重平方范数，排名1/1。局部分配使用每片段分量范数／特征范数。</p><p class="en">The epoch10 classifier is frozen. Sampling and classifier-training seed43 have distinct roles. Completeness0.662911 measures target-weight subspace coverage, not semantic quality. C001 has32 dimensions; global importance0.439451 is the squared component norm relative to the squared target-weight norm, rank1/1. Local assignment uses component norm divided by feature norm.</p>'
medical+=render_table(M['splits'],['role','images','lesions','raw_regions','retained_regions','zero_features','learned_assignments','complement_assignments','MEL_predictions'])+'<figure><img src="figures/medical-summary.png" alt="Initial cluster to final assignment and split assignment shares"><figcaption><a href="figures/medical-summary.pdf">PDF</a> · <a href="medical-cluster-to-assignment.csv">CSV</a></figcaption></figure>'
medical+='<div class="notice">零特征：训练68/917（7.42%），留出14/157（8.92%）；输入有效掩码均非空。这是观测到的模型／mask路径现象，尚未逐例证明消失机制。3个cluster数量由原片段数启发式产生；不能据此断言医学仅有一个语义概念。<br><span class="en">Zero features:68/917 training and14/157 held-out despite nonempty effective masks. Their per-region disappearance mechanism has not been audited. The initial3-cluster count follows the released segment-count heuristic; one retained subspace does not imply one medical semantic concept.</span></div>'
medical+='<p>来源版本5623abe8a843deaea7ab838a8c76c41afbad565e，作业28792059；分类器SHA07a6f65d…655。原图只有224×224有效分辨率，导出放大不增加医学细节。下列均为模型实际使用的有效mask，青色表示选中，灰色表示遮蔽，不是病灶真值。</p><p class="en">Job28792059, commit5623abe8a843deaea7ab838a8c76c41afbad565e; classifier SHA07a6f65d…655. Native inputs contain224×224 pixels; enlarged exports add no medical detail. Teal marks the selected effective region; grey marks hidden pixels. These are not lesion ground-truth masks.</p>'
medical+='<h2>完整概念索引 / Complete concept index</h2><p><a href="#HUMCD-MEL-C001">HUMCD-MEL-C001（唯一学习概念 / sole learned concept）</a> · <a href="#HUMCD-MEL-COMPLEMENT">HUMCD-MEL-COMPLEMENT（补空间，非学习概念 / complement）</a></p><p>全部学习概念和top-K视图在本轮相同（K≥1），无需重新训练。审阅状态均为未审阅，无AI医学命名。</p><p class="en">All learned concepts and top-K are identical here forK≥1. Review status remains unreviewed; no AI medical labels are asserted.</p>'
for ci,cid in enumerate(['HUMCD-MEL-C001','HUMCD-MEL-COMPLEMENT']):
    medical+='<h2 id="'+cid+'">'+cid+'</h2>'
    for key,label in [('top_prototypes','原规则top prototypes / Released top prototypes'),('random_final_members','固定随机成员 / Fixed random members')]:
        medical+='<h3>'+label+'</h3><p>每侧最多10例；top按已分配成员的局部分数降序、每图最多一个；随机为独立RandomState4301按概念顺序无放回。两者不代表临床语义。</p><p class="en">Up to10 per split. Top: descending local assignment score, one region per image. Random: independent RandomState4301, without replacement in concept order. Neither establishes clinical semantics.</p><div class="twocol">'
        for role,label2 in [('training','训练 / Training'),('held_out','留出 / Held-out')]:
            medical+='<section><h4>'+label2+'</h4>'+''.join(triptych(role,sid,key) for sid in examples[role][ci][key])+'</section>'
        medical+='</div>'
    medical+='<h3>低分配间隔诊断 / Low-margin diagnostic examples</h3><p>每侧属于该分配的最低top1−top2间隔3例；诊断选择，不是总体代表。 / Lowest3 top1−top2 margins among assigned members per split; diagnostic, not representative.</p><div class="twocol">'
    for role in ['training','held_out']:
        s=sci[role];rr=list(assign[role]);members=np.flatnonzero(s['assignments']==ci);margin=np.diff(np.sort(s['concept_activations'],axis=1),axis=1)[:,-1];sel=members[np.argsort(margin[members],kind='stable')[:3]]
        medical+='<section><h4>'+role+'</h4>'+''.join(triptych(role,rr[i],'diagnostic_low_margin3') for i in sel)+'</section>'
    medical+='</div><form data-review="'+cid+'"><h3>探索性人工审阅 / Exploratory manual review</h3><p>未审阅。此表不替代论文人类实验或专业诊断。 / Unreviewed; not a validated human study or clinical assessment.</p>'
    for key,cn,en in [('description','能否给出稳定简短描述','Stable short description'),('content','主体、部件、背景或混合','Subject, part, background or mixed'),('random','随机成员是否支持描述','Do random members support the description?'),('transfer','训练与留出是否一致','Training–held-out consistency'),('unrelated','是否有不相关成员','Unrelated members'),('evidence','依据及示例ID','Evidence and example IDs')]:medical+='<label>'+cn+' / '+en+'<textarea name="'+key+'" aria-label="'+cn+'"></textarea></label>'
    medical+='</form>'
medical+='<h2>初始cluster独立视图 / Initial clusters, separate from final assignment</h2><p>聚类输入为非零特征，L2归一化，q1不去除有限行；PCA成员映射已保存。展示各cluster固定随机8成员（独立RandomState4302），包含被过滤cluster，不把初始成员数当作最终分配数。</p><p class="en">Clustering uses nonzero L2-normalized features, no finite-row removal atq1. PCA member mappings are preserved. Eight fixed random members per cluster (independent RandomState4302), including filtered clusters; initial membership is distinct from final assignment.</p>'
with np.load(d1/'initial_clusters.npz') as z:labels=z['labels'];segids=z['segment_id']
rng=np.random.RandomState(4302)
for c in range(3):
    ix=np.flatnonzero(labels==c);selected=rng.choice(ix,min(8,len(ix)),replace=False)
    medical+='<h3>Initial cluster '+str(c)+' · '+str(len(ix))+' members · '+('retained / 保留' if c==0 else 'filtered / 过滤')+'</h3><div class="twocol">'+''.join(triptych('training',str(segids[i]),'initial_cluster_random8_seed4302') for i in selected)+'</div>'
medical+='<button id="export">导出审阅 / Export review</button><script>const key="MED26-discovery-review-28792059-v1";const forms=[...document.querySelectorAll("form[data-review]")];let saved=JSON.parse(localStorage.getItem(key)||"{}");for(const f of forms){for(const e of f.elements)e.value=saved[f.dataset.review]?.[e.name]||"";f.addEventListener("input",()=>{saved[f.dataset.review]=Object.fromEntries(new FormData(f));localStorage.setItem(key,JSON.stringify(saved));});}document.querySelector("#export").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify({scope:"Exploratory nonclinical concept review; user-entered records only",exported_at:new Date().toISOString(),records:saved},null,2)],{type:"application/json"}));a.download="medical-concept-review.json";a.click();URL.revokeObjectURL(a.href);};</script>'
(B/'medical.html').write_text(page('首次医学发现：一个学习子空间 / First medical discovery: one learned subspace',medical))
table(B/'display-selection.csv',selection)
table(B/'review-template.csv',[dict(concept_id=cid,status='UNREVIEWED',description='',content_type='',random_support='',training_heldout_consistency='',unrelated_members='',evidence_ids='',reviewer='',actual_review_date='',entry_date='') for cid in ['HUMCD-MEL-C001','HUMCD-MEL-COMPLEMENT']])
rob='<p>这项实验衡量固定概念对最终输入mask边界扰动的敏感性，不是SAM分割准确率、重新发现稳定性或像素真值评价。十类原batch不变对照全部通过。侵蚀半径1已完整验收，图片级保留率先类内平均再等权类平均为78.25%。</p><p class="en">This evaluates sensitivity to final input-mask boundary perturbations with concepts fixed, not SAM segmentation accuracy, refitting stability or pixel ground truth. All ten original-batch identity controls pass. Radius1 erosion is complete: equal-class mean of image-level retention is78.25%.</p>'
rob+='<div class="notice">六类完成五条件；四类在半径2侵蚀的7行缩小batch对照停止。仅28个对照区域观察到分配不变，不代表未运行区域通过。保留原容差，无自动重提。半径2和扩张只覆盖六类，不能称十类结论。<br><span class="en">Six classes completed all five conditions. Four stopped at seven-row reduced-batch controls during radius2 erosion. No assignments changed in those28 control rows; this does not validate unexecuted regions. Original tolerances remain. Radius2/dilation currently cover six classes, not ten.</span></div>'
rob+='<figure><img src="figures/robustness.png" alt="Class-wise retention, missing conditions grey"><figcaption><a href="figures/robustness.pdf">高分辨率PDF / PDF</a> · <a href="robustness-conditions.csv">分母及95%图片bootstrap区间 / Denominators and95% image-bootstrap intervals</a></figcaption></figure>'+render_table(A['statuses'],['class_name','job','state','completed_conditions','acceptance'])
rob+='<h2>失败证据 / Failure evidence</h2>'+render_table(A['reduced_batch_controls'],['class_name','rows','max_abs_feature','max_relative_feature','max_abs_score','changed_assignments','feature_elementwise_gate'])
rob+='<p>直接事实：8→7行的不变特征偏差超过1e−4检查门槛，所有失败均在同一新增缩批路径；原batch特征逐元素一致。可能原因是batch形状触发GPU数值路径变化，尚未用GPU对照验证具体算子或TF32因果。不能称为已证实源码数学错误，也不放宽阈值。建议后续仅修正保持batch槽位的执行方式并验收；需单独批准重提失败的四类。</p><p class="en">Observed:8→7 row controls exceed the1e−4 gate while original-batch features match exactly; all failures share the new reduced-batch path. GPU arithmetic changes induced by batch shape are a plausible explanation, not an established operator/TF32 cause. No tolerance relaxation or confirmed upstream mathematical-bug claim. A targeted fixed-slot execution correction and four-class rerun require separate approval.</p>'
rob+='<p>主分母是基线有效区域；扰动后无效仍保留在分母。无基线有效区域的图为不可评价。每图先平均，类内配对bootstrap2000次、seed20260921；同时提供区域加权与两侧有效条件指标，不能混用。/ Primary denominators retain invalidated regions. Images lacking valid baseline regions are unevaluable. Image-level paired bootstrap uses2000 draws, seed20260921. Region-weighted and both-valid conditional measures are separate.</p>'
rob+='<p><a href="robustness-per-image.csv">Per-image CSV</a> · <a href="robustness-strata.csv">Frozen area/margin strata</a> · <a href="robustness-transitions.csv">Concept/complement/invalid transitions</a></p>'
(B/'robustness.html').write_text(page('掩码敏感性：十类轻度侵蚀可分析，四类后续条件缺失 / Mask sensitivity: partial completion',rob))
summary='<div class="good"><b>已完成验收 / Accepted evidence</b><p>十类不变对照及半径1侵蚀；六类全部五条件；医学400/70发现。仅有一个医学学习概念不构成执行失败，但解释粒度有限。</p><p class="en">Ten-class identity/radius1 erosion; six complete five-condition classes; medical400/70 discovery. One learned medical concept is not an execution failure, but limits explanatory granularity.</p></div><p><a href="medical.html"><b>医学概念：训练与留出并排，原型／随机／初始cluster / Medical concept gallery</b></a></p><p><a href="robustness.html"><b>敏感性结果与四类失败诊断 / Sensitivity and failed-control diagnosis</b></a></p><figure><img src="figures/medical-summary.png" alt="Medical discovery overview"></figure><figure><img src="figures/robustness.png" alt="Robustness overview"></figure>'
summary+='<h2>结论边界 / Interpretation limits</h2><p>重构与映射通过仅说明数值与工程一致。概念名称、原型与随机成员的语义一致性仍待人工审阅；不能把completeness当可理解性。外部E224/R101/S协议和预算已批准，执行状态见主报告当前批次入口；此页没有外部结果曲线或虚构数值。</p><p class="en">Reconstruction and mapping establish numerical/engineering consistency. Semantic naming and prototype–random coherence remain for manual review; completeness is not understandability. E224/R101/S and its budget are approved; consult the main report for submission status. No external outcome is fabricated here.</p>'
summary+='<p>旧H重叠检查：R101中20例有214个相似候选，尚非确认重复；原始9例审阅不等于跨数据集审阅。病例／图片不能替代患者身份。固定模型有DF零召回和后期过拟合迹象，沿用已披露限制，不重训。</p><p class="en">Prior H found214 perceptual candidates involving20 R101 cases, not confirmed duplicates. Reviewing the original nine aliases did not review cross-dataset overlaps. Cases/images are not patient IDs. Previously disclosed classifier limits (zero DF recall and later overfitting pattern) remain; no retraining.</p>'
summary+='<h2>来源与可复查文件 / Provenance and inspectable files</h2><p>A执行版本40418edfa4e4ac81eda02d954fe59de52a0f14f0；医学D1版本5623abe8a843deaea7ab838a8c76c41afbad565e。生成仅使用本地已收集缓存，无新预测、拟合、选样或远端文件处理。/ Generated only from collected local caches, without new predictions, fitting or selection.</p><p>Generated UTC '+esc(datetime.now(timezone.utc).isoformat())+'</p><ul>'
for f in sorted(B.glob('*.csv')):summary+='<li><a href="'+f.name+'">'+f.name+'</a></li>'
summary+='</ul><p><a href="acceptance.json">完整验收凭据 / Acceptance</a> · <a href="final-sacct.txt">最终Slurm状态 / Final Slurm accounting</a></p>'
(B/'index.html').write_text(page('两条工作线验收与概念审阅 / Two-workstream acceptance and concept review',summary))
# High-resolution slide panel: original cached top3/random3, two splits; no cherry picking.
fig,axes=plt.subplots(4,3,figsize=(14,8))
for row,(role,key) in enumerate([('training','top_prototypes'),('held_out','top_prototypes'),('training','random_final_members'),('held_out','random_final_members')]):
    for col,sid in enumerate(examples[role][0][key][:3]):
        r=rawrows[role][sid];rgb=images[r['image_id']];mask=masks[r['image_id']][r['raw_index']];shown=rgb.copy();shown[~mask]=220
        axes[row,col].imshow(shown);axes[row,col].axis('off');axes[row,col].set_title(role+' / '+('top' if key.startswith('top') else 'random')+'\n'+r['image_id']+f" S{r['raw_index']:04d}",fontsize=10)
fig.suptitle('HUMCD-MEL-C001 | selected effective region; grey = hidden\nFirst3 in saved top/random order, native inputs224 x224; not clinical annotations',fontsize=13)
fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(B/'figures/medical-concept-comparison.png',dpi=300);fig.savefig(B/'figures/medical-concept-comparison.pdf');plt.close(fig)
files={str(f.relative_to(B)):sha(f) for f in B.rglob('*') if f.is_file() and f.name!='manifest.json'}
dump(B/'manifest.json',dict(id='batch-review-20260922',dataset_id='medical-and-robustness-batch-20260922',entry='index.html',created_at=datetime.now(timezone.utc).isoformat(),files=files))
print(json.dumps(dict(status='BUILT_PRIVATE',files=len(files),example_selections=len(selection),images=len(images),manifest=str(B/'manifest.json'))))
