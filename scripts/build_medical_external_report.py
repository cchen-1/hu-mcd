"""Private bilingual E224 review bundle from accepted saved outputs only."""
import csv, hashlib, html, io, json, shutil, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/bunya/medical-external-results-20260923';P=OUT/'collection';B=OUT/'bundle'
J=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def table(p,rr):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def esc(x):return html.escape(str(x))
def text(zh,en):
    for old,new in [('All101','All 101'),('for29','for 29'),('versus50','versus 50'),('beforeSAM','before SAM'),('so29','so 29'),('Of296','Of 296'),('Of259','Of 259'),('and165','and 165'),('in89','in 89'),('with57','with 57'),('to101','to 101'),('has32','has 32'),('dimensions,807','dimensions, 807'),('cluster0','cluster 0'),('importance0.','importance 0.'),('rank1/1','rank 1/1'),('The2016','The 2016'),('independentRandomState4303','independent RandomState 4303'),('up to10','up to 10'),('lowest3','lowest 3')]:en=en.replace(old,new)
    return '<p>'+zh+'</p><p class="en">'+en+'</p>'
def grid(rr,keys):
    def fmt(v):return '不适用 / N/A' if v is None else (f'{v:.6g}' if isinstance(v,float) else v)
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(k)+'</th>' for k in keys)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(fmt(r.get(k)))+'</td>' for k in keys)+'</tr>' for r in rr)+'</tbody></table></div>'
STYLE='''body{font:17px/1.6 system-ui,sans-serif;color:#193240;background:#f6f9fa;max-width:1240px;margin:auto;padding:28px}h1{font-size:32px;line-height:1.35}h2{margin-top:40px;border-bottom:2px solid #147e83}h3{margin-top:30px}.en{color:#4b6470}.note{padding:18px;border-left:5px solid #b77321;background:#fff3dd}a{color:#075d88}nav{display:flex;flex-wrap:wrap;gap:18px}figure{background:white;border:1px solid #d0dfe3;margin:18px 0;padding:14px}img{max-width:100%;height:auto}figcaption{font-size:14px;overflow-wrap:anywhere}table{border-collapse:collapse;background:#fff;font-size:14px}th,td{padding:8px 10px;border:1px solid #d0dfe3;text-align:left}th{background:#e4f0f2}.scroll{overflow:auto}.native{max-height:650px;object-fit:contain}textarea{font:inherit;width:94%;min-height:75px}label{display:block;margin:12px 0}button{font:inherit;padding:8px 16px} @media(max-width:700px){body{padding:14px}h1{font-size:26px}}'''
def page(title,body,prefix=''):
    return '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+esc(title)+'</title><style>'+STYLE+'</style><body data-dataset="medical-external-R101-E224-20260923"><script id="concept-ids" type="application/json">[]</script><nav><a href="'+prefix+'index.html">外部结果 / External results</a><a href="'+prefix+'concepts.html">概念与审阅 / Concepts and review</a><a href="'+prefix+'cases.html">全部101例 / All101 cases</a></nav><h1>'+esc(title)+'</h1>'+body+'</body></html>'

def main():
    global B
    final_bundle=OUT/'bundle'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    B=OUT/('bundle-staging-'+stamp)
    acceptance=J(OUT/'acceptance.json');assert acceptance['status']=='ENGINEERING_ACCEPTED_RESEARCH_LIMITS_DISCLOSED'
    for d in ['figures','examples','images','originals','cases']:(B/d).mkdir(parents=True,exist_ok=True)
    for f in OUT.glob('*.csv'):shutil.copy2(f,B/f.name)
    for f in ['acceptance.json','sacct.psv','collection.json']:
        shutil.copy2(OUT/f,B/f)
    for f in ['anomalies.jsonl','actual_config.json','input_binding.json','full_checks.json','segment_checks.json','actual_input_mapping.json','concept_index.json']:
        shutil.copy2(P/f,B/f)
    if (OUT/'issue-updates.json').exists():shutil.copy2(OUT/'issue-updates.json',B/'issue-updates.json')
    if (ROOT/'docs/MEDICAL_EXTERNAL_RESULTS_20260923.zh-en.md').exists():shutil.copy2(ROOT/'docs/MEDICAL_EXTERNAL_RESULTS_20260923.zh-en.md',B/'results.zh-en.md')
    for script in ['accept_medical_external_results.py','build_medical_external_report.py','collect_medical_external_results.py']:shutil.copy2(ROOT/'scripts'/script,B/script)
    rows=J(P/'actual_input_mapping.json');mapping=J(P/'segment_mapping.json');perimage=list(csv.DictReader((OUT/'per-image.csv').open()));byimage={r['image_id']:r for r in perimage}
    s=np.load(P/'segments.npz');full=np.load(P/'full_images.npz');cov=np.load(P/'coverage_matrix.npz');contrasts=J(P/'descriptive_contrasts.json');ids=['HUMCD-MEL-C001','HUMCD-MEL-COMPLEMENT'];names=['AKIEC','BCC','BKL','DF','MEL','NV','VASC']
    images={};masks={};originals={};image_meta={r['image_id']:r for r in rows}
    with zipfile.ZipFile('/mnt/c/Users/uqcche38/Downloads/release_v0.zip') as archive:
        for r in rows:
            iid=r['image_id'];images[iid]=np.array(Image.open(P/'inputs'/(iid+'.png')));masks[iid]=np.load(P/'cache'/(iid+'_effective_masks.npz'))['masks']
            shutil.copy2(P/'inputs'/(iid+'.png'),B/'images'/(iid+'.png'))
            relative=r['inventory']['relative_path'];name=iid+Path(relative).suffix;originals[iid]=name;(B/'originals'/name).write_bytes(archive.read('release_v0/images/'+relative))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
    evidence=[]
    for i,r in enumerate(mapping):
        iid=r['image_id'];rgb=images[iid];mask=masks[iid][r['raw_index']];overlay=rgb.copy();overlay[mask]=(.62*rgb[mask]+.38*np.array([0,180,175])).astype(np.uint8);selected=rgb.copy();selected[~mask]=220
        canvas=Image.new('RGB',(1416,520),'white');draw=ImageDraw.Draw(canvas)
        for j,(pix,label) in enumerate([(rgb,'E224 model input'),(overlay,'Teal = selected region'),(selected,'Grey = hidden region')]):
            canvas.paste(Image.fromarray(pix).resize((448,448),Image.Resampling.NEAREST),(12+j*472,48));draw.text((12+j*472,12),label,font=font,fill='#193240')
        f=B/'examples'/(r['segment_id']+'.png');canvas.save(f)
        evidence.append(dict(segment_id=r['segment_id'],image_id=iid,raw_index=r['raw_index'],global_row=i,concept_id=r['concept_id'],valid=not r['zero_feature'],effective_mask_sha256=r['effective_mask_sha256'],effective_pixels=r['effective_pixels'],input_sha256=image_meta[iid]['input_sha256'],original_sha256=image_meta[iid]['inventory']['sha256'],panel_sha256=sha(f),panel=str(f.relative_to(B))))
    table(B/'example-evidence.csv',evidence)
    def panel(i,prefix=''):
        r=mapping[i];iid=r['image_id'];state=r['concept_id'] or 'ZERO FEATURE / 未有效分配';score=float(s['scores'][i,s['assignments'][i]]) if s['valid'][i] else None
        return '<figure><a href="'+prefix+'examples/'+r['segment_id']+'.png"><img loading="lazy" width="1416" height="520" src="'+prefix+'examples/'+r['segment_id']+'.png" alt="E224 input, teal selected region, selected-only region with grey hidden pixels"></a><figcaption>'+esc(r['segment_id'])+'<br>'+esc(state)+' · selected area '+f'{r["effective_pixels"]/50176:.1%}'+' · local score '+('N/A' if score is None else f'{score:.5f}')+' · <a href="'+prefix+'cases/'+iid+'.html">原图、标签与全部片段 / Original, labels and all regions</a></figcaption></figure>'
    # Display-only RNG is independent of training and segmentation RNGs.
    rng=np.random.RandomState(4303);selections=[]
    for ci,cid in enumerate(ids):
        members=np.flatnonzero(s['assignments']==ci);ordered=members[np.argsort(s['scores'][members,ci])[::-1]];top=[];seen=set()
        for i in ordered:
            iid=mapping[i]['image_id']
            if iid not in seen:top.append(int(i));seen.add(iid)
            if len(top)==10:break
        random_members=rng.choice(members,min(10,len(members)),replace=False).tolist();margins=abs(np.diff(s['scores'][members],axis=1))[:,0];low=members[np.argsort(margins,kind='stable')[:3]].tolist()
        for kind,chosen in [('released_top10_one_per_image',top),('random10_seed4303',random_members),('diagnostic_low_margin3',low)]:
            for rank,i in enumerate(chosen,1):selections.append(dict(concept_id=cid,selection=kind,rank=rank,global_row=i,segment_id=mapping[i]['segment_id'],image_id=mapping[i]['image_id']))
    for i in np.flatnonzero(~s['valid'])[:3]:selections.append(dict(concept_id='INVALID',selection='diagnostic_first3_zero_in_frozen_row_order',rank=sum(r['concept_id']=='INVALID' for r in selections)+1,global_row=int(i),segment_id=mapping[i]['segment_id'],image_id=mapping[i]['image_id']))
    table(B/'selection-manifest.csv',selections)
    for r in rows:
        iid=r['image_id'];i=r['order'];region_indices=[j for j,m in enumerate(mapping) if m['image_id']==iid];pr=byimage[iid]
        body=text('原始图片保持原比例；下面的224×224是获批的模型工作输入，已改变纵横比。没有病灶或征象真值mask。','Original aspect ratio is preserved below; the approved224×224 model input is warped. No lesion/sign ground-truth masks are available.')
        body+='<figure><a href="../originals/'+originals[iid]+'"><img class="native" src="../originals/'+originals[iid]+'" alt="Original Derm7pt image at original aspect ratio"></a><figcaption>'+esc(r['inventory']['relative_path'])+f" · {r['inventory']['width']}×{r['inventory']['height']}"+'</figcaption></figure>'
        body+=text('病例编号不是患者编号。预测='+names[int(pr['prediction'])]+'；本页未进行专业医学审阅。','Case identity is not patient identity. Prediction='+names[int(pr['prediction'])]+'; this page is not professionally reviewed.')
        body+=grid([r['record']],list(r['record']))+grid([pr],['image_id','raw_regions','valid_regions','zero_features','learned_assignments','complement_assignments','C001_coverage','complement_coverage'])
        body+=grid([dict(class_name=names[j],logit=float(full['logits'][i,j])) for j in range(7)],['class_name','logit'])
        body+=text('所有原始片段按原数组顺序展示。零特征仍保留，不强行归入C001；显示放大不增加原始细节。','All raw regions appear in original array order. Exact-zero features remain invalid. Display enlargement adds no input detail.')+''.join(panel(j,'../') for j in region_indices)
        (B/'cases'/(iid+'.html')).write_text(page(iid,body,'../'))
    listing=text('101/101例保留，未按预测、覆盖率或视觉效果筛选。点击病例查看原图、原标签、七类logits与全部片段。','All101 cases are retained without filtering by predictions, coverage or appearance. Each case links to original imagery/labels, seven logits and every region.')
    listing+='<div class="scroll"><table><tr><th>Case</th><th>Prediction</th><th>Raw / valid / zero regions</th><th>C001 coverage</th><th>Complement coverage</th></tr>'
    for r in perimage:listing+='<tr><td><a href="cases/'+r['image_id']+'.html">'+r['image_id']+'</a></td><td>'+names[int(r['prediction'])]+'</td><td>'+r['raw_regions']+' / '+r['valid_regions']+' / '+r['zero_features']+'</td><td>'+f"{float(r['C001_coverage']):.1%}"+'</td><td>'+f"{float(r['complement_coverage']):.1%}"+'</td></tr>'
    (B/'cases.html').write_text(page('全部外部病例 / All external cases',listing+'</table></div>'))
    concept=text('同一个冻结的HUMCD-MEL-C001延伸到外部101例；没有在Derm7pt上聚类或重新拟合。32维，初始源域cluster0共807成员，全局重要性0.439451，排名1/1。补空间2016维，独立展示。','The same frozenHUMCD-MEL-C001 is applied to101 external cases; no external clustering/refit. It has32 dimensions,807 initial source cluster0 members, global importance0.439451, rank1/1. The2016-dimensional complement is separate.')
    concept+=text('原型按已分配成员的局部分数降序、每图最多一个；随机最多10个成员，独立RandomState4303按概念顺序无放回；低间隔诊断最多3例。选择清单可追溯，均不代表总体。','Top prototypes use descending assigned-component scores, one per image. Random uses independentRandomState4303, up to10 members without replacement in concept order. Diagnostics use the lowest3 margins. All are traceable selections, not population summaries.')
    concept+=text('全局重要性＝MEL分类权重在该子空间分量的平方范数／权重平方范数；局部分配分数＝该片段子空间分量范数／片段特征范数。外部没有重新排序或拟合全局重要性。表中小数为显示舍入，完整精度保存在CSV。','Global importance is squared norm of the MEL classifier-weight component divided by squared weight norm. Local assignment score is component norm divided by region-feature norm. Global importance is not refitted or reranked externally. Table decimals are rounded for display; CSV preserves full precision.')
    concept+='<p><a href="selection-manifest.csv">选择清单 / Selection manifest</a> · <a href="concept_index.json">原始cluster/PCA成员索引 / Initial cluster and PCA membership</a></p>'
    for cid in ids:
        concept+='<h2 id="'+cid+'">'+cid+'</h2>'+grid([next(r for r in acceptance['concept_summary'] if r['concept_id']==cid)],list(acceptance['concept_summary'][0]))
        for kind,label in [('released_top10_one_per_image','原规则原型 / Released top prototypes'),('random10_seed4303','固定随机成员 / Fixed random members'),('diagnostic_low_margin3','低分配间隔诊断 / Low-margin diagnostics')]:
            concept+='<h3>'+label+'</h3>'+''.join(panel(r['global_row']) for r in selections if r['concept_id']==cid and r['selection']==kind)
        concept+='<form data-review="'+cid+'"><h3>未审阅 / Unreviewed</h3>'+text('这是外部迁移的探索性审阅补充，不覆盖已保存的源域审阅，不是临床研究。','This exploratory external-review supplement preserves existing source-domain reviews; it is not a clinical study.')
        for key,label in [('description','稳定简短描述 / Stable short description'),('content','主体、部件、背景或混合 / Subject, part, background or mixed'),('random','随机成员是否支持 / Support from random members'),('transfer','源域与外部是否一致 / Source–external consistency'),('unrelated','明显不相关成员 / Unrelated members'),('evidence','依据和示例ID / Evidence and example IDs')]:concept+='<label>'+label+'<textarea name="'+key+'"></textarea></label>'
        concept+='</form>'
    concept+='<h2>零特征诊断 / Zero-feature diagnostics</h2>'+text('仅展示原数组顺序的前3个零特征片段；不是代表性抽样或因果解释。','First3 exact-zero regions in frozen row order; diagnostic selection, not representative or causal.')+''.join(panel(r['global_row']) for r in selections if r['concept_id']=='INVALID')
    concept+='<button id="export">导出外部审阅 / Export external review</button><script>const key="MED26-E224-review-28796209-v1";let records=JSON.parse(localStorage.getItem(key)||"{}");document.querySelectorAll("form[data-review]").forEach(f=>{const id=f.dataset.review;f.querySelectorAll("textarea").forEach(t=>{t.value=(records[id]||{})[t.name]||"";t.addEventListener("input",()=>{records[id]=records[id]||{};records[id][t.name]=t.value;localStorage.setItem(key,JSON.stringify(records))})})});document.getElementById("export").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify({job:"28796209",scope:"Exploratory external review; not clinical validation",records},null,2)],{type:"application/json"}));a.download="E224-review.json";a.click();URL.revokeObjectURL(a.href)};</script>'
    (B/'concepts.html').write_text(page('固定概念的外部成员 / External members of frozen concepts',concept))
    table(B/'review-template.csv',[dict(concept_id=cid,job='28796209',review_status='UNREVIEWED',description='',content='',random_support='',source_external_consistency='',unrelated_members='',evidence_ids='',reviewer='',review_date='') for cid in ids])
    # Fixed-order charts, no significance, smoothing, missing endpoints or new metrics.
    fig,axes=plt.subplots(1,3,figsize=(15,4.8));labels=['Source train\n400 images','Source held-out\n70 images /61 lesions','External R101\n101 cases'];vals=[341/400,50/70,29/101]
    axes[0].bar(labels,vals,color=['#7ebdb5','#438f91','#bd6f39']);axes[0].set_ylim(0,1);axes[0].set_title('Full-image MEL prediction fraction\nDifferent cohorts; not a paired effect')
    for i,(v,n) in enumerate(zip(vals,['341/400','50/70','29/101'])):axes[0].text(i,v+.025,n,ha='center')
    v=np.array([514/849,82/143,94/259]);axes[1].bar(labels,v,color='#167e83',label='Learned C001');axes[1].bar(labels,1-v,bottom=v,color='#bdc7d0',label='Complement');axes[1].set_title('Final assignment among valid regions');axes[1].set_ylim(0,1);axes[1].legend(fontsize=8)
    axes[2].bar(names,[int(acceptance['predictions'][str(i)]) for i in range(7)],color='#638d9b');axes[2].set_title('External predictions: all101 truth=MEL');axes[2].set_ylabel('Images');axes[2].tick_params(axis='x',labelsize=8)
    for i in range(7):axes[2].text(i,int(acceptance['predictions'][str(i)])+.7,acceptance['predictions'][str(i)],ha='center')
    fig.tight_layout();fig.savefig(B/'figures/transfer-summary.png',dpi=260);fig.savefig(B/'figures/transfer-summary.pdf');plt.close(fig)
    learned=[c for c in contrasts if not c['is_complement']];comp=[c for c in contrasts if c['is_complement']];fig,ax=plt.subplots(figsize=(11,7));y=np.arange(12)
    ax.scatter([c['auc'] for c in learned],y-.12,color='#137f84',label='C001');ax.scatter([c['auc'] for c in comp],y+.12,marker='s',color='#a17445',label='Complement (not learned)');ax.axvline(.5,color='#888',linestyle='--');ax.set_xlim(0,1);ax.set_yticks(y,[c['sign']+' / '+c['group']+'  n='+str(c['group_n'])+' vs '+str(c['reference_n']) for c in learned]);ax.invert_yaxis();ax.set_xlabel('AUC: higher union coverage predicts the named group (never reversed)');ax.set_title('All12 prespecified sign contrasts\nDescriptive association only; no p values / uncertainty intervals');ax.legend(loc='lower right');fig.tight_layout();fig.savefig(B/'figures/sign-contrasts.png',dpi=260);fig.savefig(B/'figures/sign-contrasts.pdf');plt.close(fig)
    # Source/held-out/external selected-only comparison, fixed first2 top + first2 random.
    d1=ROOT/'artifacts/bunya/batch-review-20260922/collections/28792059/20260922T063256.936418Z';oldbundle=ROOT/'artifacts/bunya/batch-review-20260922/bundle';fig,axes=plt.subplots(3,4,figsize=(13,10))
    comparison=[]
    for row,role in enumerate(['training','held_out','external']):
        if role=='external':chosen=[r for key in ['released_top10_one_per_image','random10_seed4303'] for r in selections if r['concept_id']==ids[0] and r['selection']==key and r['rank']<=2];rr=[mapping[r['global_row']] for r in chosen]
        else:
            ex=J(d1/(role+'_example_index.json'))[0];raw={r['segment_id']:r for r in J(d1/(role+'_raw_segments.json'))};rr=[raw[sid] for key in ['top_prototypes','random_final_members'] for sid in ex[key][:2]]
        for col,r in enumerate(rr):
            if role=='external':rgb=images[r['image_id']].copy();m=masks[r['image_id']][r['raw_index']]
            else:rgb=np.array(Image.open(oldbundle/'images'/(r['image_id']+'.png')));m=np.load(d1/'cache'/role/(r['image_id']+'_effective_masks.npz'))['masks'][r['raw_index']]
            rgb[~m]=220;axes[row,col].imshow(rgb);axes[row,col].axis('off');axes[row,col].set_title(role+' / '+('top' if col<2 else 'random')+'\n'+r['image_id']+f" S{r['raw_index']:04d}",fontsize=9)
            comparison.append(dict(role=role,view='top' if col<2 else 'random',segment_id=r['segment_id'],image_id=r['image_id']))
    fig.suptitle('Same frozen HUMCD-MEL-C001 | first2 top + first2 fixed random per domain\nSelected effective region; grey=hidden. Enlargement adds no detail. Not clinical labels.',fontsize=13);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(B/'figures/concept-transfer-comparison.png',dpi=300);fig.savefig(B/'figures/concept-transfer-comparison.pdf');plt.close(fig);table(B/'comparison-selection.csv',comparison)
    summary=text('101例全部完成；工程完整性通过，但固定分类器在外部只将29/101例预测为MEL（28.71%）。源域留出50/70（71.43%）来自不同队列，不能作为配对差值或因果迁移损失。','All101 cases completed and engineering integrity passed. The frozen classifier predicts MEL for29/101 external cases (28.71%), versus50/70 source held-out (71.43%). These are different cohorts, not a paired effect or causal estimate of transfer loss.')
    summary+=text('全图预测在SAM之前、不使用片段mask时已得到，因此29/101不是下游分割或概念分配产生的结果；输入域、几何及固定分类器因素仍未分离。','Full-image predictions were computed beforeSAM without segment masks, so29/101 is not produced by downstream segmentation or concept assignment. Domain, geometry and fixed-classifier factors remain unseparated.')
    summary+=text('外部296个片段中37个为非空mask但零特征（12.50%）。有效259个中C001占94（36.29%），补空间165；没有整图无有效片段。C001出现在89/101例，平均覆盖57.10%，不等于理解了病灶。','Of296 external regions,37 have nonempty masks but exact-zero features (12.50%). Of259 valid regions,94 map to C001 (36.29%) and165 to complement; every image has a valid region. C001 appears in89/101 cases with57.10% mean coverage; this does not establish lesion understanding.')
    summary+='<div class="note">已有支持：固定输入/权重/概念基完整性、逐图分配与图片级描述性关联。尚无支持：专家认可语义、像素定位、临床效用、患者独立性或下降的单一原因。<br>Supported: frozen input/model/basis integrity, per-image assignments and descriptive image-level associations. Not established: expert semantics, localization, clinical utility, patient independence or a single cause for lower MEL prediction.</div>'
    summary+='<h2>分类器、分配与原型 / Classifier, assignment and prototypes</h2><figure><img src="figures/transfer-summary.png" alt="MEL predictions and concept assignment across different cohorts"><figcaption><a href="figures/transfer-summary.pdf">PDF</a> · <a href="per-image.csv">逐图CSV / Per-image CSV</a></figcaption></figure><figure><a href="figures/concept-transfer-comparison.png"><img src="figures/concept-transfer-comparison.png" alt="Selected-only first2 prototypes and random members in source, held-out and external domains"></a><figcaption><a href="figures/concept-transfer-comparison.pdf">高分辨率PDF / High-resolution PDF</a> · <a href="comparison-selection.csv">选例清单 / Selection manifest</a> · <a href="concepts.html">并排原图/overlay/选中区域完整展示 / Full triptych examples</a></figcaption></figure>'
    summary+=grid(acceptance['concept_summary'],list(acceptance['concept_summary'][0]))
    summary+=text('待确认的AI视觉观察：固定对照图中外部top前2例保留较大区域，random前2例保留较局部的中心深色区域。只有4个新增外部例子的非专业观察，不代表全部成员，也不确定任何医学征象。','Provisional AI visual observation: the first2 external top examples retain broad areas, while the first2 random examples retain more bounded central dark regions. This is nonclinical inspection of four external examples, not a population conclusion or identification of medical signs.')
    summary+='<h2>12项征象关联 / Twelve sign associations</h2>'+text('研究问题：固定C001的图片覆盖面积是否随已有征象分组不同而变化？它提供图片级关联证据，不把概念等同于征象，不评价空间定位。','Question: does the image area covered by frozenC001 differ across existing sign-label groups? This supplies image-level association evidence, not semantic equivalence or spatial localization.')
    summary+=text('覆盖率＝同一概念有效分配mask的并集像素数／50176；不同概念不作归一化。ABS＝征象不存在（不是数据缺失），TYP＝典型，ATP＝非典型，REG＝规则，IR＝不规则，PRS＝存在。12项共享同一个C001覆盖率，不是发现了12个概念。','Coverage is the union of valid assigned masks divided by50176pixels, without normalization across concepts. ABS=absent (not missing data), TYP=typical, ATP=atypical, REG=regular, IR=irregular, PRS=present. The12 contrasts share one C001 response; they are not12 discovered concepts.')
    summary+=text('不规则dots/globules对ABS为AUC0.6602（84对12）；这只是12项中的描述性结果，不是显著性发现。pigmentation REG与streaks REG各仅1例，不能作为稳定证据。负向AUC保留原方向；无p值或多重检验显著性声明。','Irregular dots/globules versusABS has AUC0.6602 (84 versus12), one of12 descriptive comparisons, not a significance finding. Regular pigmentation and regular streaks each have one case and provide unstable estimates. Negative-direction AUCs remain unchanged; no p values or multiple-testing significance claims.')
    summary+='<figure><img src="figures/sign-contrasts.png" alt="All twelve learned and complement directional coverage AUCs with actual group counts"><figcaption><a href="figures/sign-contrasts.pdf">PDF</a> · <a href="contrasts.csv">全部24项CSV / All24 rows</a></figcaption></figure>'
    summary+=grid([{**c,'auc':round(c['auc'],6),'mean_difference':round(c['mean_difference'],6)} for c in contrasts],['concept_id','sign','group','group_n','reference_n','missing_n','other_category_n','auc','mean_difference'])
    summary+='<h2>限制、异常与下一步 / Limits, anomalies and next steps</h2>'+text('101张原图均非正方形，E224改变纵横比；下降可能涉及输入域、图像几何、类别构成及固定分类器能力，当前无法区分原因。全部37个零特征已保留，数值重构通过不解释其机制。','All101 originals are nonsquare; E224 changes aspect ratio. Input domain, geometry, cohort composition and fixed-classifier limitations may contribute to lower predictions; these explanations are not causally separated. All37 zero features remain recorded; correct reconstruction does not explain their mechanism.')
    summary+=text('H中20个R101病例有214个相似候选，尚未确认为重复；不排除病例。病例ID不是患者ID。原9例基础视觉审阅不是本批或跨数据集重叠审阅。','H identified214 similarity candidates involving20 R101 cases; duplicates remain unconfirmed and no cases were excluded. Case identity is not patient identity. The earlier nine-case basic review is not review of this batch or cross-dataset overlap.')
    summary+=text('本轮没有新实验或重提。最有价值的下一步是先审阅固定随机成员与H相似候选。若随后批准比较，优先隔离输入几何的影响；方案及资源需另行冻结，不优化现有classifier。A四类失败仍未补跑，原报告保留具体缺失。','No new experiment or resubmission occurred this turn. The next useful work is review of fixed random members and H similarity candidates. If a comparison is later authorized, isolating input geometry is a priority, with a separately frozen protocol/budget and no classifier optimization. Four A-class failures remain unrerun with explicit missing conditions.')
    summary+='<h2>来源与验收 / Provenance and acceptance</h2>'+text('部署28796207：COMPLETED0:0，6秒，申请1CPU、调度实际分配2CPU；未增加申请。外部28796209：COMPLETED0:0，3分16秒，8CPU/1L40S，峰值RSS6044.30MiB，显存allocated5.70GiB/reserved6.22GiB，产物98.52MiB。','Deployment28796207: COMPLETED0:0,6s, requested1CPU and Slurm allocated2CPU; request was not enlarged. External28796209: COMPLETED0:0,3m16s,8CPU/1L40S, peakRSS6044.30MiB, GPUallocated5.70GiB/reserved6.22GiB, artifacts98.52MiB.')
    summary+=text('完整执行提交：'+acceptance['commit']+'；采样、权重及精度绑定见实际配置。验收全部101输入、296掩码/特征/分配及24统计；未重新预测。','Full execution commit: '+acceptance['commit']+'; identities and precision are in the resolved configuration. Acceptance covers101 inputs,296 masks/features/assignments and24 statistics, without new predictions.')
    summary+='<ul>'+''.join('<li><a href="'+n+'">'+n+'</a></li>' for n in ['acceptance.json','actual_config.json','actual_input_mapping.json','input_binding.json','sacct.psv','anomalies.jsonl','issue-updates.json','results.zh-en.md','example-evidence.csv','review-template.csv','accept_medical_external_results.py','build_medical_external_report.py'])+'</ul><p>Generated UTC '+datetime.now(timezone.utc).isoformat()+' · Private/offline, no external assets.</p>'
    (B/'index.html').write_text(page('外部运行完整，但医学解释仍需审阅 / Complete external execution; medical interpretation remains open',summary))
    files={str(f.relative_to(B)):sha(f) for f in B.rglob('*') if f.is_file() and f.name!='manifest.json'};dump(B/'manifest.json',dict(id='medical-external-results-20260923',dataset_id='medical-external-R101-E224-20260923',entry='index.html',created_at=datetime.now(timezone.utc).isoformat(),files=files))
    dump(OUT/'generation-provenance.json',dict(script_sha256=sha(__file__),acceptance_sha256=sha(OUT/'acceptance.json'),files=len(files),case_pages=101,panels=len(evidence),selection_rows=len(selections),random_display_seed=4303,training_changed=False,new_predictions=False))
    # Published media may be hardlinked: never overwrite those source inodes.
    if final_bundle.exists():final_bundle.rename(OUT/('bundle-backup-'+stamp))
    B.rename(final_bundle)
    print('Built',len(files),'private files;',len(evidence),'traceable panels;101case pages')

if __name__=='__main__':main()
