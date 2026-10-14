"""Incremental publication of accepted A-line completion; preserve other tracks."""
from pathlib import Path
from datetime import datetime, timezone
import csv,json,html,re,hashlib,os,shutil
R=Path(__file__).resolve().parents[2];O=R/'artifacts/bunya/robustness-completion-20260924'
def bi(z,e):return '<span data-l="zh">'+html.escape(z)+'</span><span data-l="en" hidden>'+html.escape(e)+'</span>'
def p(z,e):return '<p>'+bi(z,e)+'</p>'
def h(z,e):return '<h2>'+bi(z,e)+'</h2>'
def table(head,rows):return '<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+bi(*v)+'</th>' for v in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def atomic(src,dst):
    dst=Path(dst);tmp=dst.with_name(dst.name+'.new');shutil.copy2(src,tmp);os.replace(tmp,dst)
def apply(report):
    a=json.loads((O/'acceptance.json').read_text());assert a['status']=='ACCEPTED' and a['missing_conditions']==0
    report=Path(report);data=report/'hub/data';changed=[]
    names=[r['name'] for r in json.loads((R/'artifacts/bunya/ten-class-review/atlas_final.json').read_text())['classes']]
    rows=list(csv.DictReader((O/'robustness-conditions.csv').open()))
    from .robustness_figure import build
    import matplotlib.pyplot as plt
    fig=build(rows,names)
    for ext in ['png','svg','pdf']:
        temp=O/('robustness-accepted.'+ext);fig.savefig(temp,dpi=240,facecolor='white');dest=report/'hub/figures'/('robustness.'+ext);atomic(temp,dest);changed.append(dest)
    plt.close(fig)
    files=['merged-class-status.csv','acceptance.json','four-class-acceptance.json','robustness-conditions.csv','robustness-per-image.csv','robustness-transitions.csv','robustness-strata.csv','robustness-macro.csv','warnings-and-events.json','final-sacct.psv','batch.json','deployment-provenance.txt']
    for name in files:
        dst=data/('robustness-completion-20260924--'+name);atomic(O/name,dst);changed.append(dst)
    if (O/'acceptance-events.json').exists():
        dst=data/'robustness-completion-20260924--acceptance-events.json';atomic(O/'acceptance-events.json',dst);changed.append(dst)
    labels={'identity0':('原掩码','Identity'),'erosion1':('腐蚀1像素','Erode 1 pixel'),'erosion2':('腐蚀2像素','Erode 2 pixels'),'dilation1':('膨胀1像素','Dilate 1 pixel'),'dilation2':('膨胀2像素','Dilate 2 pixels')}
    body=p('十类、每类固定50张验证图、五个条件均已验收。四类补跑复用原掩码、特征基准和概念基；没有重新运行SAM、拟合或调参。','All five conditions are accepted for ten classes, each with 50 fixed validation images. Four completion jobs reused frozen masks, baseline features and concept bases; no SAM rerun, refitting or tuning.')
    body+=table([('条件','Condition'),('类别','Classes'),('平均保持率','Mean retention'),('空区域','Empty regions'),('零特征','Zero features'),('有效片段','Valid regions')],[[bi(*labels[m['condition']]),'10 / 10',f"{m['equal_class_mean']:.2%}",m['empty'],m['zero'],m['valid']] for m in a['macro']])
    body+=p('每个条件都有5687个原始片段；固定基准中5274个有效、413个零特征。先求每张图的有效基准片段保持率，再类内平均、十类等权平均。空区域或扰动后零特征仍留在基准分母中；无有效概念时不赋概念标签。','Every condition has 5,687 raw regions; the fixed baseline contains 5,274 valid and 413 zero-feature regions. Retention is computed per image over valid baseline regions, averaged within class, then equally over ten classes. Empty or newly zero-feature regions remain in that baseline denominator and receive no concept label.')
    body+='<figure><a href="hub/figures/robustness.png" target="_blank"><img src="hub/figures/robustness.png" alt="Concept assignment retention for ten classes and four boundary perturbations"></a><figcaption>'+bi('每行一个类别，每列一种边界变化；百分比是分配保持率。可点击放大，并下载SVG或PDF。','Rows are classes; columns are boundary changes. Percentages are assignment retention. Click to enlarge; SVG and PDF exports are available.')+'</figcaption></figure><p><a href="hub/figures/robustness.svg">SVG</a> · <a href="hub/figures/robustness.pdf">PDF</a></p>'
    macro={m['condition']:m['equal_class_mean'] for m in a['macro']}
    body+=h('读图与讨论','Analysis and discussion')
    body+=p(f"在同一十类上，腐蚀半径从1到2像素时，平均保持率下降{100*(macro['erosion1']-macro['erosion2']):.2f}个百分点；膨胀则下降{100*(macro['dilation1']-macro['dilation2']):.2f}个百分点。这是固定模型和概念对输入区域边界敏感的证据，不是概念语义或掩码准确性的判断。",f"On the same ten classes, increasing erosion radius from one to two pixels lowers mean retention by {100*(macro['erosion1']-macro['erosion2']):.2f} percentage points; the corresponding dilation decrease is {100*(macro['dilation1']-macro['dilation2']):.2f} points. This supports boundary sensitivity of the fixed model and concepts, not semantic or mask accuracy conclusions.")
    body+=p('未定义好坏阈值，不按此分数判断类别概念质量。原始零特征在膨胀后可能变为有效，但不加入固定基准保持率分母。分类别和逐图片结果保留在CSV；类别区间是原协议的2000次图片bootstrap，不是跨训练seed不确定性。','No quality cutoff is defined and classes are not graded for semantic quality by this score. Baseline zero features can become valid after dilation but do not enter the fixed retention denominator. Per-class and per-image CSVs retain detail; class intervals use the original 2,000 image-bootstrap replicates, not across-training-seed uncertainty.')
    body+=h('补跑验收与失败历史','Completion acceptance and failure history')
    times={r['JobID']:r for r in csv.DictReader((O/'final-sacct.psv').open(),delimiter='|')}
    body+=table([('类别','Class'),('原失败作业','Original failed job'),('补跑作业','Completion job'),('耗时','Elapsed'),('验收','Acceptance')],[[r['class_name'],r['previous_job'],r['job'],times[r['job']]['Elapsed'],'ACCEPTED'] for r in a['completed_classes']])
    body+=p('原失败发生在腐蚀2像素导致batch8缩为7后的未改变输入对照。补跑保留原8个槽位，仅以该槽位原输入填充空区域；占位输出被丢弃。6个受影响batch共48个对照槽位的特征与原缓存完全相同，分配不变。6个真正空区域的科学数组仍为NaN、分配为−1。','The original failure occurred when 2-pixel erosion reduced a batch of eight to seven and an unchanged-input control failed. Completion retains original slots, using each empty slot’s original input only as filler; filler outputs are discarded. Features in all 48 control slots across six affected batches exactly match the cache, with unchanged assignments. The six truly empty scientific rows remain NaN with assignment −1.')
    body+=p('模型、输入、权重、精度和容差未变；原identity和腐蚀1结果逐字节复用。新增6612条片段条件记录，累计28435/28435；旧四次失败仍保留。此结果支持槽位修复消除了执行阻塞，但不能确定原差异来自哪个GPU算子或TF32。','Model, inputs, weights, precision and tolerances are unchanged; identity and 1-pixel erosion results are reused byte-for-byte. The 6,612 new region-condition records complete 28,435/28,435; all four original failures remain preserved. These controls support resolution of the execution blocker, but do not identify the original GPU operator or TF32 mechanism.')
    body+=p('执行提交：'+a['commit']+'。原六类及复用条件来自40418edfa4e4ac81eda02d954fe59de52a0f14f0。保留cuDNN TF32开启、矩阵TF32关闭、batch8。仍记录np.bool8弃用警告；本次无新的科学验收失败。','Execution commit: '+a['commit']+'. The original six classes and reused conditions come from 40418edfa4e4ac81eda02d954fe59de52a0f14f0. cuDNN TF32 remains enabled, matrix TF32 disabled, batch size eight. The np.bool8 deprecation warning is retained; no new scientific acceptance failure was observed.')
    body+=h('结论边界','Scope of conclusions')+p('这是最终有效掩码的几何敏感性分析，不是SAM分割准确率、人的可理解性或论文Fig.7背景替换／裁剪消融。暂不需要为修复本次失败新增实验；若继续评估掩码准确性，需要独立参考标注及另行明确协议。','This is geometric sensitivity of final effective masks, not SAM segmentation accuracy, human understandability or the paper’s Fig.7 background-replacement/crop ablation. No additional experiment is needed to repair this failure; mask-accuracy evaluation would require independent reference annotations and a separately defined protocol.')
    body+=h('来源与完整记录','Sources and full records')
    for name in files+(['acceptance-events.json'] if (O/'acceptance-events.json').exists() else []):body+='<p><a href="hub/data/robustness-completion-20260924--'+name+'">'+name+'</a></p>'
    body+=p('结果验收时间（UTC）：'+a['generated_at_utc'],'Result acceptance time (UTC): '+a['generated_at_utc'])
    path=report/'robustness.html';text=path.read_text();start=text.index('</h1>')+5;end=text.index('</main>',start);path.write_text(text[:start]+body+text[end:]);changed.append(path)
    # Keep unrelated methods, concepts, medical results and review state intact.
    for name in ['index.html','paper-comparison.html']:
        path=report/name;s=path.read_text()
        s=s.replace('6 + 4','10 / 10').replace('六类完成全部条件；四类仅部分条件通过。检验分配稳定性，不是掩码准确率。','十类五条件全部验收；四类保持原batch槽位补齐，原失败历史保留。检验分配稳定性，不是掩码准确率。').replace('Six classes completed all conditions; four have accepted partial results. This measures assignment stability, not mask accuracy.','All five conditions accepted for ten classes; four fixed-slot completions retain the original failure history. This measures assignment stability, not mask accuracy.').replace('r1十类完整；其余条件仅六类完整，四类失败保留','十类五条件已验收；四类原失败及补跑证据保留').replace('r1 complete for ten classes; remaining conditions complete for six, with four failures retained','All five conditions accepted for ten classes; original four failures and completion evidence retained')
        path.write_text(s);changed.append(path)
    for name in ['issues.html','evidence.html']:
        path=report/name;s=path.read_text();s=re.sub(r'<section id="robustness-accepted-update">.*?</section>','',s,flags=re.S)
        block='<section id="robustness-accepted-update">'+h('A线更新：缺失条件已补齐','A-line update: missing conditions completed')+p('ROB26-005/006/007：十类五条件现已验收，历史失败不删除。旧快照中的部分完成状态仅表示当时状态。原浮点差异的底层机制仍未确定。','ROB26-005/006/007: all five conditions are now accepted for ten classes; historical failures remain. Partial states in older snapshots describe their collection time only. The low-level mechanism behind the original numerical difference remains unresolved.')+'<a href="robustness.html">'+bi('最新结果与完整证据','Current results and full evidence')+'</a></section>'
        s=s.replace('</h1>','</h1>'+block,1);path.write_text(s);changed.append(path)
    for name in ['report-validation.json']:
        path=data/name
        if path.exists():
            d=json.loads(path.read_text());d.update(robustness_complete_classes=10,robustness_partial_classes=0,robustness_completion_acceptance='robustness-completion-20260924--acceptance.json');path.write_text(json.dumps(d,indent=2));changed.append(path)
    source_manifest=data/'source-manifest.json'
    if source_manifest.exists():
        sources=json.loads(source_manifest.read_text());sources=[v for v in sources if not str(v.get('path','')).startswith(str(O))]
        sources.extend(dict(path=str(O/n),copy='hub/data/robustness-completion-20260924--'+n,sha256=hashlib.sha256((O/n).read_bytes()).hexdigest()) for n in files)
        source_manifest.write_text(json.dumps(sources,indent=2));changed.append(source_manifest)
    renderer=report/'renderer_manifest.json'
    if renderer.exists():
        d=json.loads(renderer.read_text());d['robustness_completion_update']=dict(at=datetime.now(timezone.utc).isoformat(),acceptance=a['generated_at_utc'],generator='scripts/research_hub/robustness_completion.py',generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Incremental A-line update; other methods/reviews preserved');renderer.write_text(json.dumps(d,indent=2));changed.append(renderer)
    # Refresh only changed manifest rows, including newly added exports/evidence.
    mf=report/'FILES.sha256.csv'
    if mf.exists():
        old=list(csv.DictReader(mf.open()));mapped={r['path']:r for r in old}
        for path in changed:mapped[path.relative_to(report).as_posix()]=dict(path=path.relative_to(report).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
        with mf.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['path','sha256','bytes']);w.writeheader();w.writerows(mapped.values())
    if (report/'hub/robustness-examples/provenance.json').exists():
        from .robustness_examples import apply as apply_visual_examples
        apply_visual_examples(report)
    (O/'report-update.json').write_text(json.dumps(dict(status='UPDATED_PENDING_BROWSER_QA',generated_at_utc=datetime.now(timezone.utc).isoformat(),report=str(report),changed=[str(p.relative_to(report)) for p in changed],new_predictions=0,source_acceptance_sha256=hashlib.sha256((O/'acceptance.json').read_bytes()).hexdigest()),indent=2))
