"""Readable fixed-scale heatmap; missing outcomes are never zero."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def build(rows, class_names):
    conditions=['erosion1','erosion2','dilation1','dilation2']
    values=np.full((len(class_names),4),np.nan)
    for i,name in enumerate(class_names):
        for j,condition in enumerate(conditions):
            selected=[r for r in rows if r['class_name']==name and r['condition']==condition]
            if len(selected)>1:raise ValueError('Duplicate class/condition')
            if selected:values[i,j]=float(selected[0]['image_mean_retention'])
    fig,ax=plt.subplots(figsize=(12,7.5))
    fig.subplots_adjust(left=.22,right=.89,bottom=.16,top=.86)
    cm=plt.colormaps['YlGnBu'].copy();cm.set_bad('#e4e8eb')
    im=ax.imshow(values,vmin=0,vmax=1,cmap=cm,aspect='auto')
    ax.set_yticks(range(len(class_names)),[n.replace('_',' ').title() for n in class_names],fontsize=12)
    ax.set_xticks(range(4),['Erode\n1 pixel','Erode\n2 pixels','Dilate\n1 pixel','Dilate\n2 pixels'],fontsize=12)
    ax.tick_params(length=0,pad=10)
    for i in range(len(class_names)):
        for j in range(4):
            v=values[i,j]
            ax.text(j,i,'—' if np.isnan(v) else f'{v:.1%}',ha='center',va='center',fontsize=12,color='white' if v>.60 else '#21313b')
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,len(class_names),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.5);ax.tick_params(which='minor',length=0)
    fig.suptitle('Do regions keep their original concept assignment?',fontsize=18,y=.97)
    fig.text(.22,.91,'Each cell: average retention across 50 images in one class',fontsize=12)
    cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.035);cb.ax.yaxis.set_major_formatter(PercentFormatter(1));cb.set_label('Assignment retention',fontsize=12)
    note='Grey / — = no accepted result. Not zero.' if np.isnan(values).any() else 'All 40 class × perturbation outcomes accepted.'
    fig.text(.22,.035,note+'\nFixed images, model and concepts; this is not segmentation accuracy.',fontsize=11,color='#344956')
    return fig


def update_notice(report):
    """Publish submission status separately from unchanged accepted outcomes."""
    from pathlib import Path
    import json,re,shutil
    root=Path(__file__).resolve().parents[2]
    if (root/'artifacts/bunya/robustness-completion-20260924/acceptance.json').exists():
        from .robustness_completion import apply
        apply(report)
        return
    source=root/'artifacts/bunya/robustness-completion-20260924/batch.json'
    if not source.exists():return
    report=Path(report);batch=json.loads(source.read_text())
    jobs=', '.join(r['class_name']+' '+r['job_id'] for r in batch['jobs'])
    note='<section id="robustness-completion-20260924" class="note"><p><span data-l="zh">2026-09-24：已授权的保持batch槽位补跑已提交：'+jobs+'。原验收图表尚未更新为补跑结果；待用户通知完成后收集验收。没有重新运行SAM或概念拟合。</span><span data-l="en" hidden>24 Sep 2026: authorized fixed-slot completion submitted: '+jobs+'. The chart still shows previously accepted outcomes, not new completion results. Collection and acceptance follow the user’s completion notice. No SAM or concept refitting.</span></p><a href="hub/data/robustness-completion-20260924.json">Submission receipts / 提交凭据</a></section>'
    p=report/'robustness.html';s=p.read_text()
    s=re.sub(r'<section id="robustness-completion-20260924".*?</section>','',s,flags=re.S)
    p.write_text(s.replace('</h1>','</h1>'+note,1))
    shutil.copy2(source,report/'hub/data/robustness-completion-20260924.json')
