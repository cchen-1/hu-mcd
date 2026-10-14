"""Presentation-only compact figures from frozen examples and accepted masks."""
from pathlib import Path
import json,csv,re,hashlib,math,os
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[2];O=R/'artifacts/report-compact-20260924';T=R/'artifacts/bunya/ten-class-review'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bi(z,e):return '<span data-l="zh">'+z+'</span><span data-l="en" hidden>'+e+'</span>'
def p(z,e):return '<p>'+bi(z,e)+'</p>'
def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
def fig(rel,z,e):return '<figure><a href="'+rel+'" target="_blank"><img loading="lazy" src="'+rel+'" alt="'+e+'"></a><figcaption>'+bi(z,e)+'</figcaption></figure>'
def insert_first(text,body):return text.replace('</h1>','</h1>'+body,1)
def remove_block(text,name):return re.sub(r'<!-- '+name+' START -->.*?<!-- '+name+' END -->','',text,flags=re.S)
def mark(name,body):return '<!-- '+name+' START -->'+body+'<!-- '+name+' END -->'
def apply(stage):
 S=Path(stage);out=S/'hub/compact';out.mkdir(parents=True,exist_ok=True);assets=out/'examples';assets.mkdir(exist_ok=True)
 atlas=load(S/'hub/data/presentation-atlas.json');cards=load(S/'hub/interpretation/comparison-cards.json');records=[];figures=[];cache={}
 def save(im,name,role,ids):
  path=out/name
  if path.exists():path.unlink()
  im.save(path,dpi=(300,300));figures.append(dict(file='hub/compact/'+name,width=im.width,height=im.height,role=role,concept_ids=ids,sha256=sha(path)))
 def fit(im,path,box):
  x,y,w,h=box;v=Image.open(path).convert('RGB');scale=min(w/v.width,h/v.height);v=v.resize((max(1,round(v.width*scale)),max(1,round(v.height*scale))),Image.Resampling.LANCZOS);im.paste(v,(int(x+(w-v.width)/2),int(y+(h-v.height)/2)))
 def heading(im,title,sub):
  d=ImageDraw.Draw(im);d.text((40,22),title,font=font(44),fill='#173441');d.text((40,83),sub,font=font(26),fill='#385664')
 def get_hu(c,k,e,view,pos):
  key=e['id'];path=assets/(key+'.png')
  home=T/'evidence'/str(c['job_id']);split=e['split'];mk=(c['name'],split)
  if mk not in cache:cache[mk]=load(home/(split+'_row_mapping.json'))
  row=cache[mk][e['row_index']];assert row['source_id']==e['image_id'] and row['raw_mask_index']==e['raw_mask_index']
  masks=np.load(home/'cache'/('segments_train' if split=='training' else 'segments_validation')/(row['input_name']+'_sam.npy'),mmap_mode='r');m=masks[e['raw_mask_index']]
  assert hashlib.sha256(m.tobytes()).hexdigest()==row['mask_sha256'];m=m.astype(bool)
  fraction=float(m.sum()/m.size);assert abs(fraction-e['mask_fraction'])<1e-6
  original=S/e['original'];assert sha(original)==sha(home/'images'/split/row['input_name'])
  if not path.exists():
   im=Image.open(original);w,h=im.size
   if min(w,h)>300:im=im.resize((300,int(300*h/w)) if h>w else (int(300*w/h),300),Image.Resampling.LANCZOS)
   a=np.asarray(im,dtype=float);assert a.shape[:2]==m.shape
   a[m]=a[m]*.6+np.array([0,210,225])*.4;a[~m]=a[~m]*.4+190*.6;Image.fromarray(np.rint(a).astype('uint8')).save(path)
  rec=dict(method='HU-MCD',class_name=c['name'],concept_id=k['id'],rank=k['rank'],view=view,position=pos,example_id=key,source_image=e['image_id'],split=split,selected_pixels=int(m.sum()),canvas_pixels=int(m.size),canvas_height=m.shape[0],canvas_width=m.shape[1],selected_percent=100*fraction,mask_sha256=row['mask_sha256'],image_sha256=sha(original),figure_asset='hub/compact/examples/'+key+'.png',source_view_rule=k['views']['validation_'+view]['rule'])
  records.append(rec);return rec
 hu={}
 for c in atlas['classes']:
  kk=sorted((k for k in c['concepts'] if not k['is_complement']),key=lambda k:k['rank']);hu[c['name']]=[]
  for k in kk:
   v={view:[get_hu(c,k,e,view,i+1) for i,e in enumerate(k['views']['validation_'+view]['examples'][:4])] for view in ['top','random']}
   hu[c['name']].append(dict(id=k['id'],rank=k['rank'],views=v))
 assert sum(map(len,hu.values()))==147
 def group(im,k,view,x,y,w,h):
  d=ImageDraw.Draw(im);d.rounded_rectangle((x,y,x+w,y+h),radius=10,fill='#f0f4f5')
  d.text((x+12,y+9),k['id']+' | rank '+str(k['rank']),font=font(22),fill='#173441')
  es=k['views'][view]
  for i in range(4):
   xx=x+10+(i%2)*(w-15)/2;yy=y+48+(i//2)*(h-56)/2;ww=(w-30)/2;hh=(h-75)/2
   if i<len(es):
    e=es[i];fit(im,S/e['figure_asset'],(xx,yy,ww,hh-27));d.text((xx+4,yy+hh-24),'e'+str(i+1)+' | '+f"{e['selected_percent']:.1f}% selected",font=font(19),fill='#385664')
   else:d.text((xx+12,yy+20),'No saved example',font=font(22),fill='#8c684d')
 def hu_sheet(n,ks,view,name,label):
  rows=math.ceil(len(ks)/4);height=180+rows*620+70;im=Image.new('RGB',(3600,height),'white');heading(im,'HU-MCD | '+n.replace('_',' ')+' | '+label,'Fixed validation examples | '+('saved prototypes' if view=='top' else 'saved random members')+' | cyan SELECTED; grey HIDDEN')
  for i,k in enumerate(ks):group(im,k,view,35+(i%4)*890,145+(i//4)*620,865,595)
  ImageDraw.Draw(im).text((40,height-45),'Stable IDs. No semantic alignment, image cropping, resampling or model calls. Private / human-unreviewed.',font=font(24),fill='#785a46');save(im,name,'all-HU-concepts-'+view,[k['id'] for k in ks])
 for c in atlas['classes']:
  n=c['name'];ks=hu[n]
  for view in ['top','random']:hu_sheet(n,ks,view,n+'-HU-all-'+view+'.png',str(len(ks))+' learned concepts (all)')
  for i in range(0,len(ks),8):hu_sheet(n,ks[i:i+8],'top',n+'-HU-slide-'+str(i//8+1)+'.png','ranks '+str(i+1)+'-'+str(min(i+8,len(ks)))+' / '+str(len(ks)))
 # Fig.3-style layout, extended to all ten classes under one pre-declared rank rule.
 def overview(cs,name):
  height=180+len(cs)*520+70;im=Image.new('RGB',(3600,height),'white');heading(im,'HU-MCD | Ten-class concept overview','Fig.3-style layout, not author-matched concepts | saved top4 ranks / 4 saved validation prototypes each')
  ids=[]
  for i,c in enumerate(cs):
   y=145+i*520;n=c['name'];d=ImageDraw.Draw(im)
   for j,part in enumerate(n.replace('_',' ').split()):d.text((30,y+90+j*39),part,font=font(27),fill='#173441')
   d.text((30,y+230),str(len(hu[n]))+' learned',font=font(22),fill='#385664')
   for j,k in enumerate(hu[n][:4]):group(im,k,'top',265+j*830,y,810,500);ids.append(k['id'])
  ImageDraw.Draw(im).text((40,height-45),'Cyan SELECTED; grey HIDDEN. Top4 is a display subset; all147 learned concepts remain available. Private / unreviewed.',font=font(24),fill='#785a46');save(im,name,'ten-class-HU-top4',ids)
 overview(atlas['classes'],'HU-ten-class-overview.png')
 for i in range(0,10,3):overview(atlas['classes'][i:i+3],'HU-overview-slide-'+str(i//3+1)+'.png')
 # Each method gets one row. Keep prototype/random side by side, without implying semantic pairing.
 for c in atlas['classes']:
  n=c['name'];im=Image.new('RGB',(3600,1800),'white');heading(im,n.replace('_',' ')+' | Three-method concept summary','Each row = one method; columns = saved ranks1-4, NOT matching semantics. Each pair: P prototype / R fixed random.')
  ids=[];d=ImageDraw.Draw(im)
  for mi,m in enumerate(['ACE','MCD','HU-MCD']):
   y=150+mi*510;ks=sorted((k for k in cards if k['class_name']==n and k['method']==m),key=lambda k:k['rank']);d.text((30,y+90),m,font=font(36),fill='#173441');d.text((30,y+145),str(len(ks))+' learned',font=font(22),fill='#385664')
   if m=='ACE':d.text((30,y+188),'released-code',font=font(19),fill='#385664');d.text((30,y+215),'reference',font=font(19),fill='#385664')
   for j in range(4):
    x=245+j*835;d.rounded_rectangle((x,y,x+810,y+485),radius=10,fill='#f0f4f5')
    if j>=len(ks):d.text((x+55,y+200),'No retained concept',font=font(27),fill='#8c684d');continue
    k=ks[j];ids.append(k['id']);d.text((x+12,y+12),k['id'],font=font(23),fill='#173441')
    for vi,view in enumerate(['top','random']):
     e=k['views']['validation_'+view];xx=x+12+vi*400
     if e:
      fit(im,S/e['overlay'],(xx,y+55,386,375));d.text((xx+5,y+442),('P' if vi==0 else 'R')+' | '+f"{100*e['mask_fraction']:.1f}% selected",font=font(22),fill='#385664');records.append(dict(method=m,class_name=n,concept_id=k['id'],rank=k['rank'],view='method-summary-'+view,position=1,example_id=e['id'],source_image=e['source_image'],selected_percent=100*e['mask_fraction'],figure_asset=e['overlay'],source_view_rule=e['rule']))
     else:d.text((xx+12,y+200),'No saved member',font=font(24),fill='#8c684d')
  d.text((40,1710),'Cyan SELECTED; grey HIDDEN. Validation examples / different images. MCD P uses >.4 masks; R uses final assignment.',font=font(25),fill='#785a46');d.text((40,1750),'Discovery images: ACE50, MCD/HU400. ACE validation P = nearest-member display extension. Private / human-unreviewed.',font=font(25),fill='#785a46');save(im,n+'-methods.png','three-method-summary',ids)
 # Native collapsible HTML, no frontend dependency or storage changes.
 guide=p('紧凑图采用已保存的排名和示例；青色是所选区域，灰色是被遮蔽区域。相同排名不是语义对应，原型不是随机总体代表。点击图片可查看原尺寸。','Compact figures reuse saved ranks and examples. Cyan is selected; grey is hidden. Equal ranks are not semantic matches and prototypes are not representative random samples. Click for full resolution.')
 for c in atlas['classes']:
  n=c['name'];path=S/('comparison-'+n+'.html');text=remove_block(path.read_text(),'COMPACT')
  body=fig('hub/compact/'+n+'-methods.png','每行一个方法；每组P原型、R固定随机成员。','One method per row; P prototype, R fixed random member.')+guide+p('<a href="concept-summary.html#'+n+'">HU-MCD全部概念紧凑图</a> · <a href="masking-and-area.html">掩码、inpainting与面积说明</a>','<a href="concept-summary.html#'+n+'">All HU-MCD concepts</a> · <a href="masking-and-area.html">Masking, inpainting and area definitions</a>')
  text=insert_first(text,mark('COMPACT',body))
  if '<details class="compact-details" id="method-member-details">' not in text:
   a=text.index('<section id="prototype">');b=text.index('</section>',text.index('<section id="random">'))+len('</section>');old=text[a:b];text=text[:a]+'<details class="compact-details" id="method-member-details"><summary>'+bi('展开原有逐方法、逐概念详细视图','Expand original per-method, per-concept detailed views')+'</summary>'+old+'</details>'+text[b:]
  path.write_text(text)
  path=S/'classes'/(n+'.html');text=remove_block(path.read_text(),'COMPACT');body=fig('../hub/compact/'+n+'-HU-all-top.png','本类全部学习概念；每概念沿用前4个已保存验证原型。','All learned concepts in this class; first4 saved validation prototypes per concept.')+p('<a href="../concept-summary.html#'+n+'">分页导出及随机成员对照</a>','<a href="../concept-summary.html#'+n+'">Paged exports and random-member comparison</a>');text=insert_first(text,mark('COMPACT',body))
  if 'id="original-concept-cards"' not in text:
   start=text.rfind('<h2>',0,text.index('<div class="tools">'));end=text.index('<h2>',text.index('<div class="concept-grid">'))
   text=text[:start]+'<details class="compact-details" id="original-concept-cards"><summary>'+bi('展开原有完整概念索引与详细卡片','Expand original complete concept index and detailed cards')+'</summary>'+text[start:end]+'</details>'+text[end:]
  path.write_text(text)
 template=(S/'baseline-progress.html').read_text()
 def page(name,title,body):
  main='<main id="main"><div class="eyebrow">'+bi('汇报图与方法说明 · 2026-09-24','Presentation figures and methods · 24 Sep2026')+'</div><h1>'+bi(*title)+'</h1>'+body+'</main>';s=re.sub(r'<main\b[^>]*>.*?</main>',lambda _:main,template,flags=re.S);s=re.sub(r'<title>.*?</title>','<title>'+title[1]+' | HU-MCD</title>',s);(S/name).write_text(s)
 body=fig('hub/compact/HU-ten-class-overview.png','十类总览：每类按保存排名展示4个概念，每概念4个已保存验证原型。','Ten-class overview:4 saved-rank concepts per class,4 saved validation prototypes per concept.')+guide
 body+=p('这张总览显示40/147个学习概念，不是全部。下方按类完整保留147个概念（补空间另列在原详情）；未按视觉效果筛除任何类别或概念。原文Fig.3每概念6例，这里沿用现有4例，未为凑数新增抽样。它们是ImageNet类别；CIFAR-10 alike不是说使用了CIFAR-10数据。','This overview shows40/147 learned concepts, not all of them. The complete class figures below include all147 (complements remain separate in original details); no class or concept is removed based on appearance. Fig.3 has6 examples per concept; we reuse the existing4 without new sampling. These are ImageNet classes; CIFAR-10 alike does not mean CIFAR-10 data.')
 body+=p('总览汇报分页：'+' · '.join('<a href="hub/compact/HU-overview-slide-'+str(i)+'.png">'+str(i)+'</a>' for i in range(1,5)),'Overview presentation pages: '+' · '.join('<a href="hub/compact/HU-overview-slide-'+str(i)+'.png">'+str(i)+'</a>' for i in range(1,5)))
 body+='<nav class="compact-nav">'+' · '.join('<a href="#'+c['name']+'">'+c['name'].replace('_',' ')+'</a>' for c in atlas['classes'])+'</nav>'
 for c in atlas['classes']:
  n=c['name'];body+='<section id="'+n+'"><h2>'+bi(c['label_zh']+' · '+str(len(hu[n]))+'个学习概念',n.replace('_',' ')+' · '+str(len(hu[n]))+' learned concepts')+'</h2>'+fig('hub/compact/'+n+'-HU-all-top.png','全部学习概念，保存的全局排名顺序；未人工审阅。','All learned concepts in saved global-rank order; human-unreviewed.')
  body+=p('汇报分页：'+' · '.join('<a href="hub/compact/'+n+'-HU-slide-'+str(i)+'.png">'+str(i)+'</a>' for i in range(1,math.ceil(len(hu[n])/8)+1)),'Presentation pages: '+' · '.join('<a href="hub/compact/'+n+'-HU-slide-'+str(i)+'.png">'+str(i)+'</a>' for i in range(1,math.ceil(len(hu[n])/8)+1)))
  body+='<details class="compact-details"><summary>'+bi('展开全部概念的固定随机成员对照','Expand fixed random members for all concepts')+'</summary>'+fig('hub/compact/'+n+'-HU-all-random.png','相同概念与排名，不同选择规则；不是原型。','Same concepts and ranks, different selection rule; not prototypes.')+'</details>'+p('<a href="classes/'+n+'.html#original-concept-cards">原有概念卡片与训练／验证详情</a> · <a href="comparison-'+n+'.html">三方法对比</a>','<a href="classes/'+n+'.html#original-concept-cards">Original concept cards and train/validation details</a> · <a href="comparison-'+n+'.html">Three-method comparison</a>')+'</section>'
 body+=p('<a href="hub/compact/selection.csv">全部示例映射CSV</a> · <a href="hub/compact/figures.json">图像清单与哈希</a>','<a href="hub/compact/selection.csv">Complete example mapping CSV</a> · <a href="hub/compact/figures.json">Figure inventory and hashes</a>')
 page('concept-summary.html',('HU-MCD十类紧凑概念图：总览与全部147个学习概念','HU-MCD ten-class compact atlas: overview and all147 learned concepts'),body)
 faq=load(O/'methods-faq.json');body=''.join('<section><h2>'+bi(a['title_zh'],a['title_en'])+'</h2>'+''.join(p(b['zh'],b['en']) for b in a['paragraphs'])+'</section>' for a in faq)
 body+=p('<a href="hub/compact/source-evidence.json">源码版本、位置与证据</a> · <a href="hub/compact/selection.csv">面积分子、分母与示例身份</a>','<a href="hub/compact/source-evidence.json">Source versions, locations and evidence</a> · <a href="hub/compact/selection.csv">Area numerators, denominators and example identities</a>');page('masking-and-area.html',('Input Masking、inpainting与selected area','Input Masking, inpainting and selected area'),body)
 for name in ['index.html','reproduction.html','baseline-progress.html']:
  path=S/name;text=remove_block(path.read_text(),'COMPACT');body=p('<a href="concept-summary.html">十类紧凑概念图／全部147概念／汇报分页</a> · <a href="masking-and-area.html">inpainting及selected area解释</a>','<a href="concept-summary.html">Compact ten-class atlas / all147 concepts / presentation pages</a> · <a href="masking-and-area.html">Inpainting and selected area explained</a>')
  if name=='reproduction.html':body=fig('hub/compact/HU-ten-class-overview.png','十类统一规则总览（40/147概念）；全部概念见链接。','Ten-class overview with one selection rule (40/147 concepts); all concepts at the link.')+body
  path.write_text(insert_first(text,mark('COMPACT',body)))
 fields=list(dict.fromkeys(k for r in records for k in r))
 with (out/'selection.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
 (out/'figures.json').write_text(json.dumps(figures,indent=2)+'\n');(out/'source-evidence.json').write_text((O/'source-evidence.json').read_text());(out/'methods-faq.json').write_text(json.dumps(faq,ensure_ascii=False,indent=2)+'\n')
 css=S/'hub/style.css';add='\n/* Compact presentation20260924 */\n.compact-details{border:1px solid #b9ccd2;padding:1rem;margin:1rem 0;border-radius:6px}.compact-details>summary{font-weight:650;cursor:pointer;color:#116977}.compact-nav{line-height:2.1}details:target{outline:2px solid #1294a1}\n'
 if 'Compact presentation20260924' not in css.read_text():css.write_text(css.read_text()+add)
 # Deep links into folded evidence remain usable without changing review storage.
 js=S/'hub/compact-folds.js';js.write_text("(()=>{function openHash(){let e=document.getElementById(decodeURIComponent(location.hash.slice(1)));if(!e)return;for(let p=e;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;e.scrollIntoView();}addEventListener('hashchange',openHash);addEventListener('DOMContentLoaded',openHash);})();\n")
 for path in [S/('comparison-'+c['name']+'.html') for c in atlas['classes']]+[S/'classes'/(c['name']+'.html') for c in atlas['classes']]:
  text=path.read_text();src='../hub/compact-folds.js' if path.parent.name=='classes' else 'hub/compact-folds.js'
  if src not in text:text=text.replace('</body>','<script src="'+src+'"></script></body>');path.write_text(text)
 result=dict(status='PASS',at=datetime.now(timezone.utc).isoformat(),classes=10,HU_learned=147,overview_concepts=40,figures=len(figures),frozen_example_records=len(records),new_predictions=0,new_sampling=0,style='Exact source masks, unified cyan/grey, full canvas fitted without crop',source_atlas_sha256=sha(S/'hub/data/presentation-atlas.json'))
 (O/'build-result.json').write_text(json.dumps(result,indent=2)+'\n')
 if __package__:
  from .navigation import apply as apply_navigation
 else:
  from navigation import apply as apply_navigation
 apply_navigation(S)
 print(json.dumps(result))
if __name__=='__main__':
 import sys
 apply(sys.argv[1])
