"""Structured navigation over existing private result pages; no scientific changes."""
from pathlib import Path
import json,re,html
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[2]
def bi(z,e):return '<span data-l="zh">'+html.escape(z)+'</span><span data-l="en" hidden>'+html.escape(e)+'</span>'
def link(url,z,e,extra=''):return '<a href="'+html.escape(url,quote=True)+'" '+extra+'>'+bi(z,e)+'</a>'
def mark(s):return '<!-- CLASS-NAV START -->'+s+'<!-- CLASS-NAV END -->'
def strip(s):return re.sub(r'<!-- CLASS-NAV START -->.*?<!-- CLASS-NAV END -->','',s,flags=re.S)
def apply(stage):
 S=Path(stage);atlas=json.loads((S/'hub/data/presentation-atlas.json').read_text());cs=atlas['classes'];changed=[]
 labels={c['name']:(c['label_zh'],c['name'].replace('_',' ').capitalize()) for c in cs}
 def tabs(active):
  return '<div class="result-tabs" role="navigation" aria-label="Reproduction sections">'+''.join(link(u,z,e,'aria-current="page"' if key==active else '') for key,u,z,e in [('classes','reproduction.html#class-directory','按类别浏览','Browse classes'),('evaluation','baseline-progress.html','十类数值评价','Ten-class evaluation'),('figures','concept-summary.html','汇报图册','Presentation atlas')])+'</div>'
 directory='<section id="class-directory" class="class-directory"><h2>'+bi('十类概念与方法对照','Concepts and method comparisons: ten classes')+'</h2><p>'+bi('先选类别，再选择阅读视角。三个入口使用相同已验收结果；概念详情仍保留在原页面。','Choose a class, then a view. All three links use the same accepted results; detailed concept evidence remains on its original pages.')+'</p><div class="table-scroll"><table><thead><tr>'+''.join('<th>'+bi(z,e)+'</th>' for z,e in [('类别','Class'),('HU-MCD','HU-MCD'),('ACE / MCD / HU-MCD','ACE / MCD / HU-MCD'),('预测影响','Prediction impact')])+'</tr></thead><tbody>'
 for c in cs:
  n=c['name'];z,e=labels[n];directory+='<tr data-class-row="'+n+'"><th scope="row">'+bi(z,e)+'</th><td>'+link('classes/'+n+'.html','全部概念','All concepts')+'</td><td>'+link('comparison-'+n+'.html','三方法对照','Compare methods','class="compare-entry"')+'</td><td>'+link('comparison-'+n+'.html#class-evaluation','双向评价','Both directions')+'</td></tr>'
 directory+='</tbody></table></div></section>'
 root_blocks={
 'reproduction.html':tabs('classes')+directory,
 'baseline-progress.html':tabs('evaluation'),
 'concept-summary.html':tabs('figures')+'<p>'+link('reproduction.html#class-directory','按类别查看三方法对照','Browse method comparisons by class')+'</p>',
 'index.html':'<section class="class-gateway"><div><h2>'+bi('十类概念与方法对照','Ten-class concepts and method comparisons')+'</h2><p>'+bi('从类别目录进入概念图、ACE/MCD/HU-MCD对照及双向评价。','Use the class directory to reach concept figures, ACE/MCD/HU-MCD comparisons and bidirectional evaluation.')+'</p></div>'+link('reproduction.html#class-directory','浏览十类结果 →','Browse ten-class results →','class="gateway-action"')+'</section>'}
 for name,block in root_blocks.items():
  path=S/name;s=strip(path.read_text())
  # Replace only our redundant prior link strip, retaining the compact figures and unrelated workstreams.
  if name in ['index.html','baseline-progress.html']:
   s=re.sub(r'<!-- COMPACT START -->.*?<!-- COMPACT END -->','',s,flags=re.S)
  if name=='reproduction.html':
   m=re.search(r'<!-- COMPACT START -->(.*?)<!-- COMPACT END -->',s,re.S)
   if m:s=s[:m.start()]+ '<!-- COMPACT START -->'+re.sub(r'<p>.*?</p>','',m.group(1),flags=re.S)+'<!-- COMPACT END -->'+s[m.end():]
  s=s.replace('</h1>','</h1>'+mark(block),1);path.write_text(s);changed.append(name)
 for c in cs:
  n=c['name'];z,e=labels[n]
  for view,name in [('concepts','classes/'+n+'.html'),('comparison','comparison-'+n+'.html')]:
   path=S/name;pre='../' if view=='concepts' else '';s=strip(path.read_text())
   # Remove the old buried one-line cross-method entry: the new header replaces it.
   s=re.sub(r'<p class="cross-method-link">.*?</p>','',s,flags=re.S)
   header='<div class="class-context"><p class="class-breadcrumb">'+link(pre+'reproduction.html#class-directory','方法复现 · 十类目录','Reproduction · class directory')+'<span aria-hidden="true"> / </span>'+bi(z,e)+'</p><label class="class-switch-label">'+bi('切换类别','Switch class')+' <select data-class-switch aria-label="Switch class">'
   for other in cs:
    k=other['name'];oz,oe=labels[k];target=pre+('classes/'+k+'.html' if view=='concepts' else 'comparison-'+k+'.html');header+='<option value="'+target+'" data-zh="'+html.escape(oz,quote=True)+'" data-en="'+html.escape(oe,quote=True)+'"'+(' selected' if k==n else '')+'>'+html.escape(oe)+'</option>'
   header+='</select></label><div class="class-view-tabs" role="navigation" aria-label="Views for this class">'+link(pre+'classes/'+n+'.html','HU-MCD 概念','HU-MCD concepts','aria-current="page"' if view=='concepts' else '')+link(pre+'comparison-'+n+'.html','三方法对照','Method comparison','aria-current="page"' if view=='comparison' else '')+link(pre+'comparison-'+n+'.html#class-evaluation','本类评价','Class evaluation','data-evaluation-link')+'</div></div>'
   s=s.replace('</h1>','</h1>'+mark(header),1)
   if view=='comparison':
    pattern=r'<figure>(?=\s*<a[^>]*href="hub/figures/baselines-20260924/'+re.escape(n)+r'\.png")'
    if 'id="class-evaluation"' not in s:s,count=re.subn(pattern,'<figure id="class-evaluation">',s,count=1);assert count==1,('Evaluation figure not found',n)
   path.write_text(s);changed.append(name)
 # In the atlas, put class-view links before each tall class figure, not after it.
 path=S/'concept-summary.html';s=path.read_text();s=re.sub(r'<!-- CLASS-ATLAS-NAV START -->.*?<!-- CLASS-ATLAS-NAV END -->','',s,flags=re.S)
 for c in cs:
  n=c['name'];pattern=r'(<section id="'+re.escape(n)+r'"><h2>.*?</h2>)';row='<div class="atlas-class-links">'+link('classes/'+n+'.html','本类概念详情','Class concept details')+' · '+link('comparison-'+n+'.html','三方法对照','Compare methods')+' · '+link('comparison-'+n+'.html#class-evaluation','本类评价','Class evaluation')+'</div>';s,count=re.subn(pattern,lambda m:m.group(1)+'<!-- CLASS-ATLAS-NAV START -->'+row+'<!-- CLASS-ATLAS-NAV END -->',s,count=1,flags=re.S);assert count==1,n
 path.write_text(s)
 css=S/'hub/style.css';text=css.read_text();text=re.sub(r'/\* CLASS NAV START \*/.*?/\* CLASS NAV END \*/','',text,flags=re.S)
 text+='''\n/* CLASS NAV START */
.class-gateway{display:flex;gap:1.5rem;align-items:center;justify-content:space-between;background:#edf5f5;border-left:4px solid #087f8c;padding:1.1rem 1.4rem;margin:1rem 0 1.5rem}.class-gateway h2{margin:0 0 .4rem}.class-gateway p{margin:0}.gateway-action{background:#087782;color:white!important;padding:.85rem 1.1rem;border-radius:5px;white-space:nowrap;text-decoration:none}.result-tabs,.class-view-tabs{display:flex;flex-wrap:wrap;gap:.45rem;margin:.75rem 0 1rem}.result-tabs a,.class-view-tabs a{padding:.5rem .85rem;border:1px solid #c8d8dc;border-radius:5px;text-decoration:none}.result-tabs a[aria-current],.class-view-tabs a[aria-current]{background:#e6f2f3;border-color:#087f8c;color:#084e5b;font-weight:650}.class-context{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.25rem 1rem;margin:.65rem 0 1.5rem;border-bottom:1px solid #d2dfe2}.class-context .class-breadcrumb{margin:.25rem 0;font-size:.92rem}.class-switch-label{display:flex;align-items:center;gap:.5rem}.class-switch-label select{max-width:15rem;padding:.45rem;border:1px solid #b5c9cf;border-radius:4px;background:white}.class-view-tabs{flex-basis:100%;margin:.55rem 0 .85rem}.class-directory{scroll-margin-top:1rem}.class-directory table{width:100%}.class-directory th,.class-directory td{padding:.6rem .8rem}.compare-entry{font-weight:650}.atlas-class-links{margin:.6rem 0 1rem}@media(max-width:700px){.class-gateway{align-items:flex-start;flex-direction:column;gap:1rem}.class-directory th,.class-directory td{padding:.55rem;min-width:6rem}.class-context{align-items:flex-start}.class-switch-label{width:100%}.class-view-tabs a{padding:.5rem .55rem}}\n/* CLASS NAV END */\n''';css.write_text(text)
 (S/'hub/class-navigation.js').write_text('''(()=>{document.querySelectorAll('[data-class-switch]').forEach(s=>s.addEventListener('change',()=>{const u=new URL(s.value,location.href);u.searchParams.set('lang',document.documentElement.lang==='en'?'en':'zh');if(location.hash==='#class-evaluation'&&u.pathname.includes('/comparison-'))u.hash='class-evaluation';location.href=u.href;}));function active(){if(!document.querySelector('[data-class-switch]'))return;const tabs=document.querySelector('.class-view-tabs');if(!tabs)return;tabs.querySelectorAll('a').forEach(a=>{a.removeAttribute('aria-current');const u=new URL(a.href,location.href);if(u.pathname===location.pathname&&((location.hash==='#class-evaluation')===!!a.hasAttribute('data-evaluation-link')))a.setAttribute('aria-current','page');});}addEventListener('hashchange',active);active();})();\n''')
 for name in changed:
  path=S/name;s=path.read_text();src=('../' if name.startswith('classes/') else '')+'hub/class-navigation.js'
  if src not in s:s=s.replace('</body>','<script src="'+src+'"></script></body>');path.write_text(s)
 out=S/'hub/data/navigation-update.json';out.write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),classes=10,changed_pages=changed,entry='reproduction.html#class-directory',scientific_changes=0,new_pages=0,review_storage_changes=0),indent=2)+'\n')
 print('Navigation updated:',len(changed),'existing pages, ten class rows; no new report entry.')
if __name__=='__main__':
 import sys
 apply(sys.argv[1])
