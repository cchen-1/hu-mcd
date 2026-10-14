from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import json,sys,hashlib
S=Path(sys.argv[1]).resolve()
class Parse(HTMLParser):
 def __init__(self):super().__init__();self.refs=[];self.ids=set();self.hub=False
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.add(a['id'])
  if 'data-hub' in a:self.hub=True
  for name in ['src','href']:
   if name in a:self.refs.append((tag,name,a[name]))
parsers={}
for f in S.rglob('*.html'):
 p=Parse();p.feed(f.read_text());parsers[f]=p
bad=[];out=[];n=0
for f,p in parsers.items():
 for tag,name,ref in p.refs:
  u=urlsplit(ref)
  if u.scheme or u.netloc or not ref:continue
  n+=1;target=(f.parent/unquote(u.path)).resolve() if u.path else f
  if not target.is_relative_to(S):out.append([str(f.relative_to(S)),ref]);continue
  if not target.exists():bad.append([str(f.relative_to(S)),ref,'missing file'])
  elif u.fragment and target in parsers and unquote(u.fragment) not in parsers[target].ids:bad.append([str(f.relative_to(S)),ref,'missing anchor'])
r={'html_pages':len(parsers),'core_pages':sum(p.hub for p in parsers.values()),'local_references':n,'broken':bad,'outside_bundle':out};(S/'hub/data/link-validation.json').write_text(json.dumps(r,indent=2));print(json.dumps({'pages':len(parsers),'references':n,'broken':len(bad),'outside':len(out),'first':bad[:15]},ensure_ascii=False));sys.exit(bool(bad or out))
