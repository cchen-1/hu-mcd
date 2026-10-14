"""Adapt saved supporting pages to the report language control; no scientific changes.
Run with the bundled Python (lxml), passing the staged report root.
"""
from pathlib import Path
import sys,re,json,os
from lxml import html,etree
from legacy_translations import translate
S=Path(sys.argv[1]);unknown={};count=0
CJK=re.compile('[\u3400-\u9fff]')
def bi(z,e):
 box=html.Element('span');a=etree.SubElement(box,'span',{'data-l':'zh'});a.text=z;b=etree.SubElement(box,'span',{'data-l':'en','hidden':'hidden'});b.text=e;return box
# Structural translation uses existing paired source translations only.
for path in sorted(S.rglob('*.html')):
 doc=html.document_fromstring(path.read_text(encoding='utf-8'))
 body=doc.find('body')
 if body is None or body.get('data-hub'):continue
 source=S.parent/'source-report'/path.relative_to(S)
 if source.exists():doc=html.document_fromstring(source.read_text(encoding='utf-8'));body=doc.find('body')
 for a in doc.xpath('//a[@href="../index.html#workstreams"]'):a.set('href','../evidence.html#jobs')
 count+=1
 for node in doc.xpath('//*[contains(concat(" ",normalize-space(@class)," ")," en ")]'):
  if node.tag in ['script','style']:continue
  prev=node.getprevious()
  if prev is not None and CJK.search(prev.text_content()) and not prev.get('data-l'):
   node.set('data-l','en');node.set('hidden','hidden');prev.set('data-l','zh')
 for node in list(doc.iter()):
  if node.tag in ['script','style','pre','code','title'] or not isinstance(node.tag,str):continue
  if node.get('data-l') or any(x.get('data-l') for x in node.iterancestors()):continue
  for field in ['text','tail']:
   text=getattr(node,field)
   if not text or not CJK.search(text):continue
   translated=translate(text)
   parts=re.split(r'\s+/\s*|\s*/\s+',text,maxsplit=1)
   z,e=text,translated
   if e is None and len(parts)==2:
    if CJK.search(parts[0]) and not CJK.search(parts[1]) and re.search('[A-Za-z]',parts[1]):z,e=parts
    elif not CJK.search(parts[0]) and CJK.search(parts[1]):
     e=parts[0];z=parts[1]
     # Preserve trailing English protocol caveats after the Chinese parallel sentence.
     extras=re.findall(r'(?<=[。；])\s*([A-Z][^\u3400-\u9fff]+)(?= /|$)',z)
     if extras:e+=' '+' '.join(extras)
   if e is not None:
    if node.tag=='option' and field=='text':node.set('data-zh',z);node.set('data-en',e);node.text=z;continue
    child=bi(z,e)
    if field=='text':node.text=None;node.insert(0,child)
    else:node.tail=None;node.addnext(child)
   else:unknown.setdefault(text,[]).append(str(path.relative_to(S)))
 for table in list(doc.xpath('//table')):
  wrapper=html.Element('div',{'style':'overflow:auto;max-width:100%;margin:12px 0'});table.addprevious(wrapper);wrapper.append(table)
 pre=os.path.relpath(S/'hub',path.parent).replace('\\','/')
 head=doc.find('head');head.append(html.Element('script',src=pre+'/legacy.js',defer='defer'));style=html.Element('style');style.text='body{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}input,select,textarea{max-width:100%}[hidden]{display:none!important}.hubbar{display:flex;justify-content:space-between;gap:20px;padding:14px 18px;background:#142f3d;color:white;margin:0 0 24px;border-radius:6px}.hubbar a{color:white}.hubbar button{font:inherit;padding:6px 14px;background:white;color:#173d4b;border:0;border-radius:4px}';head.append(style)
 bar=html.Element('div',{'class':'hubbar'});a=etree.SubElement(bar,'a',href=pre+'/../index.html');a.append(bi('← 研究报告总览','← Research overview'));button=etree.SubElement(bar,'button',id='language');button.text='English';body.insert(0,bar)
 doc.set('lang','zh');doc.set('data-lang','zh');body.set('data-adapted','1')
 path.write_text('<!doctype html>'+html.tostring(doc,encoding='unicode'),encoding='utf-8')
(S/'hub/data/legacy-translation-unmatched.json').write_text(json.dumps(unknown,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pages':count,'unmatched_unique':len(unknown)}))
