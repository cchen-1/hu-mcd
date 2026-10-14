"""Portable Markdown copy of the accepted meeting manuscript; no scientific edits."""
from pathlib import Path
import re,shutil,zipfile,json,hashlib
R=Path(__file__).resolve().parents[1]
source=R/'artifacts/bunya/ten-class-review/report/hub/medical-meeting'
out=R/'artifacts/exports/medical-meeting-loop-20260924'
out.mkdir(parents=True,exist_ok=True)
s=(source/'meeting-report.md').read_text()
# Remove report-site navigation, retaining all scientific paragraphs and result tables.
s=re.sub(r'^\[Medical transfer overview\].*\n','',s,flags=re.M)
figures=['concept-summary.png','same-case.png'];seen=[]
def figure(m):
 name=Path(m[2]).name;assert name in figures;seen.append(name)
 return '\n\n**[Insert Figure '+str(figures.index(name)+1)+' here: `'+name+'`]**\n\n'
s=re.sub(r'\[!\[(.*?)\]\(([^)]+)\)\]\(([^)]+)\)',figure,s)
assert seen==figures
s=s.replace(' Click for full-resolution export.','')
# Remove local-site-only link groups instead of leaving broken links in Loop.
s=re.sub(r'^\[A: source, held-out and external members together\].*$', 'The detailed private report retains the source, held-out and external members, complements, all saved examples and exploratory review forms.',s,flags=re.M)
s=re.sub(r' \[Classifier receipt\].*?\[source-image audit\]\([^)]+\)\.', ' Source identities and model provenance are documented in the accepted audit records.',s)
s=re.sub(r' \[Cluster-to-assignment table\]\([^)]+\)\.', ' The full cluster-to-assignment table is retained in the private report.',s)
s=re.sub(r' \[All 12 contrasts and interpretation limits\].*?\[full-precision results, including complement\]\([^)]+\)\.', ' All 12 contrasts and separate complement results are retained in the private report.',s)
s=re.sub(r'^\[Figure IDs, masks and hashes\].*$', 'Figure IDs, mask hashes, source receipts and execution versions are retained in the private project report. This copy preserves its scientific findings and limitations.',s,flags=re.M)
s=re.sub(r'\n{3,}','\n\n',s).strip()+'\n'
assert '../../' not in s and 'file://' not in s and '[![' not in s
assert s.count('**[Insert Figure ')==2
assert s.count('\n| ')==(source/'meeting-report.md').read_text().count('\n| ')
# Numeric content is retained, apart from repeated image alt text and removed UI links.
for value in ['10,015','8,215/573/1,227','68/917','72/423','0.6629','0.7853','78/118','19.77','0.660','214','28792059','28889071']:
 assert value in s,value
(out/'HU-MCD_Medical_Transfer_Meeting_Loop.md').write_text(s)
for name in figures:shutil.copy2(source/name,out/name)
(out/'README.txt').write_text('使用方法 / Usage\n\n1. 将 HU-MCD_Medical_Transfer_Meeting_Loop.md 的正文复制到 Loop。\n2. 在两个 Insert Figure 占位处分别插入 concept-summary.png 和 same-case.png，然后删掉占位提示。\n3. 保留图下英文说明。不要把本地图片路径当作其他成员可访问的链接。\n\nThe Markdown contains the English manuscript and result tables. Images are supplied separately; insert them at the marked positions. The two PNGs retain their original resolution.\n\nThis is a local export only. Nothing has been uploaded to Loop or published online.\n')
manifest={'source_markdown':str((source/'meeting-report.md').relative_to(R)),'source_sha256':hashlib.sha256((source/'meeting-report.md').read_bytes()).hexdigest(),'scientific_content_changed':False,'remote_uploads':0,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='export.json'}}
(out/'export.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=out.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(out.iterdir()):
  if p.is_file():z.write(p,p.name)
print(json.dumps({'markdown':str(out/'HU-MCD_Medical_Transfer_Meeting_Loop.md'),'bundle':str(archive),'figures':len(figures),'bytes':archive.stat().st_size}))
