"""Scoped report repair; retain other pages and avoid overwriting hardlinks."""
import csv,hashlib,json,os,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
import matplotlib
matplotlib.use('Agg')
from research_hub.robustness_figure import build
R=Path(__file__).resolve().parents[1];report=R/'artifacts/bunya/ten-class-review/report'
out=R/'artifacts/bunya/robustness-completion-20260924';out.mkdir(exist_ok=True)
if (out/'acceptance.json').exists():
    from research_hub.robustness_completion import apply
    apply(report)
    raise SystemExit(0)
rows=list(csv.DictReader((R/'artifacts/bunya/batch-review-20260922/robustness-conditions.csv').open()))
names=[r['name'] for r in json.loads((R/'artifacts/bunya/ten-class-review/atlas_final.json').read_text())['classes']]
fig=build(rows,names);changed=[]
for ext in ['png','svg']:
    dest=report/'hub/figures'/('robustness.'+ext);backup=out/('robustness-before.'+ext)
    if not backup.exists():shutil.copy2(dest,backup)
    temp=out/('robustness.'+ext);fig.savefig(temp,dpi=240,facecolor='white')
    new=dest.with_name(dest.name+'.new');shutil.copy2(temp,new);os.replace(new,dest);changed.append(dest.relative_to(report).as_posix())
# Captions are explanatory; existing page/interpretation/review state is preserved.
p=report/'robustness.html';old=p.read_text();backup=out/'robustness-before.html'
if not backup.exists():backup.write_text(old)
new=old.replace('灰格是未验收条件，不是零。','每行是一个类别，每列是一种边界扰动；百分比表示原有效片段保留原概念分配的平均比例。灰格／—表示没有已验收结果，不是零。').replace('Grey cells are unaccepted conditions, not zero.','Rows are classes; columns are boundary changes. Percentages show how often valid baseline regions keep their original concept assignment, averaged across images. Grey / — means no accepted result, not zero.')
p.write_text(new);changed.append('robustness.html')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=report/'FILES.sha256.csv'
if manifest.exists():
    rr=list(csv.DictReader(manifest.open()))
    for r in rr:
        if r['path'] in changed:r.update(sha256=sha(report/r['path']),bytes=str((report/r['path']).stat().st_size))
    with manifest.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
(out/'chart-repair.json').write_text(json.dumps(dict(date=datetime.now(timezone.utc).isoformat(),changed=changed,source='batch-review-20260922/robustness-conditions.csv',figure_generator_sha256=sha(R/'scripts/research_hub/robustness_figure.py'),scientific_values_changed=False,missing_cells=12,backup=str(out)),indent=2))
print('Repaired only',changed)
