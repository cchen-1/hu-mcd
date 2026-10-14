"""Publish validated local report, preserving the immediately preceding version."""
from pathlib import Path
import json,hashlib,csv,os,shutil,subprocess
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[2];O=R/'artifacts/report-redesign-20260923';S=O/'staged-report';LIVE=R/'artifacts/bunya/ten-class-review/report'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
links=json.loads((S/'hub/data/link-validation.json').read_text());assert not links['broken'] and not links['outside_bundle']
qa=json.loads((O/'browser-qa.json').read_text());assert not qa['errors']
assert all(not r.get('overflow') and not r.get('broken') and len(r.get('cjk',[]))<=1 for r in qa['checks'])
assert any(r.get('reviewSchema')=='PASS' for r in qa['checks'])
for entry in json.loads((S/'hub/data/source-manifest.json').read_text()):assert sha(Path(entry['path']))==entry['sha256'],entry['path']
protected=json.loads((O/'preserved-before.json').read_text())
for f,h in protected.items():
 if f=='hpc/check_job.sh' or f.startswith('outputs/'):assert sha(R/f)==h,f
for f in ['browser-qa.json','preservation-after.json','concurrent-report-additions.json']:shutil.copy2(O/f,S/'hub/data'/f)
for name in ['FILES.sha256.csv','renderer_manifest.json']:
 if (S/name).exists():shutil.copy2(S/name,S/'hub/data'/('previous-'+name))
code={}
for p in [R/'scripts/build_research_hub.py',*sorted((R/'scripts/research_hub').glob('*.py')),*sorted((R/'scripts/research_hub').glob('*.js')),*sorted((R/'scripts/research_hub').glob('*.css'))]:
 code[str(p.relative_to(R))]=sha(p);dst=S/'generation/research-hub'/p.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
now=datetime.now(timezone.utc).isoformat();manifest={'renderer':'research-hub-20260923','generated_at_utc':now,'report_sources':code,'workspace_head_at_publication':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'report_edits_not_committed':True,'scientific_execution_versions':'Per-job accepted source records, not workspace HEAD','previous_report_preserved':True,'private':True,'new_predictions':0,'human_storage_schema':1,'dataset_id':'humcd-fixed400-50-seed43-audit28222205','AI_entries':157,'new_AI_entries':99,'nine_class_baseline_stages_submitted':63,'new_baseline_results_accepted':0}
baseline_summary=R/'artifacts/bunya/nine-class-baselines-20260923/final-20260924/analysis/summary.json'
if baseline_summary.exists():
 baseline=json.loads(baseline_summary.read_text());assert baseline['status']=='PASS'
 manifest.update(new_baseline_results_accepted=baseline['accepted_stages'],nine_class_baseline_stages_accepted=baseline['accepted_stages'],all_method_direction_class_cells=baseline['direction_method_class_cells'],baseline_summary_sha256=sha(baseline_summary))
(S/'renderer_manifest.json').write_text(json.dumps(manifest,indent=2))
(S/'README.txt').write_text('Private offline research report\nOpen index.html; use the global language button. No network or build is needed for reading.\nAnalytical figures use English; explanatory HTML switches language. Raw evidence fields retain source language.\nHuman-review keys and schema1 are unchanged; export JSON for durable backups.\nGeneration instructions: hub/data/REPORT_REVIEW_20260923.zh-en.md\nReport source hashes: renderer_manifest.json\nOriginal scientific evidence and all failed histories are retained.\nNine-class MCD/ACE stages are submitted, NOT accepted results.\n')
if baseline_summary.exists():
 p=S/'README.txt';p.write_text(p.read_text().replace('Nine-class MCD/ACE stages are submitted, NOT accepted results.','Nine-class MCD/ACE stages are accepted; ten-class four-method bidirectional results are included.'))
with (S/'FILES.sha256.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['path','sha256','bytes'])
 for p in sorted(S.rglob('*')):
  if p.is_file() and p.name!='FILES.sha256.csv':w.writerow([p.relative_to(S).as_posix(),sha(p),p.stat().st_size])
backup=O/('live-report-before-publish-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));old_index_sha=sha(LIVE/'index.html')
os.replace(LIVE,backup)
try:os.replace(S,LIVE)
except Exception:os.replace(backup,LIVE);raise
receipt={'published_at_utc':now,'report':str(LIVE),'backup':str(backup),'old_index_sha256':old_index_sha,'new_index_sha256':sha(LIVE/'index.html'),'link_validation':links['local_references'],'browser_checks':len(qa['checks']),'new_experiments_by_report_task':0}
(O/'publication.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
