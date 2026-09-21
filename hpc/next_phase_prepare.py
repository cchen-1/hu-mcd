"""Slurm-only, hash-bound preparation; no model execution or data substitution.

Explicit plan approval is checked by the submission coordinator. This worker
records A and B independently so a missing A cache does not erase B evidence.
"""
import csv,hashlib,json,os,shutil,socket,subprocess,sys,time,traceback
from pathlib import Path

def sha(path,algorithm='sha256'):
 h=hashlib.new(algorithm)
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return h.hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def main():
 if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().split('.')[0].startswith('bun'):
  raise RuntimeError('All remote file processing requires a Bunya Slurm allocation')
 plan=json.loads(Path(sys.argv[1]).read_text())
 if plan.get('authorization')!='APPROVED' or not isinstance(plan.get('commit'),str) or len(plan['commit'])!=40:
  raise RuntimeError('A frozen approved plan and full execution commit are required')
 out=Path(plan['output_parent'])/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
 shutil.copy2(sys.argv[1],out/'plan.json')
 result=dict(job_id=os.environ['SLURM_JOB_ID'],commit=plan['commit'],worker_sha256=sha(Path(__file__)),status='RUNNING',A=[],B={'status':'NOT_REQUESTED'},started=time.time())
 write(out/'progress.json',result)
 for group in plan['A_checks']:
  record={'class_name':group['class_name'],'status':'RUNNING','files':[]}
  try:
   for row in group['files']:
    p=Path(row['path']);actual=sha(p);record['files'].append(dict(path=str(p),sha256=actual,bytes=p.stat().st_size));
    if actual!=row['sha256']:raise ValueError('Cache identity mismatch: '+str(p))
   record['status']='PASS'
  except (OSError,ValueError) as e:
   record.update(status='FAILED',error=str(e),traceback=traceback.format_exc())
  result['A'].append(record);write(out/'progress.json',result)
 # Failure above is explicit and blocks only its class; no silent fallback.
 if plan.get('B_download'):
  record=result['B'];record['status']='RUNNING';write(out/'progress.json',result)
  try:
   b=plan['B_download'];root=out/'dataset'
   for name in ['raw','provenance','manifests','prepared']:(root/name).mkdir(parents=True)
   for source,target,expected in b['support_files']:
    src=Path(source)
    if sha(src)!=expected:raise ValueError('Metadata/provenance identity mismatch: '+source)
    shutil.copy2(src,root/target)
   name='dermamnist_corrected_224.npz';old=Path(b['previous_root'])/'raw'/name;dest=root/'raw'/name
   if old.exists():
    if sha(old,'md5')!=b['md5']:raise ValueError('Existing exact-name archive has unexpected MD5; no replacement')
    shutil.copy2(old,dest);record['download']='REUSED_VERIFIED_EXISTING_FILE'
   else:
    cmd=['curl','--location','--fail','--show-error','--silent','--retry','0','--connect-timeout','20','--max-time','6600','--max-filesize',str(b['bytes']),'--output',str(dest)+'.part',b['url']]
    completed=subprocess.run(cmd,capture_output=True,text=True,check=False)
    (out/'download.stderr.txt').write_text(completed.stderr);record['curl_exit']=completed.returncode
    if completed.returncode:raise RuntimeError('Exact-release download failed; partial retained; no retry')
    if sha(Path(str(dest)+'.part'),'md5')!=b['md5']:raise ValueError('Downloaded release MD5 mismatch; not promoted')
    Path(str(dest)+'.part').rename(dest);record['download']='EXACT_RELEASE_CHECKSUM_PASS'
   if dest.stat().st_size!=b['bytes']:raise ValueError('Release byte count mismatch')
   record['archive_sha256']=sha(dest);write(out/'progress.json',result)
   audit=Path(__file__).with_name('prepare_dermamnist.py')
   if sha(audit)!=b['auditor_sha256']:raise ValueError('Auditor identity mismatch')
   r=subprocess.run([sys.executable,str(audit),str(root)],capture_output=True,text=True)
   (out/'array-audit.stdout.txt').write_text(r.stdout);(out/'array-audit.stderr.txt').write_text(r.stderr)
   if r.returncode:raise RuntimeError('Exact-release array/split audit failed; see saved diagnostics')
   metadata=list(csv.DictReader((root/'raw/DermaMNIST-C.csv').open()));by_id={x['image_id']:x for x in metadata}
   # Author-confirmed duplicate pairs are checked separately from pixel hashes.
   pairfile=root/'provenance/author_confirmed_duplicates.csv';pairs=list(csv.DictReader(pairfile.open()));record['duplicate_pair_columns']=list(pairs[0]) if pairs else []
   # The exact schema is part of the frozen plan; unknown columns fail explicitly.
   left,right=b['duplicate_id_columns'];crossings=[]
   for pair in pairs:
    a,c=Path(pair[left]).stem,Path(pair[right]).stem
    if a not in by_id or c not in by_id:raise ValueError('Author duplicate pair outside metadata identity set')
    if by_id[a]['split']!=by_id[c]['split']:crossings.append({'left':a,'right':c,'left_split':by_id[a]['split'],'right_split':by_id[c]['split']})
   record['known_duplicate_crossings']=crossings
   if crossings:raise ValueError('Known cross-split duplicates: no experiment approval')
   record.update(status='PASS_PENDING_LOCAL_ACCEPTANCE',dataset_root=str(root),patient_independence='UNVERIFIED: no patient ID',pixel_identity_scope='Published MD5 + author row-order/labels; no independent original JPEG comparison')
  except Exception as e:
   record.update(status='FAILED',error=str(e),traceback=traceback.format_exc())
  write(out/'progress.json',result)
 result['status']='PASS_PENDING_LOCAL_ACCEPTANCE' if all(x['status']=='PASS' for x in result['A']) and result['B']['status'] in ['NOT_REQUESTED','PASS_PENDING_LOCAL_ACCEPTANCE'] else 'FAILED_OR_PARTIAL'
 result['elapsed_seconds']=time.time()-result['started'];write(out/'result.json',result);write(out/'progress.json',result)
 manifest=[]
 for f in out.rglob('*'):
  if f.is_file():manifest.append({'path':str(f.relative_to(out)),'sha256':sha(f),'bytes':f.stat().st_size})
 write(out/'artifacts.json',{'job_id':result['job_id'],'commit':plan['commit'],'status':result['status'],'files':manifest})
 print(json.dumps({'output':str(out),'status':result['status']}),flush=True)
 if result['status']=='FAILED_OR_PARTIAL':sys.exit(1)
if __name__=='__main__':main()
