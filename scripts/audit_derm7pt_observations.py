#!/usr/bin/env python3
"""Private, local evidence audit. Does not call existing Derm7pt parser or modify data.
Run with humcd-upstream Python. Original ZIP and prior Slurm receipts are read only.
"""
import csv, hashlib, html, io, json, os, subprocess, zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'artifacts/bunya/next-phase-20260921'
D=O/'derm7pt-review'
C=O/'derm7pt-resumption/collections/28734478/20260921T051458.316863Z'
ZIP=Path('/mnt/c/Users/uqcche38/Downloads/release_v0.zip')
EXEC='83972e99763ef6ecaebba3cbad6c0bf14e62e63a'
AUTHOR='ce436877573c0a53cfa0d224bda20d9479b795aa'
sha=lambda b:hashlib.sha256(b).hexdigest()
load=lambda p:json.loads(p.read_text())
def write(name,obj):
 p=D/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def table(name,rows,fields=None):
 with (D/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 D.mkdir(exist_ok=True);(D/'images').mkdir(exist_ok=True);(D/'figures').mkdir(exist_ok=True);(D/'raw').mkdir(exist_ok=True)
 source=subprocess.check_output(['git','show',EXEC+':hpc/derm7pt_audit.py'],cwd=ROOT);(D/'sources/prior-slurm-audit.py').write_bytes(source)
 assert source==(ROOT/'hpc/derm7pt_audit.py').read_bytes()
 original=load(O/'derm7pt-resumption/acceptance.json');b=ZIP.read_bytes();assert sha(b)==original['zip_sha256']
 a=load(C/'derm7pt_audit.json');remote_meta=load(C/'metadata_sha256.json')
 z=zipfile.ZipFile(io.BytesIO(b));raw=z.read('release_v0/meta/meta.csv');assert sha(raw)==remote_meta['meta/meta.csv']
 (D/'raw/meta.csv').write_bytes(raw);(D/'raw/README.txt').write_bytes(z.read('release_v0/README.txt'))
 # Independent reader A: physical UTF-8 lines and literal delimiters. Guard the
 # actual format so quotes/multiline rows cannot silently invalidate this reader.
 text=raw.decode('utf-8');assert '"' not in text
 lines=text.splitlines();header=lines[0].split(',');assert len(header)==len(set(header))==19
 cells=[line.split(',') for line in lines[1:]];assert all(len(r)==19 for r in cells)
 rows=[dict(zip(header,r)) for r in cells]
 # Independent reader B: pandas C CSV engine, strings only, NA inference OFF.
 frame=pd.read_csv(io.BytesIO(raw),engine='c',dtype=str,keep_default_na=False,na_filter=False)
 assert frame.columns.tolist()==header and frame.values.tolist()==cells
 # Compare all original fields, not only suspect fields, to the Slurm export.
 prior=list(csv.DictReader((C/'cases.csv').open()));assert len(prior)==len(rows)==1011
 for i,row in enumerate(rows):
  assert int(prior[i]['row_index'])==i
  for k,v in row.items():assert prior[i][k]==v,(i,k)
 splits={}
 for split in ('train','valid','test'):
  name=f'meta/{split}_indexes.csv';sb=z.read('release_v0/'+name);assert sha(sb)==remote_meta[name]
  (D/'raw'/Path(name).name).write_bytes(sb)
  for s in sb.decode().splitlines()[1:]:
   i=int(s);assert i not in splits;splits[i]=split
 assert set(splits)==set(range(len(rows)))
 inventory=load(C/'image_inventory.json');by={(r['row_index'],r['role']):r for r in inventory};assert len(by)==2022
 for i,row in enumerate(rows):
  assert prior[i]['official_split']==splits[i]
  for role in ('derm','clinic'):
   rec=by[i,role];assert rec['relative_path']==row[role] and rec['case_num']==row['case_num'] and rec['case_id']==row['case_id'] and rec['diagnosis']==row['diagnosis'] and rec['split']==splits[i]
 # Identify ALL affected cases without using prior duplicate grouping.
 equal=[i for i,r in enumerate(rows) if r['derm']==r['clinic']]
 missing=[i for i,r in enumerate(rows) if r['case_id']==''];nonmissing=[i for i,r in enumerate(rows) if r['case_id']!='']
 assert len(equal)==9 and len(missing)==984 and len(nonmissing)==27
 assert not any(r['case_id'] and not r['case_id'].strip() for r in rows)
 assert not any(r['case_id'].lower() in {'nan','null','none','na','n/a'} for r in rows)
 index=[];pairs=[]
 for i,row in enumerate(rows):
  rec=dict(csv_physical_line=i+2,data_row_one_based=i+1,row_index_zero_based=i,**row,official_split=splits[i],case_id_empty_exact=row['case_id']=='',derm_clinic_field_equal=i in equal,derm_resolved_path=a['release_root']+'/images/'+row['derm'],clinic_resolved_path=a['release_root']+'/images/'+row['clinic'])
  index.append(rec)
  if i not in equal:continue
  entry=dict(rec);entry.update(review_id='D7P-release_v0-case-'+row['case_num'],string_equal=True,resolved_path_equal=True,different_files_same_bytes=False,different_files_same_bytes_interpretation='NOT_APPLICABLE: same ZIP member / same resolved path',decoded_pixels_equal=True,review_status='UNREVIEWED',ai_image_type='NOT_ASSIGNED')
  rolebytes=[]
  for role in ('derm','clinic'):
   v=z.read('release_v0/images/'+row[role]);rolebytes.append(v);ir=by[i,role]
   assert sha(v)==ir['sha256']
   with Image.open(io.BytesIO(v)) as im:
    im.load();arr=np.asarray(im);ph=sha(str((im.mode,arr.shape,str(arr.dtype))).encode()+arr.tobytes())
    assert ph==ir['pixel_sha256'] and im.size==(ir['width'],ir['height'])
   entry[role+'_sha256']=sha(v);entry[role+'_pixel_sha256']=ph
  assert rolebytes[0]==rolebytes[1]
  filename='case-'+row['case_num']+Path(row['derm']).suffix
  (D/'images'/filename).write_bytes(rolebytes[0]);entry['display_image']='images/'+filename
  entry['width']=by[i,'derm']['width'];entry['height']=by[i,'derm']['height']
  # Documentary side-by-side figure, both original views displayed unchanged.
  fig,ax=plt.subplots(1,2,figsize=(12,6.4),dpi=220)
  im=Image.open(io.BytesIO(rolebytes[0]))
  for j,role in enumerate(('derm','clinic')):
   ax[j].imshow(im);ax[j].axis('off');ax[j].set_title(role+' field: '+row[role],fontsize=11)
  fig.suptitle('Derm7pt case '+row['case_num']+' | CSV line '+str(i+2)+' | '+splits[i]+'\nSAME SOURCE FILE - two roles, one image',fontsize=15)
  fig.text(.5,.035,'Source notes: '+row['notes']+'\nUnreviewed image type; no crop or preprocessing. For private review only.',ha='center',fontsize=9)
  fig.subplots_adjust(top=.82,bottom=.13,left=.025,right=.975,wspace=.04)
  figure='figures/case-'+row['case_num']+'-pair.png';fig.savefig(D/figure,dpi=220);plt.close(fig);entry['pair_figure']=figure;pairs.append(entry)
 table('all-records-index.csv',index);table('case-id-empty-index.csv',[index[i] for i in missing]);table('case-id-present-index.csv',[index[i] for i in nonmissing]);table('nine-pairs.csv',pairs)
 review=[dict(review_id=x['review_id'],case_num=x['case_num'],csv_physical_line=x['csv_physical_line'],derm_image_type='',clinic_image_type='',assessable='',obvious_input_anomaly='',evidence_and_example_id='',suggested_action='',reviewer='',reviewed_at='',status='UNREVIEWED') for x in pairs]
 # Never overwrite user-entered review records.
 if not (D/'manual-review.csv').exists():table('manual-review.csv',review)
 samples=[missing[:3],nonmissing[:3]]
 (D/'raw/raw-examples.txt').write_text('Original header (line 1):\n'+lines[0]+'\n\n'+ '\n\n'.join(('EXACT EMPTY' if j==0 else 'NONEMPTY')+'; first three by source order\n'+'\n'.join('CSV physical line '+str(i+2)+': '+lines[i+1] for i in ids) for j,ids in enumerate(samples))+'\n')
 (D/'raw/nine-original-lines.txt').write_text('Original header (line 1):\n'+lines[0]+'\n\n'+'\n\n'.join('CSV physical line '+str(i+2)+':\n'+lines[i+1] for i in equal)+'\n')
 write('nine-pairs.json',pairs)
 derm=[x for x in inventory if x['role']=='derm'];fieldcounts={k:dict(total=len(rows),exact_empty=sum(r[k]=='' for r in rows),unique_nonempty=len({r[k] for r in rows if r[k]!=''})) for k in header}
 scope=[]
 for split in ('all','train','valid','test'):
  for rule in ('all diagnoses','melanoma except metastasis','melanoma including metastasis'):
   ids=[i for i,r in enumerate(rows) if (split=='all' or splits[i]==split) and (rule=='all diagnoses' or (r['diagnosis'].startswith('melanoma') and (rule.endswith('including metastasis') or r['diagnosis']!='melanoma metastasis')))]
   scope.append(dict(scope_status='CONDITIONAL_CANDIDATE_NOT_FROZEN',split=split,diagnosis_rule=rule,cases=len(ids),same_role_path_cases=sum(i in equal for i in ids),case_id_empty=sum(i in missing for i in ids),case_id_present=sum(i in nonmissing for i in ids),affected_case_nums=';'.join(rows[i]['case_num'] for i in ids if i in equal)))
 table('scope-impact.csv',scope)
 check=dict(status='PASS_OBSERVATIONS_CONFIRMED_NOT_DATA_ERRORS',created_at=datetime.now(timezone.utc).isoformat(),archive='release_v0',archive_sha256=sha(b),archive_bytes=len(b),local_archive=str(ZIP),original_csv_zip_member='release_v0/meta/meta.csv',original_csv_sha256=sha(raw),actual_slurm_csv_path=a['release_root']+'/meta/meta.csv',rdm_recovery_zip=a['rdm_zip'],slurm_job='28734478',slurm_audit_commit=EXEC,slurm_code_locations=['hpc/derm7pt_audit.py:40-64','hpc/derm7pt_audit.py:89-98','hpc/derm7pt_audit.py:118-135','hpc/workstream_runtime.py:72-74'],independent_script=str(Path(__file__).relative_to(ROOT)),independent_script_sha256=sha(Path(__file__).read_bytes()),local_git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),independent_readers=['UTF8 literal line/comma split with no-quote and19-field guards','pandas C CSV engine dtype=str, na_filter=False, keep_default_na=False'],all_original_cells_equal_to_slurm=True,columns=header,field_counts=fieldcounts,records=1011,case_id_empty_exact=984,case_id_nonempty=27,case_id_distinct_nonempty=len({rows[i]['case_id'] for i in nonmissing}),whitespace_only_case_id=0,case_id_na_sentinel_count=0,same_field_string_cases=9,same_resolved_path_cases=9,different_path_equal_bytes_groups=0,same_file_pixel_match_cases=9,notes_missing_clinic_count=sum('Clinical missing' in rows[i]['notes'] for i in equal),notes_named_replacement_count=sum('Replaced clinic' in rows[i]['notes'] for i in equal),derm_records=len(derm),derm_unique_paths=len({x['relative_path'] for x in derm}),derm_unique_file_hashes=len({x['sha256'] for x in derm}),derm_unique_pixel_hashes=len({x['pixel_sha256'] for x in derm}),manual_review_status='UNREVIEWED',no_data_change=True,no_remote_job_submitted=True)
 assert check['derm_records']==check['derm_unique_paths']==check['derm_unique_file_hashes']==check['derm_unique_pixel_hashes']==1011
 # Verify no two distinct paths in the complete audit have matching file bytes.
 groups=defaultdict(set)
 for x in inventory:groups[x['sha256']].add(x['relative_path'])
 assert all(len(x)==1 for x in groups.values())
 write('verification.json',check)
 write('parser-reconciliation.json',dict(status='PASS',rows=1011,columns=19,all_raw_fields_compared=19209,selected_fields=['case_num','case_id','derm','clinic','notes'],empty_rule='literal zero-length field after comma tokenization; no whitespace stripping or NaN defaults',path_rule='release_root/images/original relative field, no fallback, no case folding, no lookup by case_id',original_code='csv.DictReader; enumerate(metadata rows); direct row[role]; not row[case_id] join; not strip/fillna/default substitution',slurm_checks='Reused hash-verified CSV/image inventory; all roles and labels mapped back to original rows',independent_readers=check['independent_readers']))
 print(json.dumps({k:check[k] for k in ['status','case_id_empty_exact','case_id_nonempty','derm_unique_paths','notes_missing_clinic_count','notes_named_replacement_count']},indent=2));print(json.dumps(scope,indent=2))

if __name__=='__main__':main()
