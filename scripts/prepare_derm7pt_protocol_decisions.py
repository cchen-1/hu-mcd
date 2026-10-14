#!/usr/bin/env python3
"""Metadata-only decision support. No sampling, model, image or remote computation."""
import csv,json,hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'artifacts/bunya/next-phase-20260921';D=O/'medical-decisions'
C=O/'derm7pt-resumption/collections/28734478/20260921T051458.316863Z'
GROUPS={
'pigment_network':{'absent':'ABS','typical':'TYP','atypical':'ATP'},
'blue_whitish_veil':{'absent':'ABS','present':'PRS'},
'vascular_structures':{'absent':'ABS','arborizing':'REG','comma':'REG','hairpin':'REG','within regression':'REG','wreath':'REG','dotted':'IR','linear irregular':'IR'},
'pigmentation':{'absent':'ABS','diffuse regular':'REG','localized regular':'REG','diffuse irregular':'IR','localized irregular':'IR'},
'streaks':{'absent':'ABS','regular':'REG','irregular':'IR'},
'dots_and_globules':{'absent':'ABS','regular':'REG','irregular':'IR'},
'regression_structures':{'absent':'ABS','blue areas':'PRS','white areas':'PRS','combinations':'PRS'}}

def dump(name,obj): (D/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def table(name,rows):
 with (D/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 D.mkdir(exist_ok=True);rows=list(csv.DictReader((C/'cases.csv').open()));summary=[];raw=[];contrasts=[]
 for scope,split,inc in [('test_primary_melanoma','test',False),('test_author_MEL','test',True),('exploratory_melanoma_all_splits',None,False),('development_melanoma_train_valid','development',False)]:
  selected=[r for r in rows if r['diagnosis'].startswith('melanoma') and (inc or r['diagnosis']!='melanoma metastasis') and (split is None or (r['official_split'] in ('train','valid') if split=='development' else r['official_split']==split))]
  summary.append(dict(candidate=scope,status='PROPOSED_NOT_FROZEN',cases=len(selected),diagnoses=json.dumps(dict(Counter(r['diagnosis'] for r in selected))),same_derm_clinic_reference=sum(r['derm']==r['clinic'] for r in selected),missing_case_id=sum(r['case_id']=='' for r in selected),affected_alias_case_nums=';'.join(r['case_num'] for r in selected if r['derm']==r['clinic'])))
  for sign,mapping in GROUPS.items():
   counts=Counter(r[sign] for r in selected);assert set(counts)<=set(mapping)
   g=Counter()
   for label,n in counts.items():g[mapping[label]]+=n
   for label in mapping:raw.append(dict(candidate=scope,sign=sign,raw_label=label,proposed_group=mapping[label],cases=counts[label],denominator=len(selected)))
   for group in dict.fromkeys(mapping.values()):
    if group=='ABS':continue
    pos=g[group];neg=g['ABS'];contrasts.append(dict(candidate=scope,sign=sign,group=group,reference='ABS',group_n=pos,reference_n=neg,contrast_n=pos+neg,cohort_n=len(selected),other_category_not_in_this_contrast=len(selected)-pos-neg,defined=bool(pos and neg),small_group_flag=min(pos,neg)<10,note='Proposal only: grouped category vs absent, not all-rest binarization; no ordinal correlation'))
 table('candidate-scopes.csv',summary);table('raw-sign-counts.csv',raw);table('proposed-contrasts.csv',contrasts)
 dump('proposed-label-groups.json',dict(status='PROPOSED_NOT_APPROVED',scope='Original categorical fields retained; grouped copy only if approved',mapping=GROUPS,author_code_commit='ce436877573c0a53cfa0d224bda20d9479b795aa',author_code_locations=['derm7pt/dataset.py:489-508','base categorical definitions'],melanoma_primary_proposal=['melanoma','melanoma (in situ)','melanoma (less than 0.76 mm)','melanoma (0.76 to 1.5 mm)','melanoma (more than 1.5 mm)'],metastasis='Explicit proposed exclusion from primary, diverges from author MEL grouping; no actual records excluded'))
 dump('preparation-receipt.json',dict(status='METADATA_DECISION_SUPPORT_COMPLETE',created_at=datetime.now(timezone.utc).isoformat(),cases_sha256=hashlib.sha256((C/'cases.csv').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),no_model_or_sampling=True,no_case_id_reaudit=True,no_cohort_frozen=True))
 print(json.dumps(summary,indent=2));print(json.dumps([x for x in contrasts if x['candidate']=='test_primary_melanoma'],indent=2))
if __name__=='__main__':main()
