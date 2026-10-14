"""Update stable issue entries inside the user's existing Implementation Notes."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/bunya/medical-external-results-20260923'
def main():
    path=Path('/mnt/c/Users/uqcche38/github/obsidian-vault/PhD/Literature/HU-MCD.md');original=path.read_text();text=original;start=text.index('## 6. Implementation Notes')
    if not (OUT/'obsidian-before.md').exists():(OUT/'obsidian-before.md').write_text(original)
    records=json.loads((OUT/'issue-updates.json').read_text())
    for r in records:
        id=r['id'];marker='<!-- BATCH-20260922-'+id+':BEGIN -->';stop='<!-- BATCH-20260922-'+id+':END -->'
        body=marker+'\n\n**2026-09-23 update — '+id+' ('+r['stage']+')**\n\n'
        body+='- **Observed behaviour / Evidence:** '+r['observed_behaviour']+' Evidence: `'+r['evidence_path']+'`. Executioncommit `dcc61934973aa9054a4e689ccf2334652447c575`; `hpc/medical_external.py:100–177`, `utils/utils_general.py:49–98` where applicable. Jobs: '+(', '.join(r['jobs']) or 'local verification only')+'.\n'
        body+='- **Potential impact:** '+r['potential_impact']+'\n- **Possible explanation:** '+r['possible_explanation']+'\n- **Action / Decision / Status:** '+r['handling']+' **'+r['status']+'**. Earlier dated job/issue evidence remains in the append-only project ledger.\n\n'+stop
        if marker in text:text=re.sub(re.escape(marker)+'.*?'+re.escape(stop),lambda m:body,text,flags=re.S)
        else:
            pos=text.index('<!-- MED26-HRSM-RATIONALE:END -->');text=text[:pos]+'\n'+body+'\n\n'+text[pos:]
    marker='<!-- BATCH-20260922-CURRENT:BEGIN -->';stop='<!-- BATCH-20260922-CURRENT:END -->'
    current=marker+'''\n**Current accepted state (entry date2026-09-23 Brisbane):** E224deployment28796207 andexternal28796209COMPLETED0:0;524producerfiles verified and101cases accepted. Fixedepoch10 classifier:MEL29/101;296regions,37nonempty-mask zeros,C00194/259validregions. One frozen32-D concept, no external fitting; all12C001+12complement descriptive contrasts independently reproduced. This is engineering acceptance, not medical certification or semantic/localization validation. SourceD1 remains400/70accepted; A remains6complete/4partialfailed, with no retry. No new experiments or automaticclassifieroptimization this turn. Earlier pending/unapproved statements below are historical, superseded only by explicit scoped user approvals. Evidence: `docs/MEDICAL_EXTERNAL_RESULTS_20260923.zh-en.md`, `artifacts/bunya/medical-external-results-20260923/`; same private report `artifacts/bunya/ten-class-review/report/index.html`. All external concept reviews remainUNREVIEWED. The assistant's inspection of four fixed external examples is provisional/nonclinical; it does not modify the user's earlier nine-case review or resolve overlap/identity questions.\n'''+stop
    text=re.sub(re.escape(marker)+'.*?'+re.escape(stop),lambda m:current,text,flags=re.S)
    assert text[:start]==original[:start]
    path.write_text(text)
    receipt=dict(status='UPDATED_EXISTING_IMPLEMENTATION_NOTES',path=str(path),old_sha256=hashlib.sha256(original.encode()).hexdigest(),new_sha256=hashlib.sha256(text.encode()).hexdigest(),issue_ids=[r['id'] for r in records],prefix_preserved=True,stable_issue_entries_updated=True,backup=str(OUT/'obsidian-before.md'))
    (OUT/'obsidian-sync.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Updated9existing/stableissue entries andcurrentstate; unrelatednotes preserved')
if __name__=='__main__':main()
