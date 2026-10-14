"""Update current local status while preserving original submission/failure history."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/bunya/medical-external-results-20260923';now=datetime.now(timezone.utc).isoformat()
J=lambda p:json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
    a=J(OUT/'acceptance.json');assert a['status']=='ENGINEERING_ACCEPTED_RESEARCH_LIMITS_DISCLOSED'
    evidence='artifacts/bunya/medical-external-results-20260923/acceptance.json';issues=[]
    def issue(id,stage,fact,impact,explanation,handling,status='DISCLOSED',evidence_path=evidence,jobs=['28796209']):
        issues.append(dict(id=id,recorded_at_utc=now,stage=stage,jobs=jobs,observed_behaviour=fact,evidence_path=evidence_path,potential_impact=impact,possible_explanation=explanation,handling=handling,status=status))
    issue('MED26-021','E224 execution and acceptance','Deployment28796207 andexternal28796209COMPLETED0:0,6s/196s;524files hash-verified,101originals/224workinginputs and296mask-feature rows accepted. All101 originals nonsquare. Fullcommitdcc61934973aa9054a4e689ccf2334652447c575.','Approved aspect-ratio warp remains a scientific restriction, despite exact execution.','No new causal explanation inferred.','User completion notice triggered collection; unchangedR101/S, no fitting/retraining/new prediction/retry. Prior submission-only snapshot remains history.','ENGINEERING_ACCEPTED_RESEARCH_LIMITS_DISCLOSED',jobs=['28796207','28796209'])
    issue('MED26-018','external zero-feature regions','37/296externalregions have exact-zero2048features with nonempty effective masks; source68/917 and14/157 remain. No entirely invalid external image.','Invalid regions excluded from valid assignments/coverage; never forcedC001. Region validity is not a semantic-quality metric.','Masked-network geometry or activations are possibilities, not causally isolated.','Retain allrawrows, masks and invalid−1; no removal or new forward pass.')
    issue('MED26-023','frozen classifier and concept transfer','MEL29/101externally versus50/70sourceheldout; predictions2AKIEC,2BCC,22BKL,0DF,29MEL,46NV,0VASC. C00194/259externalvalidregions versus82/143sourceheldout. Accepted weights/bases/inputs unchanged.','Performance gap is observed on different cohorts; cannot certify clinical transfer or attribute failure to discovery.','Domain, approved geometry, cohort composition and classifier limitations are plausible, unseparated causes.','Keep fixed classifier and allcases; reportactualdenominators. No automatic optimization or follow-up experiment.')
    issue('MED26-024','R101 descriptive sign analysis','All12C001 plus12complement contrasts independently reproduced; no missingprocessedcases. Regularpigmentation1vs38; regularstreaks1vs41. Irregulardots/globules84vs12,AUC0.660218.','Sparse groups and twelve descriptive comparisons prevent a selected AUC from being treated as a validated semantic association. Unknownpatientidentity furtherlimitsindependence.','Coverage can reflect region size/content and mask behavior rather than a named sign; unverified explanation.','No pvalues/CI/signflip; retainall12directions and fullgroupdenominators. No semantic-equivalence/localization claim.')
    issue('MED26-008','external overlap limitation','Prior acceptedH214thresholdpairs across20R101cases remain unconfirmed; all101cases included. No new pair review supplied.','Cross-dataset independence is unknown; negative exact matches and nine-case aliasreview do not settle this.','Some perceptualsimilarities may be benign; pairidentity requires review.','ReusepriorHpairpages; do notexclude or inferpatients.','OPEN_MANUAL_REVIEW')
    issue('MED26-011','local cached acceptance replay','First local SAM-composition replay omitted Pythonseed43 and failed ordered-mask comparison atcase578. Released utils_general.py49-98 uses random integerregionIDs and sorteduniqueIDs. After settingseed43 andcompletefrozenimageorder,all101rawmask arrays matched exactly.','The failed localcheck did not establish a remote defect; no saved result changed.','Missing verifierRNG state explains this resolved localreplay mismatch.','Preservefirsttraceback;correctonlylocalverification. No Slurmjob retried.','RESOLVED',evidence_path='artifacts/bunya/medical-external-results-20260923/acceptance-unseeded-replay.failure.txt',jobs=[])
    issue('EV26-003','dependency warning','One np.bool8DeprecationWarning recorded by externalworker; no observed integrity failure.','Futuredependencycompatibility warning, not evidenceofchangedscientificoutputs.','Installedskimage usesdeprecatedNumPyalias.','Preserveoriginalwarning; no dependencychange.','DISCLOSED',evidence_path='artifacts/bunya/medical-external-results-20260923/collection/anomalies.jsonl')
    issue('MED26-025','Slurm allocation accounting','Deployment requestedcpus-per-task1, SlurmacctAllocCPUS2;6sexecution,1GiB request unchanged.','Actualallocation differs fromsubmittedrequest; discloseactualusage.','Schedulerallocationgranularity is plausible but not independently established.','No enlargedrequest or newjob; retainsubmissionscript andaccounting.','DISCLOSED',evidence_path='artifacts/bunya/medical-external-results-20260923/sacct.psv',jobs=['28796207'])
    issue('MED26-020','limited cross-domain visual inspection','Assistant inspected first2top andfirst2fixedrandomC001examples per domain in12-panel comparison;8source/heldout examples reused. Externaltopcase760S0000/case793S0002retainbroadimageareas; randomcase578S0001/case781S0001retainmoreboundedcentraldarkregions.','This limited nonclinical inspection cannot establish globalprototype/randommemberquality, expertsemantics orsegmentaccuracy.','Differentselectedregionextent is directly visible; medicalmeaning remains unknown.','All101casepages and296panels retained; noAIgroundtruthlabels. Externalhumanreview remainsUNREVIEWED.',evidence_path='artifacts/bunya/medical-external-results-20260923/bundle/comparison-selection.csv')
    dump(OUT/'issue-updates.json',issues)
    ledger=ROOT/'docs/ANOMALIES.jsonl';old=ledger.read_text();new=[]
    for r in issues:
        # Idempotent within this acceptance; prior same-ID events remain intact.
        if not any(json.loads(line).get('id')==r['id'] and json.loads(line).get('evidence_path')==r['evidence_path'] and json.loads(line).get('observed_behaviour')==r['observed_behaviour'] for line in old.splitlines() if line.strip()):new.append(r)
    with ledger.open('a') as f:
        for r in new:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    h=ROOT/'artifacts/bunya/medical-external-20260922/handover.json';hand=J(h)
    if not (OUT/'submission-handover.before.json').exists():dump(OUT/'submission-handover.before.json',hand)
    hand.update(current_status=a['status'],current_evidence=evidence,external_results_not_collected=False,updated_at_utc=now,terminal_states={'28796207':'COMPLETED_0:0','28796209':'COMPLETED_0:0'},current_note='Original deployment/external objects preserve actual submission receipts; final status/acceptance is recorded separately. No resubmission.')
    dump(h,hand)
    p=ROOT/'artifacts/bunya/next-phase-20260921/status.json';state=J(p);state['updated_at']=now;state['medical_external_20260922']=hand;state['medical_external_results_20260923']=dict(status=a['status'],evidence=evidence,no_new_jobs=True);dump(p,state)
    p=ROOT/'artifacts/bunya/next-phase-20260922/authorization-ledger.json';state=J(p)
    if not (OUT/'authorization-ledger.before.json').exists():dump(OUT/'authorization-ledger.before.json',state)
    for item in state['items']:
        if item['id']=='MED26-E224-20260922':
            if item['status']!=a['status']:item.setdefault('previous_statuses',[]).append(item['status'])
            item.update(status=a['status'],current_evidence=evidence)
    state['latest_user_scope']='Prior explicitly approved D0/D1 and E224/R101/S have executed and are now engineering accepted with research limits. No new experiment/retry/optimization authorized by completion notification.'
    state['actual_medical_execution']['historical_snapshot']=True
    state['current_execution_20260923']=dict(H_M='ACCEPTED_PRIOR',D0_D1='ACCEPTED_PRIOR',E224=a['status'],A='6_COMPLETE_4_PARTIAL_FAILED_NO_RETRY',evidence=evidence)
    state['updated_at_utc']=now;dump(p,state)
    prefix='''## 2026-09-23 当前验收 / Current acceptance

外部部署28796207及推理28796209均COMPLETED0:0（6秒/3分16秒），524文件收集校验通过；101例全部完成，29/101预测MEL，296片段中37零特征，C00194/259有效片段。固定输入/权重/概念基与24项描述性统计验收通过；医学适用性、语义、空间定位与患者独立性未获证明。A仍6完整/4部分失败，未补跑。当前无待收集的E224作业，没有新增作业/重训/重提。

External jobs28796207/28796209 completed0:0(6s/196s);524files verified,all101cases processed,MEL29/101,37zero-feature regions,C00194/259validregions. Frozen input/model/basis and24descriptive comparisons accepted; clinical utility,semantic/localization validity andpatientindependence remain unestablished. Aremains6complete/4partialfailed withoutretry. No E224 jobawaitingcollection andnonewjobthisturn.

当前证据与双语结果：[外部验收报告](MEDICAL_EXTERNAL_RESULTS_20260923.zh-en.md)、`artifacts/bunya/medical-external-results-20260923/acceptance.json`。本节替代下方历史“外部已提交未收集”状态，不增加任何新实验授权；保留所有旧失败与后续成功历史。

'''
    for name in ['MAIN_CHAT_HANDOVER.md','NEXT_PHASE_CURRENT_DECISIONS_20260922.zh-en.md']:
        p=ROOT/'docs'/name;original=p.read_text()
        if not original.startswith('## 2026-09-23 当前验收'):p.write_text(prefix+original)
    p=ROOT/'docs/NEXT_PHASE_MAIN_CHAT_PLAN_20260921.zh.md';original=p.read_text()
    if not original.startswith('> **2026-09-23'):
        p.write_text('> **2026-09-23当前结果入口：** 已明确批准的E224/R101/S执行现已验收，见[外部结果](MEDICAL_EXTERNAL_RESULTS_20260923.zh-en.md)与[集中决策表](NEXT_PHASE_CURRENT_DECISIONS_20260922.zh-en.md)。本轮仅收集/缓存分析，无新实验授权；原计划通用规则与历史仍保留。A四类失败未重提。\n\n'+original)
    dump(OUT/'state-update.json',dict(updated_at=now,issues=[r['id'] for r in issues],evidence=evidence,execution_commit=a['commit'],new_jobs=[]))
    print('Updated current state and',len(new),'issue events')
if __name__=='__main__':main()
