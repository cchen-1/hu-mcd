"""Planning-only overlay in the existing private report; no result placeholders."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'artifacts/bunya/next-phase-20260921'
W=ROOT/'artifacts/bunya/evaluation-baselines-20260910'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 status=json.loads((O/'status.json').read_text());p=W/'workstream_status.json';data=json.loads(p.read_text())
 if not (O/'workstream_status.before.json').exists():(O/'workstream_status.before.json').write_bytes(p.read_bytes())
 summary=data['research_summary'];summary['paragraphs']=[x for x in summary['paragraphs'] if not x.get('next_phase_20260921')]
 submitted=status.get('submitted_jobs',[])
 text={'zh':'下一阶段（2026-09-21）：A最终输入掩码敏感性、B医学迁移。下方既有结果仍为已验收；新工作线的准备不代表模型运行。A已核对500图/5687区域（413原零特征，5274原有效），候选5条件28435条记录，几何预检查r2有6空mask；没有新模型推理。B精确Zenodo档案仍未验收，melanoma元数据训练1021图/533病灶、测试70图/61病灶；Derm7pt实际标签计数未知。协议／预算等待用户明确选择，不将推荐当批准。',
 'en':'Next phase (2026-09-21):A final-input mask sensitivity; B medical transfer. Existing results below remain accepted; new-workstream preparation is not model execution. A verified500 images/5687regions (413original zeros,5274valid), candidate5conditions28435records, geometry-only r2 preview6empty masks; no new inference. B exact Zenodo archive remains unaccepted; melanoma metadata has1021training images/533lesions and70test images/61lesions. Actual Derm7pt sign counts are unknown. Protocol/resource recommendations await explicit user decisions.'}
 if status['authorizations'].get('A'):
  text={'zh':'下一阶段：A五条件协议和资源上限已批准；10类新执行路径的缓存得分／分配检查通过，GPU不变区域对照仍是逐类运行门槛。B已批准病灶优先400病灶各1图，名单须待图像、模型及预算确认后冻结。Derm7pt已提供RDM原始ZIP，Slurm复制到scratch并审计；医学训练未批准。用户要求先澄清已有DermaMNIST corrected版本身份，暂不下载数据。已有全部复现结果保持已验收状态。',
        'en':'Next phase: A five conditions and resource caps are approved. The new score/assignment path passed all ten saved-cache checks; GPU unchanged-mask controls remain mandatory per-class gates. B lesion-first sampling (400 distinct lesions, one image each) is approved, with list freezing conditional on image/model/budget acceptance. Derm7pt RDM ZIP is available for Slurm staging/audit on scratch. Medical training is not approved. DermaMNIST downloads are on hold while the existing corrected-release identity is clarified. All earlier accepted replication results remain unchanged.'}
 text['next_phase_20260921']=True;summary['paragraphs'].insert(0,text)
 summary['paragraphs'].insert(1,{'next_phase_20260921':True,'zh':'新工作线实际状态：A='+status['A']+'；B='+status['B']+'；本批已提交作业='+str(submitted)+'。提交后仅一次调度核对，等待用户通知完成；不自动监控或重提。','en':'Actual new-workstream state:A='+status['A']+'; B='+status['B']+'; submitted jobs='+str(submitted)+'. One post-submission scheduler check; wait for user completion notification, no automatic monitor/retry.'})
 summary['artifacts']=[a for a in summary['artifacts'] if 'next-phase-20260921' not in a['path']]
 for name,kind,label in [('A-workload.csv','table','A候选工作量与已有阶段耗时；未执行 / A proposed workload and prior measured stages; not executed'),('B-source-metadata-counts.csv','table','B源域元数据图片/病灶计数；不是图像验收 / B source metadata image/lesion counts, not image acceptance'),('current_ten_class_overview.csv','evidence','当前十类发现＋评价状态；历史发现表仍保留 / Current joined discovery/evaluation table; historic table preserved')]:
  f=O/name;summary['artifacts'].insert(0,dict(path=str(f),sha256=sha(f),kind=kind,caption=label))
 docs={str(Path(d['path']).resolve()):d for d in data['documents']}
 files=[(ROOT/'docs/NEXT_PHASE_READINESS_20260921.zh-en.md','下一阶段实际准备、待批协议与预算 / Next-phase readiness, pending decisions and budgets'),(O/'inventory/A-inventory.json','A全部区域身份及候选几何计数 / A complete region identity and candidate geometry counts'),(O/'status.json','A/B当前准备与提交状态 / Current A/B preparation/submission state'),(O/'B-sampling-feasibility.json','B图片/病灶抽样差异 / B image-versus-lesion sampling'),(O/'sources/release-download.json','精确发布获取证据 / Exact-release retrieval evidence')]
 for f,label in files:docs[str(f)]=dict(path=str(f),label=label,sha256=sha(f))
 for d in docs.values():d['sha256']=sha(Path(d['path']))
 data['documents']=list(docs.values());data['updated_at']=status['updated_at'];data['preparation_workstreams']=status
 data['anomalies']=[json.loads(line) for line in (ROOT/'docs/ANOMALIES.jsonl').read_text().splitlines() if line.strip()]
 p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 print('Updated existing report data: preparation only; accepted collections unchanged')
if __name__=='__main__':main()
