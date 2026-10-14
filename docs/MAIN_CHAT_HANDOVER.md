## 2026-09-24 — English medical meeting report

User-requested English meeting manuscript published as `artifacts/bunya/ten-class-review/report/medical-meeting.html`, linked from Medical transfer and both medical subpages. Covers rationale and source relationship, fixed classifier/discovery settings, four learned concepts, classifier limitations, same-case mask/assignment evidence, descriptive sign findings and limits of medical/ImageNet comparison. Editable companion and two high-resolution figures: `hub/medical-meeting/{meeting-report.md,concept-summary.png,same-case.png}`. All concepts retained; old route external examples and new route fitting examples explicitly distinguished. Prototypes/random selections unchanged; no new inference, fitting, sampling or jobs.

Persistent generator: `scripts/research_hub/medical_meeting.py`, called last from `scripts/build_research_hub.py`. Evidence IDs/mask hashes and accepted-receipt hashes accompany exports. Offline English rendering, images, mobile layout and30local references verified;22prior review files hash-preserved. Publication receipt: `artifacts/medical-meeting-20260924/publication.json`. Full-resolution figures visually checked. No new scientific claim or clinical label inferred.

## 2026-09-24 — Two-route compact medical image evidence / 两路线紧凑医学图像证据

Published within existing `medical-dermamnist.html` and `medical-ham-derm/index.html`, linked from Medical transfer. Ten high-resolution figures: four same-case process sheets (cases578/752 in each route) and six member sheets (one old learned + complement; three HAM learned + complement). No new jobs, predictions or display sampling. Same-case originals were already present in the accepted local E224 report, validated against both input bindings; no additional remote image transfer occurred. Pre-SAM images were pixel-replayed from these originals with the saved E224 bicubic / HAM aspect-preserving LANCZOS rules.

Old route member rows remain source fitting / source held-out / external, each with first2 saved prototypes and first2 saved random members. HAM rows are fitting-only. All earlier panels remain, collapsed and deep-link accessible. Expanded views include remaining saved examples, initial members and low-margin diagnostics; all concepts and complements retain stable IDs. New compact review forms use separate explicit storage and exports; existing22 review files were hash-preserved. Source images may be enlarged for display only, without added detail or crop.

Limited visual observation: for HAM case578, peripheral S0000 and central darker S0002 both map to C002; old E224 maps central S0001 to C001 and peripheral S0000/thin edge S0002 to complement. This distinguishes candidate generation from final assignment on one fixed case, not a segmentation benchmark, medical label or population-level conclusion. The two routes differ in model, input geometry and fitting roles and are not a controlled causal comparison.

Generator: `scripts/research_hub/medical_compact.py`, called after question-led interpretation. Validation: `artifacts/medical-compact-20260924/publication.json`; 1,635 local references checked, all old IDs/images preserved, bilingual/mobile/image/review-export/deep-link checks passed. Resolved local rendering issues: MEDVIS26-001. Existing jobs and results unchanged.

## 2026-09-24 A线视觉解释补充 / Compact robustness comparison

已在原robustness.html热图后加入两张紧凑诊断总览及十类逐类对照图；40固定显示样例×5条件。R随机、S全条件稳定控制、D最高IoU有效切换、E空/零或补空间诊断，记录seed20260921+类别序号。每格叠加/selected-only，金色新增/品红移除，五条件同片段，保留完整ID、原图链接、概念参考成员与人工审阅CSV。图片输入、raw/effective掩码哈希和几何逐例验证；无新推理、远端操作或改实验。

Two diagnostic overviews and ten compact class sheets now accompany bilingual discussion beneath the figures. All40 examples have source hashes and actual224-input/mask alignment checks. All22748 perturbed region records underpin pooled transition/margin summaries. The maximum-IoU switch selection gives onlydilation1 cases (IoU0.965–0.990); this diagnostic selection is not a comparison of population risks. AI Airliner/Golden visual observations remain pending human confirmation. Lower baseline margin among switches is descriptive association, not causal proof or an independent-region test. No semantic-quality or segmentation-accuracy conclusion.

Generator: `scripts/research_hub/robustness_examples.py`, called by existing robustness completion publisher. Evidence: `artifacts/report-robustness-visual-20260924/{local-qa,browser-qa}.json`; report assets/CSV/provenance under `report/hub/robustness-examples/`. ROB26-009/010 retain selection limitations and resolved local rendering problems. Original hpc/check_job.sh and outputs/ preserved.

## 2026-09-24 — Medical report: question-led interpretation / 医学报告研究问题入口

Presentation-only update: `scripts/research_hub/medical_questions.py` runs after medical walkthrough generation. `medical.html` now answers SAM quality, image-level seven-point associations, external-vs-fitting roles, classifier provenance and interpretation without clinicians. Both child pages link back; the long two-case trace is collapsible. Existing concepts, image panels, review records and scientific results are preserved.

Evidence boundaries: no expert pixel-reference segmentation validation or no-SAM comparison; old route has 12 descriptive learned-concept contrasts (plus separate complement results); HAM 101-case all-fit has no held-out concept validation or sign analysis. Complete 12-row results are shown, not selected favourable contrasts. Dataset/classifier preparation is accepted, while similarity review and patient-identity limitations remain. No new experiment authorization is inferred.

Validation: three pages, both languages, image decoding and mobile overflow passed; original IDs and image references preserved. Receipt: `artifacts/medical-questions-20260924/publication.json`. Direct screenshot-view tool was blocked by a host sandbox mount error; browser checks and screenshots completed. Author sources: https://github.com/jeremykawahara/derm7pt and https://derm.cs.sfu.ca/Welcome.html (checked 2026-09-24).

## 2026-09-24 — Cached same-case medical representation update

User requested clearer representation before more experiments. HAM Medical transfer subpage now begins with a bilingual same-case walkthrough: actual short-side300input → original SAM candidates → HU-MCD postprocessed regions → effective masks → saved concept assignment. Cases are first fixed cohort record578 plus previously discussed frame diagnostic752 (not representative sampling);11regions,67display assets,all derived from existing caches. Raw/effective hashes checked;no new inference. Gray panels are display aids,not literal network input;postprocessed regions are not raw SAM candidates. Existing concepts/IDs/images/review records retained. Generator `scripts/research_hub/medical_walkthrough.py` hooked into main report build. Offline desktop/mobile and language QA PASS;temporary staging symlinks failed Windows preview and were replaced by copies,first QA retained. Receipt: `artifacts/medical-walkthrough-20260924/publication.json`. No new scientific claim or experiment.

## 2026-09-24 — Existing report navigation unified

Report-only: homepage gateway → existing reproduction ten-class directory → HU concepts / three-method comparison / class evaluation. All20 class/comparison pages share class/view controls; language and current view persist during switches. Atlas section links precede long figures. No new report entry, scientific arrays/images or review changes. REPRO26-008 and English notes updated. Source: `scripts/research_hub/navigation.py`; evidence/receipt: `artifacts/report-navigation-20260924/`. Preserve unrelated medical/robustness work below.

## 2026-09-24 — Medical transfer report navigation and comparison

User-requested report-only reorganization completed. `medical.html` is now a bilingual two-route comparison (design,accepted results,interpretation,limitations,next priorities), followed by DermaMNIST-C and HAM child entries. `medical-dermamnist.html` retains previous frozen-concept transfer results; `medical-ham-derm/index.html` keeps all accepted HAM concept examples with shared report CSS/header/navigation/language toggle. No scientific result,sample,concept ID,review record or model execution changed. Static/offline desktop/mobile QA passes for all three pages in both languages. Generator: `scripts/research_hub/medical_comparison.py`, hooked after the medical results extension in `scripts/build_research_hub.py`; prior pages backed up at `artifacts/medical-report-layout-20260924/before`,receipt `publication.json`. Do not restore the former standalone HAM layout on rebuild. Differences are descriptive,not evidence that switching to HAM improves transfer: fitting cohort,geometry andcheckpoint all differ.

## 2026-09-24 — Compact concept figures and masking definitions

Presentation-only update:10 per-class three-method summary figures appear first, detailed member views fold; all ten HU classes have a Fig3-style overview40/147 plus complete147-concept sheets and random-member companions,57PNG exports. No scientific execution or resampling. Source-verified inpainting/area explanation distinguishes ACE filling, HU Input Masking, original display masks and final benchmark prediction; Fig7 ablation remains unrun. See `docs/COMPACT_CONCEPT_PRESENTATION_20260924.zh-en.md`, `artifacts/report-compact-20260924/publication.json`. REPRO26-006/007 and English notes updated. Independent medical/robustness work below remains unchanged; do not infer its current state from older report discussion.

## 2026-09-24 当前：A线补跑验收完成 / Fixed-slot completion accepted

部署28889386与四类28889387/28889388/28889389/28889391均COMPLETED0:0；四GPU作业耗时29/26/28/39秒，共122 GPU秒（按Slurm elapsed相加，不是GPU利用率）。完整执行提交`4551aa8f817a066e18424264407204f3c25aab17`。本地收集哈希、模型/精度/输入/概念基身份、原identity/r1逐字节复用、掩码几何、分配/重构、占位输出丢弃验收通过。6个原槽位控制batch共48行特征差为0；6个空片段保持NaN/-1，分母未变。十类五条件28435/28435片段条件记录完整。原失败与机制不确定性保留；没有新的提交、SAM、拟合、调参或放宽容差。

All ten classes/five conditions accepted. Equal-class means after within-image/within-class averaging: identity100%; erosion1 **78.2522%**, erosion2 **68.6089%**, dilation1 **80.8992%**, dilation2 **74.5703%**. Assignment stability only, not segmentation accuracy or human interpretability. Old six-class summaries remain historical. Known np.bool8 warnings retained. No further repair experiment is needed; later scientific comparisons require their own protocol/authorization.

证据/Evidence: `artifacts/bunya/robustness-completion-20260924/acceptance.json`, `final-sacct.psv`, merged `robustness-*.csv`, `warnings-and-events.json`. Canonical report: `artifacts/bunya/ten-class-review/report/robustness.html`; scoped generator `scripts/research_hub/robustness_completion.py`; other medical/baseline branches and reviews retained.

## 2026-09-24 A线已授权保持batch槽位补跑 / Fixed-slot completion submitted

用户“好 那我们试试”批准本次四类修正提交。部署 **28889386**；Airliner **28889387**、Hummingbird **28889388**、Container Ship **28889389**、Beach Wagon **28889391**。完整执行提交 `4551aa8f817a066e18424264407204f3c25aab17` 已推送。每类1L40S/4CPU/16GiB/20min/1GiB，两资源lane、本批最多两GPU；部署1CPU/1GiB/5min。原identity0/erosion1缓存复用；只计算缺失三个条件。原槽位的原始输入仅占位，输出丢弃，空区域保持NaN/-1且分母不变。不放宽阈值、不变精度、不重跑SAM/拟合。

Submitted, NOT accepted. First/onlypostcheck:deploymentRUNNING,fourGPUjobsPENDING/Dependency. Wait user completion notice before collection; no further polling or automaticretry. BothlocaltestsPASS; GPUentrycontrol remainsrequired. Oldfailures retained. Figure at `report/robustness.html?lang=en` fixed and browser-checked; displayedresults are still previousaccepteddata. Othermedical/baseline branches below remain unchanged. Evidence: `artifacts/bunya/robustness-completion-20260924/`; protocol [ROB26-007](ROBUSTNESS_COMPLETION_20260924.zh-en.md).

## 2026-09-24 当前：Derm7pt拟合28889071工程验收通过 / Current acceptance

28889071COMPLETED0:0/4分39秒/MaxRSS6310204K，433文件本地校验通过。101/101图顺序和身份、2886SAM候选→423保留区域、原/有效掩码及特征映射全部通过；72零特征排除拟合、351有效片段；5初始cluster→3学习概念（8/33/21维）+1986维补空间。最终分配62/106/45/138；completeness0.78525。全423片段及101图分类头代数通过，概念+补空间重构各8保存样例通过，不声称全量分解检查。C003的118初始成员有78最终分给补空间，需语义审阅；有限AI检查见C001边框例，未改规则。101图无掩码缺失或全零图。当前MEL预测34/101，是拟合队列描述，不是独立外部概念验证。

Engineering acceptance only; no new predictions,training,retries or evaluation submitted. All101cases fit the concepts;zeroheld-out. Stable IDsHUMCD-DERM7PT-R101-MEL-C001..003;complement separate. Human semantics andclinical relevance remain unverified. Evidence: `artifacts/bunya/ham-derm-discovery-20260924/discovery-acceptance.json`, `discovery-concepts.csv`, `cluster-to-final.csv`;generator `scripts/accept_derm_fitting.py`. Local first replay failure omitted original PythonRNG;seed43fixed-order replay is pixel-exact,history retained(DERMFIT26-002). Original-resolution additional image export was rejected by automatic approval;report uses previously collected truepre-SAMinputs,explicitly labelled(DERMFIT26-005).

Canonical report gains `medical-ham-derm/index.html` under the existing entry,with128traceable example selections/108panels and human-reviewCSV;no prior results or review keys replaced. No medical experiment is pending after this batch. Next useful work is human concept review and an explicitly approved evaluation protocol;do not silently treat these101fitting cases as held-out.

## 2026-09-24 — Paper coverage and interpretation report update

Report-only update; no new experiments or predictions. All ten classes now have uniform ACE/MCD/HU concept comparisons and four-method bidirectional evaluation links; Golden remains included in the pooled500 images. Added34 bilingual chart discussions and high-resolution fixed-selection exports. Paper coverage is explicitly incomplete: formal Table1 human study and supplement Fig7 inpainting ablations are missing. Benchmark MCD has<5 concepts in six classes; do not silently use it as the paper human-study configuration. See `docs/PAPER_COVERAGE_AND_INTERPRETATION_20260924.zh-en.md`, `artifacts/report-interpretation-20260924/publication.json`. REPRO26-001–005 and English Obsidian notes record scope, interpretation and resolved display normalization. All human ratings remain unreviewed; limited Airliner/Golden AI descriptions are provisional. Preserve the independent medical work below; this update does not inspect or change its jobs.

## 2026-09-24 最新：HAM分类器已验收；Derm7pt拟合28889071已提交

28885816COMPLETED0:0/12m30s/MaxRSS3309832K。72份索引文件全部本地验收；50轮、6450次更新、410750次训练图片访问，逐轮无重复/缺失；验证集选中epoch27，五项适配/分类头代数检查通过。源域测试准确率89.32%、平衡准确率68.97%、MEL45/70。仅验收固定模型的工程完整性，不认证医学适用性，不调参或重训。报告与原始证据：`artifacts/bunya/ham-derm-discovery-20260924/classifier-results.zh-en.md`及`classifier-acceptance.json`，含类别分母、曲线、混淆矩阵。

Derm7pt正式拟合 **28889071** 已提交，执行版本 **649ce176c7d2e4e17b9e9be196fb0e7c33ef3e42** 已推送；固定epoch27checkpoint SHA256`bee957a2a13b894ea12a6f41cc9d5b38f206fc65b4a60fd1ebd81446099c0d37`；1L40S/8CPU/32GiB/1小时/8GiB。R101全部101用于拟合、0留出；保留SAM前短边300原比例，未新增224方图。发布算法、原阈值、mincluster50、outlierquantile1.0不变。旧E224掩码不复用；旧实验与失败历史不改动。10项局部软件测试通过，没有SAMprobe。test-only虚拟编号28889051不是提交作业。

Classifier engineering acceptance PASSED, with class-dependent limitations. Discovery SUBMITTED, not completed or accepted. New adapters bind identities, nonsquare geometry, stable method/cohort concept IDs and fit-only roles; the released science modules are unchanged. Raw masks/features/clusters/assignments/reconstruction evidence will be retained. No held-out inference, retraining, retries, budget extension or new evaluation. Await user completion notice before collecting28889071. Failure must be reported without resubmission.

## 2026-09-24 — Remaining nine-class MCD/ACE results accepted

Publication complete (2026-09-24 Brisbane): original private report entry updated after 88,771 local-reference checks and 44 offline browser checks; previous live report backed up. `final-20260924/delivery.json` and `publication.json` record final hashes. New baseline cards:185 including9 complements; all semantic reviews remain unreviewed. Ten-class MCD47 learned; ACE138 nominal-retained (113 above/25 below control). English Obsidian notes updated with preservation receipts. No further baseline work or queued baseline jobs remain for this batch; future experiments need their own authorization.

All 63stages COMPLETED/0:0 and accepted; Golden reused. Ten-class four-method bidirectional evaluation complete under approved reference protocols. See `docs/NINE_CLASS_BASELINE_RESULTS_20260924.zh-en.md`, final root `artifacts/bunya/nine-class-baselines-20260923/final-20260924/`. No new predictions/retries. Equal-score ACE tie-order differences resolved using recorded valid order; original failed replay preserved (BASE26-006). ACE nominal-retained counts/directions and all warnings in summary/registry; human review remains independent. Existing report stays the canonical private entry. This baseline update does not change the separate medical/robustness work below.

## 2026-09-24 当前：输入已验收，HAM分类器28885816已提交 / Current

MED26-HAM-DERM-DISCOVERY-01：28883445COMPLETED0:0/158秒/MaxRSS2307524K，110份本地文件哈希、101例非方形SAM输入及24项张量检查通过；1.5GB分类器缓存留scratch，由训练worker读取前核验SHA256。无SAM前224方图。正式分类器 **28885816** 已提交，完整执行提交 **4a7360c31e41fb88f94734b0cd8167ae497c9b42** 已推送；1L40S/4CPU/16GiB/1小时/5GiB上限，50epochs/batch64/trainingseed43/普通CE/原精度。调度test-only通过，虚拟编号28885804不是实际作业。不会自动重提、延长或扩容。

Input preparation is accepted; classifier training is submitted, NOT completed or accepted. Derm7pt discovery has not been submitted; it needs the new classifier checkpoint and adapter acceptance. All101R101cases remain authorized for fitting,0heldout; no old checkpoint substitution. Wait for user completion notice before collection. Evidence: `artifacts/bunya/ham-derm-discovery-20260924/{inputs-acceptance.json,classifier.submission.json,classifier-release.json,classifier-preparation-tests.log}`. Historical statuses below remain historical.

## 2026-09-24 最新：原图验收通过，原比例输入准备28883445已提交

28883403COMPLETED0:0/86秒，10,015张600×450RGB、ID/标签/lesionID/划分及直接224像素对应全部通过。用户再次明确SAM前禁止额外224方图；只用发布版短边300保持比例。新CPU输入准备 **28883445**已提交：2CPU8GiB20min0GPU，版本 **1131bd71f43c2dc202edc938bc6f3c2674c2f097**。SAM输入与分类器专用缓存分离并设硬检查；R101全部101拟合、无留出。等待用户完成通知再收集，不硬等、不重提；训练/SAM/拟合尚未提交。详见[新分支记录](HAM_DERM7PT_DISCOVERY_20260924.zh-en.md)。下方旧待审计状态为历史。

## 2026-09-24 HAM原图已下载和传输；数据审计28883403已提交

原始HAM官方ISIC2018Task3Training原图/标签/病灶分组已下载至独立WindowsDownloads目录并scp至scratch。10,015唯一ID、诊断与lesionID本地初查一致。审计28883403：2CPU8GiB20min0GPU，绑定已推送7547f3decad201530eeb13e1fb8b8d0d5b510779，尚未收集/验收。不等待完成；用户通知后收集。目录准备28883377已成功。R101全部101例拟合已获批，无本轮留出；训练与发现尚未提交。详见[当前状态](HAM_DERM7PT_DISCOVERY_20260924.zh-en.md)。下方下载进行中/问题待答复为历史。

## 2026-09-24 R101全量拟合与原始HAM下载已批准

更新 MED26-HAM-DERM-DISCOVERY-01：用户已明确批准101例全部拟合（不设本轮Derm7pt留出），以及原始HAM下载到独立Downloads目录后scp到scratch。不要再询问这两项。官方Harvard元数据HTTP403，改用作者支持的官方ISIC2018Task3Training入口；原图下载进行中，尚未验收。scratch目录准备28883377已COMPLETED0:0/4秒，数据审计2CPU8GiB20min等待完整下载/传输后提交。训练/发现尚未提交。详细状态与未变边界见[新分支记录](HAM_DERM7PT_DISCOVERY_20260924.zh-en.md)。下方“范围待答复”为已解决历史。

## 2026-09-24 HAM分类器 → Derm7pt概念发现：已批准方向，本地准备完成，未提交

用户明确将新分支改为HAM训练ResNet50、Derm7pt拟合概念，先跑通发布算法，评价后补；加权CE已撤回，不在批准范围。旧HAM400/70发现与R101外部转移结果保持不变。SSH master确认有效。已添加复用原图处理代码的输入适配与两项通过的合成软件测试，无模型执行。原始HAM文件位置及Derm7pt拟合范围已询问，等待答复；不得默认R101评价授权就是101例拟合清单授权。没有提交作业，也未收集独立九类基线批次。详见 [新分支决定与准备](HAM_DERM7PT_DISCOVERY_20260924.zh-en.md)，稳定ID MED26-HAM-DERM-DISCOVERY-01。

## 2026-09-23 九类基线新批次已提交 / Nine-class baseline batch submitted

用户已批准剩余九类MCD与ACE — HU-MCD released-code reference发现及双向评价，沿用Golden协议，人工审阅不是前置条件。已核对旧回执、收集与实际队列/会计，未重复提交。执行版本`1e094cd7fa9f3e6990e6aad1294e85b96b82905e`已推送；部署28881848，63阶段作业均成功提交。唯一提交后快照：部署及ACE Siamese_cat输入28881853完成，62阶段PENDING；新结果均未收集/验收。其他医学工作状态按下方各自最新验收，不受本批改变。

Nine remaining classes' MCD/ACE-R chains authorized and submitted once:63stagejobs plus deployment28881848,commit1e094cd. Post-submission snapshot only:deployment and ACE Siamese_cat inputs28881853completed,62stagespending; none of these new results locally collected/accepted. Golden and all tenHU/Random accepted results reused. Human review independent. No waiting/polling for completion; **collect/accept/update result curves only after user batch-completion notice**. Failuresnotretried, budgetsnotexpanded. Twoindependentmethodlanes,max2L40S,total27GPU-hourallocationceiling,notestimatedactualuse.

当前凭据：`artifacts/bunya/nine-class-baselines-20260923/{batch.json,submissions/,post-submission-scheduler.json,input-reuse-audit.json,local-source-preflight.json,release-tests.log}`；协议：`docs/NINE_CLASS_BASELINES_20260923.zh-en.md`。BASE26-001/002/003记录工程适配、本地已解决问题与继承科研限制。原私有报告已增加待验收阶段；后续不得把SUBMITTED或schedulerCOMPLETED等同ACCEPTED。保留所有固定清单、灰度映射、旧证据/审阅及无关修改。

<!-- REPORT-HUB-20260923 -->
## Current report update / 当前报告更新 — 2026-09-23

The private report has been reorganized around reference reproduction, segmentation sensitivity and medical transfer, with global language selection. All157 HU-MCD entries now have limited-scope AI observations (99 newly inspected); human judgments remain separate. No new experiments. See `docs/REPORT_REVIEW_20260923.zh-en.md` and `scripts/build_research_hub.py`; preserve the original report snapshot under `artifacts/report-redesign-20260923/source-report/`. This presentation update supersedes old display-only Not run labels, not scientific records.

## 2026-09-23 当前验收 / Current acceptance

外部部署28796207及推理28796209均COMPLETED0:0（6秒/3分16秒），524文件收集校验通过；101例全部完成，29/101预测MEL，296片段中37零特征，C00194/259有效片段。固定输入/权重/概念基与24项描述性统计验收通过；医学适用性、语义、空间定位与患者独立性未获证明。A仍6完整/4部分失败，未补跑。当前无待收集的E224作业，没有新增作业/重训/重提。

External jobs28796207/28796209 completed0:0(6s/196s);524files verified,all101cases processed,MEL29/101,37zero-feature regions,C00194/259validregions. Frozen input/model/basis and24descriptive comparisons accepted; clinical utility,semantic/localization validity andpatientindependence remain unestablished. Aremains6complete/4partialfailed withoutretry. No E224 jobawaitingcollection andnonewjobthisturn.

当前证据与双语结果：[外部验收报告](MEDICAL_EXTERNAL_RESULTS_20260923.zh-en.md)、`artifacts/bunya/medical-external-results-20260923/acceptance.json`。本节替代下方历史“外部已提交未收集”状态，不增加任何新实验授权；保留所有旧失败与后续成功历史。

## 2026-09-22 本批验收及E224实际提交 / Batch acceptance and E224 submission

本批最终验收：A十类原batch不变对照与侵蚀r1通过，r1图片级类内平均再十类等权平均78.25%；6类五条件完整，4类r2缩批对照失败，6612后续region-condition缺失。医学D1 28792059完成400/70，3初始cluster→1学习概念（32维），训练514/849、留出82/143分配该概念；零特征68/917和14/157，原始证据完整，重构/映射通过。外部E224批准并提交：部署28796207、推理28796209，版本dcc61934973aa9054a4e689ccf2334652447c575；首次核对待依赖，尚未收集/验收。失败A未重提；不等待新作业。

Final batch acceptance: all10 original-batch identity and radius1erosion controls complete; equal-class image-mean retention78.25%. Six classes complete5conditions; four fail reduced-batch controls at radius2, leaving6612region-condition slots missing. MedicalD128792059 completes400/70images:3initial clusters→1learned32-dimensional concept;514/849training and82/143held-out assignments;68/917and14/157zero features preserved. Masks/IDs/reconstruction accepted. ApprovedE224submitted: deployment28796207,inference28796209,commitdcc61934973aa9054a4e689ccf2334652447c575; pendingdependency at the first check, notcollected/accepted. FailedAjobs not retried; no waiting for new jobs.

当前证据：[本批结果与限制](BATCH_RESULTS_20260922.zh-en.md)、[E224完整协议](MEDICAL_EXTERNAL_EXECUTION_20260922.zh-en.md)、`artifacts/bunya/batch-review-20260922/acceptance-both.json`。以下旧“待提交/未授权/未完成”等均为历史快照；成功不删除失败历史。用户通知新批次完成后才收集。

## 2026-09-22 最新D1成功提交 / Latest successful D1 submission

D1已成功提交：28792059，提交后唯一状态核对为PENDING，尚未运行完成或验收。代码5623abe8a843deaea7ab838a8c76c41afbad565e，qos=short，1L40S/8CPU/32GiB/2h/15GiB。部署28790767及D0 28790768均成功，现有凭据已核对；仅移除失效调度afterok，保留运行时版本、D0成功/清单/输入/权重哈希和GPU适配门槛。此前QoS和过期依赖两次拒绝完整保留（MED26-016/017）。没有重提部署或D0，不等待实验完成；用户通知后再收集验收。

D1successfully submitted as28792059; PENDING at the sole post-submission check, not completed or accepted. Commit5623abe8a843deaea7ab838a8c76c41afbad565e, qosshort,1L40S/8CPU/32GiB/2h/15GiB. Completed deployment28790767 and D028790768receipts verified. Only expired schedulerafterokreferences removed; runtime commit/D0PASS/manifest/input/weight/GPUadapter gates retained. Both prior submission rejections remain (MED26-016/017). Deployment/D0not repeated. Stop waiting; collect after user completion notice.

以下待重提文字为历史快照，不覆盖本次成功提交；资源及科学配置不变，外部推理和分类器重训仍未授权。

## 2026-09-22 最新：D1第二次提交被拒，未运行 / Latest: second D1 submission rejected

D1最新：获准的QoS修正重提被Slurm再次拒绝（Job dependency problem），未创建作业。部署28790767与D0 28790768均COMPLETED0:0（6秒/29秒），但已退出控制器活动记录，原afterok失效。已核验Slurm生成的部署/D0凭据；仅移除失效调度依赖的方案已通过sbatch --test-only（不是提交，预估编号不是实际作业）。同版本/科学配置/资源/运行时输入和GPU适配门槛保持不变；等待用户批准再提交一次，不自动重提（MED26-017）。

Latest D1: the authorized QoS-only retry was rejected for Job dependency problem; no job created. Deployment28790767 and D028790768 both completed0:0 (6s/29s), but their records left the active controller, invalidating old afterok references. Compute-generated readiness/input receipts verified. A plan removing only expired scheduler references passed sbatch --test-only (no submission; its estimated ID is not a real job). Same commit/scientific config/resources and runtime input/GPU-adapter gates retained. Await one-time resubmission approval; no automatic retry (MED26-017).

完整版本仍为`5623abe8a843deaea7ab838a8c76c41afbad565e`；GPU上限1L40S/8CPU/32GiB/2h/15GiB。已准备具体更正`artifacts/bunya/medical-discovery-20260922/D1-expired-dependency-correction.json`，未第三次提交；不要把test-only预估编号28791892当作已提交作业。下方旧“QoS待修正”等属历史。


## 2026-09-22 D0/D1最新状态 / Current discovery state

用户已批准D0+D1+部署；固定epoch10，独立RandomState43与病灶/图片排序规则400病灶各1图，全部70held-out不变。完整执行版本 `5623abe8a843deaea7ab838a8c76c41afbad565e` 已推送。医学发现预算上限1L40S/8CPU/32GiB/2h/15GiB；D0为1CPU/4GiB/10min/0.5GiB；部署1CPU/1GiB/5min。外部推理仍未授权。

- 部署 **28790767**：COMPLETED0:0/6s；部署就绪门槛在Slurm内执行。
- D0 **28790768**：已提交；唯一提交后核对时RUNNING。尚未收集/验收，不以已完成选样计划代替图片验收。
- D1：**未成功提交、无作业ID**。助手误填gpu_cuda的QoS为normal，Slurm明确拒绝；队列/会计未发现该任务。仅改QoS为既有short的计划/script已准备，科学配置/版本/资源上限/依赖均不变。按“失败不自动重提”，等待用户批准重提一次；原拒绝回执保留（MED26-016）。
- MED26-015：原“1%离群”和“cluster搜索”为文档错误。保持outlier_percentile1.0（100%分位点，有限行无该离群剔除）及int(mean+std)原启发式；SAM1%面积阈值是独立规则。

Latest actual status and exact receipts: `artifacts/bunya/medical-discovery-20260922/handover.json`. D0 remains independently submitted; D1 is not queued and its QoS-only retry requires explicit permission. Do not resubmit deployment or D0. Do not wait/poll for completion; collect only after user notice.

# 当前状态摘要 / Current reconciled status — 2026-09-22

<!-- MEDICAL-EXEC-CURRENT:BEGIN -->
## 2026-09-22 最新验收 / Latest acceptance

- 部署28781904、H28781914、M28781915均COMPLETED0:0，Slurm耗时6s / 88s / 782s。刷新凭据和完整收集：`artifacts/bunya/medical-results-20260922/`。部署验收复用，不重新运行。
- 执行完整commit `00f06dc549439db92e4842cb5ecd761f9b38bc43`；本地验收脚本不改变已执行源码。H/M输入、实际配置、checkpoint/数组/映射均通过工程验收，见`acceptance.json`；医学适用性单独判断。
- H计划10,125,165对×每hash8方向全部完成；0 native / 0 canonical224精确匹配。5913候选含4291阈值候选；R101中214阈值对涉及20病例，源train/val/test=180/12/22。不是确认重复；全部并排图片及人工表保持未审阅，不自动排除。
- M50轮/6450更新/410750训练图次完整；验证宏AUC选中epoch10，SHA256 `07a6f65dbb23eb7c97510bef09c4b624a1a86007b27a64e324151fd1dfd2a655`；test accuracy88.9976%、balanced59.6981%、macroAUC0.968671，MEL50/70（61病灶）、precision50/89、AUC0.944104。
- MED26-012：DF验证0/5、测试0/8召回，整个val/test均无DF argmax；不是已确认代码错误。MED26-013：后期train loss下降而val loss上升，固定epoch10不变，不重训。MED26-014：GPU利用率/显存峰值和train accuracy未记录，不编造。
- R101/S仍仅评价协议批准；医学发现和外部推理**未执行、未授权**。下一批源域400病灶/70held-out的具体提案与预算见`docs/MEDICAL_RESULTS_20260922.zh-en.md`，不因本次验收升级为批准。
- 本轮无新实验/预测/远端处理作业/选样。私有原报告增量更新，既有157概念页、审阅与历史证据保留。

All three jobs completed; both H retrieval and M classifier passed engineering acceptance with explicit research limitations. H candidates await human identity review. The frozen M model supports a conditional case-study proposal, not clinical or seed-stability certification. New medical discovery and external inference remain unapproved; no jobs were submitted this turn.
<!-- MEDICAL-EXEC-CURRENT:END -->

### 下方为之前的状态与决定历史 / Earlier status and decision history

<!-- MEDICAL-PREP-CURRENT -->
> **2026-09-22当前准备更新：H/R/S/M仍未批准。** 用户已授权继续代码/配置/软件检查；H/M worker、Slurm入口、S纯计算函数及18项本地检查已完成，没有远端医学处理或新实验/作业。首轮推荐解释一个固定分类器，不验证训练seed稳定性。50/b64=410750训练图次/6450更新；100/b128=821500/6500；50不是充分训练证明，不自动延长。覆盖＋12比较只回答预定外部MEL范围内的图片级关联。最终推荐H＋R101＋S描述性＋M50/b64/训练seed43，详见当前集中决定表；资源上限不变，等待明确选择。本地准备提交`227eedf5a51c2c8dfd93b03f205e67347f4a248e`未推送/未部署；实现/证据版本见`artifacts/bunya/next-phase-20260922/preparation/`。以下“仅文档/实现尚未完成”等是较早状态，不能覆盖本更新；较早SSH与队列状态仅为当时快照，本轮没有新远端查询。
<!-- /MEDICAL-PREP-CURRENT -->

> **2026-09-22决策依据补充：用户再次明确H/R/S/M尚未批准。** 当前集中决定表与Obsidian已按MED26-008/H、MED26-002/R-MEL、S-GROUP、S-COVER、S-12、M-SEED、M-EPOCHS补证据与替代。400病灶选样seed43仅属D11方向批准；分类器seed43未批准。50epochs无已验证最优/充分/收敛依据，是预算提案；作者默认100epochs/batch128并循环3次。覆盖与12比较为本项目分析设计，不是原论文标准。当前仅文档更新，没有新作业/实验或选样。

本摘要优先于下方历史计划/状态；仅用它定位证据，不将摘要本身当授权。证据：`artifacts/bunya/next-phase-20260922/{context-reconciliation.json,authorization-ledger.json,accounting.psv}`。SSHmaster88416有效；实时队列为空。代码HEAD822860db，工作区还有此前及本次未提交改动；不将当前源码假定为既有作业执行版本。

1. **完成且验收可复用：** 十类HU-MCD发现/双向HU+Random，GoldenMCD6概念及ACE released-code-reference3概念/双向评价（final-20260917/delivery.json）；DermaMNIST-C28738133及Derm7pt28734478数据准备分别已验收。源数据/RDM/scratch路径复用，不重新下载或完整审计。
2. **实际执行但未验收：** A原10作业已结束：6COMPLETED、4FAILED（28731217airliner、28731220hummingbird、28731223container_ship、28731228beach_wagon），退出1:0；结果未收集/验收，失败原因未知，不把退出状态当identity门槛通过/失败。保留全部历史，不重提、不扩大预算；按用户通知节奏后续收集。
3. **已批准边界：** A五条件/10类，每类1L40S4CPU16GiB20min、最多并发2；原模型概念精度/得分不变，空/零保留，对照失败停止该类；不自动重提。B仅病灶优先seed43/400不同病灶各1图方向（待模型/预算），固定70测试图按61病灶考虑相关性；数据准备已经完成，不扩大成医学实验授权。
4. **仍待批准：** H跨数据集相似/重复审计4CPU8GiB30min0GPU2GiB；R外部101/100/248范围；S覆盖/类别合并/12描述性对比（可选bootstrap另行选择）；M标准ResNet50训练提案1L40S4CPU16GiB4h5GiB。原始主方案/旧提案的“pendingA”“数据缺失”“等待重提”均是历史，不覆盖后续批准和验收。
5. **用户9例审阅：** 全部正常显示、未见明显损坏/不相关；像皮肤镜只是非医学初步印象。仅9例，非1011例；未审诊断/征象/患者/跨数据集身份；不要求删改。录入2026-09-22Brisbane，实际审阅日未知。已记录到原manual-review.csv及user-review-20260922.json；专业确认仍开放。
6. **下一批：** 当前无尚未提交且具备完整授权的GPU任务。H可在明确批准、实现/版本与预算门槛满足后单独提交；M须批准其具体配置并完成实现/新路径检查，不依赖R/S；医学HU发现及外部推理另需预算/授权。已授权的本地准备、报告/审阅登记继续完成，不请求整个项目总确认。

模型具体方案：`docs/MEDICAL_RESNET50_PROPOSAL_20260922.zh-en.md`。简明H/R/S/M决定入口：`docs/NEXT_PHASE_CURRENT_DECISIONS_20260922.zh-en.md`。分类器训练8215图，与后续单类400病灶概念发现不同。未启用医学SAM/SSC、模型推理、训练、删图或隐式默认协议。

---

# 历史交接记录（保留，非当前状态） / Preserved historical handovers

> **Derm7pt observation clarification — 2026-09-21:** Read `docs/DERM7PT_OBSERVATIONS_REVIEW_20260921.zh-en.md`. Independent audit/render code eb3c77506ed279c6adc6dd5891f5f1ad54dc5855 confirms raw counts and all original cells. All9source notes document replacement of clinic images with derm images;4explicit missing-clinical cases agree with paperp4,5motives unspecified. This corrects prior blanket cause-unknown wording; do NOT call these original data errors.984exact-empty case_id/27nonempty, field semantics unknown; case≠patient.1011distinct derm paths/file/pixel hashes; cross-role aliases do NOT duplicate derm-only samples. Existing report now links private nine-case review and full1011row index. Manual review remains UNREVIEWED. No new Slurm job or model/data changes; original medical protocol choices still pending.

> **Derm7pt prepared and locally accepted — 2026-09-21:** corrected job28734478 COMPLETED0:0/13s,bun148. Exact code/launcher guards passed; ZIP identity/copy/CRC extraction and all images audited in Slurm. Acceptance: `artifacts/bunya/next-phase-20260921/derm7pt-resumption/acceptance.json`; paths: `dataset_paths.json`.1011cases,official413/203/395;2022RGB role records/2013unique image paths.9 same-case derm/clinic aliases;984missing case_id;0exactcrosssplitpixelgroups. Data staging accepted only, patient/cross-dataset independence and medical protocols NOT established. Failure28734311 resolved with user-authorized correction, all history retained. No new experiments or DermaMNIST download.

> **User-approved launcher correction submitted — 2026-09-21:** user explicitly approved one corrected Derm7pt submission. Job28734478, deployed commit83972e99763ef6ecaebba3cbad6c0bf14e62e63a,2CPU/8GiB/20min/0GPU,bun148, reads RDM ZIP and writes only scratch. Runtime hash now checked against git show; historical failed28734311 preserved. Current status in `next-phase-20260921/status.json`; no additional retry authority. Exact DermaMNIST HTTP504 remains independently paused.

> **Latest Derm7pt resumption — 2026-09-21:** user independently read supplied RDM ZIP in interactive Slurm (28733184/bun147); no claim of ZIP integrity or user-tested write permission. User explicitly authorized renewed staging. Job28734311 FAILED1:0/4s onbun148 BEFORE data access: assistant populated worker_sha256 with derm7pt_audit.py hash instead of runtime entry hash. See MED26-005 and `artifacts/bunya/next-phase-20260921/derm7pt-resumption/`. Corrected plan rendered and verified against deployed commit83972e99763ef6ecaebba3cbad6c0bf14e62e63a; NOT submitted pending explicit one-time retry approval. No scratch working copy prepared. Earlier failures preserved; no model or DermaMNIST retry authorized. Latest paragraph supersedes historical no-Derm7pt-resubmission wording below.

> **Final actual data-preparation outcome — 2026-09-21:** first actual download job28733630 FAILED1:0/35s on bun148, execution83972e99763ef6ecaebba3cbad6c0bf14e62e63a. Existing-file inventory COMPLETE (target NPZ absent in inspected scratch/Q9903dataset trees), official CSV verified/reused, RDM parent+lock writes succeeded. Exactly ONE network request, curl22/HTTP504 after31s,0bytes. No retry, NPZ audit, sealed archive or ready path config. Complete collection: `dermamnist-exact/verified-node-collections/28733630/20260921T050725.304151Z`. Asked user to choose local browser download vs pause; pending. All earlier data stages attempted zero downloads. Keep independent A unchanged and do not requeue Derm7pt.

> **Q9903 access diagnosis succeeded on another node — 2026-09-21:** user confirms HPC+uqcche38RW. Read-only job28733281 PASS0:0/4s on bun148; actual Q9903/datasets/Derm7pt raw directory lists succeed. Account id-G includes Q9903RW549903, process groups omit it; do not infer a unique cause. Prior bun147 failures preserved. First-acquisition job28733630 now submitted explicitly on bun148, commit83972e99763ef6ecaebba3cbad6c0bf14e62e63a,1CPU8GiB1h59min/noGPU, deployment28733627. All earlier completed jobs attempted zero downloads. It will recheck/reuse existing files, enforce one network attempt, audit then archive and publish paths. No permission change, no Derm7pt resubmission, no model authority. Wait for completion notification after one scheduler snapshot.

> **Final first-stage result — 2026-09-21:** exact data job28732647 FAILED1:0/4s, complete hash-verified collection in `next-phase-20260921/dermamnist-exact/collections/28732647/20260921T045921.036585Z`. On bun147, even /QRISdata/Q9903 root access is Permission denied; /QRISdata itself readable. Scratch target CSV verifies; no exact target NPZ in inspected scratch; RDM presence unknown. No download attempted, nothing archived, no ready data-path config published. User asked to verify HPC-enabled Q9903 and uqcche38 read/write membership; pending. Do not auto-retry. Deployment28732646 succeeded0:0/3s. Data-only authorization persists but does not authorize bypassing access restrictions or models.

> **Exact medical data preparation authorization — 2026-09-21:** New user approval supersedes the earlier NO DOWNLOAD hold. Deployment28732646 and data-only preparation28732647 submitted at commit743d29886aed3c4878c5ffb07aef83de2b5a0d7e. Budget1CPU/8GiB/2h/0GPU, one attempt, no retry. First inspect existing scratch/RDM copies and access; reuse exact target or download once only if absence established. Audit then archive underQ9903 distinct Zenodo12739457+MD5 version; configuration published only after archive verification. Read `docs/DERMAMNIST_EXACT_PREPARATION_20260921.zh-en.md` and `artifacts/bunya/next-phase-20260921/dermamnist-exact/`. No model/training authority added.
>
> **Access blocker:** Derm7pt28731216 actually FAILED1:0/4s with PermissionError at supplied Q9903 ZIP before any copy/extraction. Collected evidence preserved; no retry. Exact-release worker must stop before download if remote inventory/access cannot be completed. Account/collection permission cause requires evidence, not assumption. A jobs remain original submissions and release40418ed; do not modify or duplicate.

> **Current authorized batch — 2026-09-21:** execution `40418edfa4e4ac81eda02d954fe59de52a0f14f0` pushed to reproduction. Read `docs/NEXT_PHASE_EXECUTION_20260921.zh-en.md` and `artifacts/bunya/next-phase-20260921/status.json`. CPU deployment28731214 COMPLETED0:0/3s at submission snapshot. Derm7pt RDM-to-scratch audit28731216 RUNNING; A five-condition jobs 28731217,28731219,28731220,28731221,28731222,28731223,28731224,28731228,28731229,28731232 PENDING at that snapshot, max2concurrent. No results collected; do not poll/wait or retry; user will notify completion. Exact input/weight/config/cache hashes and per-class unchanged-mask gates are enforced in each Slurm worker.
>
> **Medical data clarification:** user Downloads corrected NPZ hash equals the previously audited OneDrive947968c4 file, not the locked Zenodo84920fb7 release; existing7author-confirmed cross-split duplicate pairs remain relevant. User expressly requested NO DOWNLOAD until disagreement clarified. None submitted. B lesion-first400distinctlesions/oneimage and70testimages/61lesions approved, but final lists/model/training budgets not yet accepted. Derm7pt path now supplied and audit independently authorized; metastasis/crop/sign protocols still undecided. Old missing-access/pending-A wording below is historical.

## 2026-09-21 next-phase preparation — actual status, no new experiment authorization

Read `docs/NEXT_PHASE_READINESS_20260921.zh-en.md` and `artifacts/bunya/next-phase-20260921/status.json`. Existing SSH master passed; actual queue empty and no accounting rows since2026-09-17. Original reproduction remains accepted. A input inventory500images/5687regions/413zeros/5274valid; candidategeometry5conditions28435records,6r2empty; noCNN. B exactZenodo image archive still unaccepted (partial download retained), melanoma metadata1021trainimages/533lesions; Derm7pt actual data/counts unavailable. A scientific/resource approval, B sampling choice and CPUpreparation budget requested together; do not interpret recommendations as permission. No new jobs submitted. Scoped local preparation commit `624c4fb8cd94879fee7497c56d2257e17e081a44` is not yet deployed; user medical/NotebookLM files and outputs preserved. On approval execute only satisfied dependencies; after batch submission one scheduler check then wait for user completion notice, no polling/automation.

## 2026-09-17 final baseline refresh — supersedes pending baseline status below

MCD evaluation28237227 and ACE gradient/evaluation28238065 actually COMPLETED0:0 (33s/31s), collected and independently accepted. Cache-only semantic replay of gradients, tests, assignments and masks PASS. All ten HU/Random evaluations remain accepted. Golden final concepts: HU13, MCD6, ACE-R3. ACE retained K04/K19/K24 all have lower meanTCAV than control under the approved two-sided nominal gate; see ACE26-007. This is disclosed source-reference behavior, not a newly repaired method. No remaining queued job in this authorized batch; no resubmission or new prediction was performed.

Current final results/provenance: `artifacts/bunya/evaluation-baselines-20260910/final-20260917/RESULTS.zh-en.md`, `analysis.json`, `semantic-replay.json`, collection/acceptance receipts and final accounting. Existing report entry remains `artifacts/bunya/ten-class-review/report/index.html`; regenerate status with `final-20260917/integrate_final.py` and the existing staging/publish generator. All fixed lists, existing HU reviews and unrelated changes preserved. Human review and any corrected statistical sensitivity comparison are future work, not executed. Historical pending descriptions below are superseded.

> **Latest actual acceptance / report handover — 2026-09-10 (supersedes queue/preparation text below):** Read `artifacts/bunya/evaluation-baselines-20260910/STATUS.json`, `RESULTS.zh-en.md`, current scheduler evidence and accepted receipts before any new action. Nine remaining HU-MCD/Random bidirectional evaluations all COMPLETED0:0 and independently PASS; Golden28214892 reused. Ten-class500-image raw-native aggregation accepted (107746 predictions). No rediscovery or SAM probe.
>
> **MCD:** features28227213 and CPUfit28234850 accepted;6learned, completeness0.5585487484931946 at first crossing k6; all attemptedk3–6 preserved. Evaluation **28237227 already submitted**, last PENDING,1L40S/4CPU/16GiB/30min. **ACE — HU-MCD released-code reference:** features28227518 and CAV28236990 accepted;25candidates,14structuraleligible,300CAVfits. FinalTCAV filter NOT DONE. Evaluation **28238065 already submitted**, last PENDING,1L40S/4CPU/16GiB/1h. Do not duplicate or auto-retry either job. Same frozen inputs/model/precision; Random28214892 signed reuse planned for both.
>
> **Actual ACE intersections:** discovery/gradient6; validation-role0; actualcontrolpositive/negative0. Twenty rounds repeat one negativeidentityset per candidate/control despite20distinctweights. CandidateCAV train/testshare3–8 sourceimages, so savedaccuracy is not independent-image generalization. Nominalp will affectfilter/evaluation but not calibrated significance. SeeACE26-006.
>
> **Report:** same private `artifacts/bunya/ten-class-review/report/index.html`, now all10HU/Random evaluation curves/tables, baselinefit summaries, MCDall6+OC train/valtop/random panels, stableIDs/reviewCSV, ACE25candidate index/filter audit and actualqueue status. Original157HU IDs/pages/reviews preserved. InitialMCD membership currently numerictransitiontable, no per-membervisualatlas; finalACEatlas/evaluation pending actualresults. English Obsidian notes appended with fullprefixpreserved (`obsidian-results-receipt.json`). All35 protectedcheck_job.sh/outputs files unchanged.
>
> **Local continuation:** `bulk_collect.py` batchesSFTP and reuses only hashverifiedpartials; no remoteprocessing. `accept_baseline_evaluation.py` prepared, notyetusedonbaselineoutputs. `analyze_ten_evaluations.py`, `analyze_accepted_fits.py`, `mcd_display_plan.py`, `render_mcd_display.py`, `refresh_result_report.py` and `ten-class-review/update_workstreams.py` generate currentreport. Preserve immutable acceptedadapters; addnewonesfornewresults. Allremotechecks/processing throughSlurm, scheduler/SFTPonlyonlogin. Medical/NotebookLM unrelatedfiles unchanged.

> **Latest execution handover — 2026-09-10:** This supersedes the historical no-new-evaluation/no-fit instructions below. User explicitly authorized remaining nine HU-MCD bidirectional evaluations, Golden MCD, and now **ACE — HU-MCD released-code reference (R)** including fitting/evaluation after input/cache acceptance. Read `artifacts/bunya/evaluation-baselines-20260910/` and `docs/ACE_PROTOCOL_DECISION_20260910.zh-en.md`. Do not duplicate submitted tasks or automatically retry failures.
>
> **Accepted/reusable:** ten discoveries/audit28222205 and Golden HU+Random28214892 unchanged. ACE input preparation28227379 completed0:0 (1m45s), independent local acceptance PASS: fixed400/50, discovery prefix50, random2000 from2025 examined with25 L skips. Validation intersections0; random includes3Golden,2of400,0ofdiscovery50. Gradient/control and actual CAV round roles freeze from actual feature-stage RNG before fitting, not guessed now.
>
> **Current GPU queue (last observed):** nine HU evaluations28227175/28227179/28227180/28227181/28227182/28227183/28227184/28227185/28227186, commit6317b659c95c4b36aedb0f6bbf9d1ff0bff1c229; Golden MCD features28227213, commit2cf770b34c1b1190942c1613e8a8f78f026f1b4c. MCD fit/eval not submitted; actual accepted feature manifest hash required. Original nine A28226913–28226922 (explicit ledger, not every ID in range) were cancelled while never started, after pre-execution metadata bug was caught; preserve EV26-004 and replacement ledger, no failed experiment retried.
>
> **ACE:** input release9c4ef94e5b98036ec8a3f4f1e417ea64b798f82a. Scientific worker released as a17d2de28813db5e70c6c5066d71163755dae7b0; deployment28227508 completed0:0 in4s. Feature job28227518 submitted once (L40S4CPU16GiB1h), awaiting execution/acceptance; CAV/evaluation not submitted, require actual accepted cache hashes. Approved caps: inputs4CPU8GiB15min (done), features1L40S4CPU16GiB1h, CAV4CPU16GiB1h, gradient+both evaluations1L40S4CPU16GiB1h. Preserve all prefilter candidates, all20round scores/control/test outputs/mean directions/filter reasons. Nominal p participates in selection/evaluation, not calibrated independent-repeat evidence. No corrected comparison, resampling or resource expansion without new authorization.
>
> **Private report/notes:** existing `artifacts/bunya/ten-class-review/report/index.html` only; workstream status updated through generator,157existing stable concept entries retained; offline browser QA passed. Obsidian Implementation Notes appended with hash-checked preservation; latest receipt `evaluation-baselines-20260910/obsidian-validator-receipt.json`. Keep updating anomalies/notes/report upon acceptance. Medical/NotebookLM files are unrelated parallel user work; do not stage/run them in this task. SSH/MFA user only; all remote checks/computation through Slurm, login only scheduler/SFTP.

> **Latest final-results handover — 2026-09-09:** All nine A jobs are now COMPLETED/0:0 and locally collected; ten-class input/cache and all-row algebra audits pass with recorded SSC warnings. Read private `artifacts/bunya/ten-class-review/STATUS.json`, `report/index.html`, and `anomalies_consolidated.json` before acting. Golden discovery28208840 and B28214892 are reused, not rerun. C remains preparation-only: no ACE/MCD fit. Slurm28222205 performed the authorized read-only remote audit/export. Do not duplicate discovery or automatically submit further evaluations, fits, sensitivity experiments or retries. The older workstream status below is historical.

> **Current handover — 2026-09-09:** Read [WORKSTREAM_REPORT.zh-en.md](WORKSTREAM_REPORT.zh-en.md) and the local `artifacts/bunya/workstreams-20260909/STATUS.json` before acting. Nine A discoveries were submitted exactly once; B28214892 completed and passed independent acceptance; C preparation is complete with no fit submitted. Execution releases982045f/f73216b are immutable. Do not treat the historical not-yet-deployed status below as current, or duplicate these jobs. See [ANOMALIES.jsonl](ANOMALIES.jsonl) for all recorded exceptions and dispositions.

# Main-chat handover — 2026-09-08

## 本次范围与交付状态

用户在侧对话授权：整理、检查现有复现代码，提交并推送到自己的 fork，
然后回到主对话推进部署和实验。本次不执行 Bunya 部署，不提交 Slurm 作业。

本次代码版本包含：本地结果 collector、每次运行独立输出/缓存目录、
配置与代码/输入文件记录、阶段进度、Git release 准备工具及现有诊断记录。
提交的精确 SHA 和推送结果见侧对话最终消息；本文件随该提交保存。

- 工作目录：`/home/chen/projects/hu-mcd`
- 分支：`reproduction`
- fork：`https://github.com/cchen-1/hu-mcd.git`
- upstream：`https://github.com/grobruegge/hu-mcd.git`
- 已核对的 upstream commit：`168eb5bf1717ab46d1c84be86bd81f0c51861413`
- 本地保留但不随本次提交发布：`outputs/`、`artifacts/` 中的实验结果、
  数据衍生图片和缓存，以及独立的 NotebookLM 文件。
- 用户原有的 `hpc/check_job.sh` 修改原样纳入提交；原有 outputs 文件内容未改动。

## 用户刚提供的信息

Bunya 上可用的 ImageNet1k 路径：

```text
/scratch/licenseddata/imagenet/imagenet-1k
```

路径来自用户。本次没有登录远端验证它的读取权限、目录结构、解包状态、
标签映射或 train/validation 划分。不要假设它与 Imagewoof 的目录结构相同。
后续在 Slurm compute 作业内检查并建立选样清单；保持共享数据源只读。

用户表示目前已登录 Bunya。SSH 登录和 MFA 始终由用户完成。
后续实操前检查 `ssh -O check bunya`，复用现有 ControlMaster；
它可能在主对话恢复时已过期，不自动发起新登录。

## 研究方向

侧对话建议优先推进单类别论文配置的参考复现，用户希望回主对话继续实操：

- golden_retriever（ImageNet synset `n02099601`）；
- 原始 ImageNet training split 中固定种子随机选取 400 张；
- 对应 validation split 中 50 张，检查与训练清单不重合；
- SAM ViT-H，32 points per side；
- 保持已固定的 released-code 行为，包括最小 cluster size 50、
  masked ResNet50/global_pool、自动 SSC/PCA 等；
- 先分析该类别，再决定扩展十类和 ACE/MCD 对照。

这是下一里程碑的方案，不代表完整论文已经复现，也不证明此前零验证
分配只由数据量导致。Supplement 的 golden-retriever 参考是 13 clusters、
completeness 0.67；不要强制设成 13 或围绕 0.67 调参以获得表面一致。
原论文的完整采样 ID 和历史环境未提供，逐位一致不是已确定的验收条件。

## 已有能力与待接通处

1. `hpc/prepare_git.sbatch` 与 `hpc/submit_git_prepare.sh`
   能在 CPU Slurm 作业中从 fork 取得指定 commit，产生
   `/scratch/user/uqcche38/hu-mcd-git/releases/FULL_SHA/code`。
   旧版 commit f22adf2 的准备作业 28159694 已有成功记录。
   **此次新提交尚未在 Bunya 准备。**
2. `hpc/phase3_smoke.sbatch` 仍然写死旧目录
   `/scratch/user/uqcche38/hu-mcd`，并使用 10/5、ViT-B smoke 配置。
   **不要直接把该脚本当成新版代码或论文规模 launcher。**
   下一步要让 launcher 显式选择 prepared release，复用既有模型/环境，
   并引用新的参考配置。
3. `run_smoke.py` 已接入 `utils/run_tracking.py`。
   Slurm 下要求 `--run-id "$SLURM_JOB_ID"`，输出和缓存分别放到
   对应根目录的 `runs/JOB_ID`；记录 resolved config、Git commit/dirty
   state、选定源码和实际读取图片的 hashes。启动/阶段完成/异常/结束写进度。
   新 producer 已有本地测试，尚无新版本远端完整执行的证据。
4. `hpc/collect_results.py --run-layout` 已支持新目录和身份/哈希检查，
   通过 `--watch` 轮询。现有 profile 仍要求 smoke 的 job name、
   日志命名和结果文件布局。新 launcher 改动这些时，必须一并适配 collector。
   它只用 sacct、squeue、SFTP；不做远程文件处理，不提交或取消作业。
5. 原有数学路径未因本次整理而修改。拟合的 bases、cluster labels、
   segment-to-image mappings、局部 relevance、环境/模型完整身份等额外
   科学记录，仍需按审计文档补齐。
6. `outputs/` 和下载快照留在本地，不在 Git 中。诊断脚本依赖那些历史
   文件，因此它是特定缓存的复核工具，不是 fresh clone 即可运行的实验入口。

## 建议主对话的执行顺序

- 阅读 `docs/PAPER_REPRODUCTION_AUDIT.md` 和 `docs/VALIDATION_DIAGNOSIS.md`。
- 以侧对话最终消息中的已推送 SHA 为起点，检查当前主对话是否又产生新改动；
  若有，不覆盖它们，不假设旧 SHA 包含新改动。
- 完成指定 release 的 launcher、参考配置、collector 路径的衔接，
  先用本地测试验证身份、配置、路径和进度对应。
- 在后续获得运行指示的范围内，用 Slurm 检查 ImageNet 结构、
  准备确定的图片清单和 ViT-H 权重，核对安装的 masking 代码与模型身份。
- 做必要的 feature/logit 重构与评估边界检查，然后安排单类别参考实验。
  新增计算或文件处理均在 Slurm allocation 内完成。
- 根据真实耗时/内存安排资源；不要把 10-image ViT-B smoke 的耗时线性外推到
  ViT-H/400-image。SSC 的 dense coherence matrix 随片段数平方增长。
- 保存并收集日志、manifest、原始科学结果和图。后续再推进
  C-Insertion/C-Deletion、ACE/MCD；人工可理解性研究另列范围。

## 本次验证

- collector / run tracking 的 19 项 unittest 全部通过；
- `run_smoke.py`、tracker、collector、缓存诊断脚本的 Python 语法通过；
- 所有现有 HPC .sh/.sbatch 的 bash 语法通过；
- 提交前检查 whitespace、文件范围与原有结果保留情况；
- 不把本地测试等同于远端部署成功，不把运行 PASS 等同于科学复现成功。


## Medical data preparation update — 2026-09-10

Read `docs/MEDICAL_PILOT_PREPARATION.md` and
`artifacts/medical-pilot-review/STATUS.json` before any medical run.
The older OneDrive archive passed full WSL/Slurm integrity audits (28227929),
but has seven author-confirmed cross-split duplicate pairs and is not approved
as the locked Zenodo release. An exact-release recovery candidate failed the
published MD5; it remains unapproved. A ResNet18 medical checkpoint was downloaded
and passed strict torchvision / installed masked-timm loading; no inference ran.
Do not treat corrected validation as a fresh holdout for this candidate.
No model experiments or medical sample selection were executed. Legacy files,
failed attempts and original ImageNet work are preserved. See DERMA-002–009.

## 2026-09-21 latest medical preparation and protocol decisions

- User accepts Derm7pt observations; say **用 derm 图替代 clinic 图 / replacement of clinic images with derm images**. User will review9cases. No new case_id audit without new evidence; identity limitation retained, no sample/label/ID change.
- Exact current DermaMNIST-C NPZ newly found in Downloads, published MD5 matches; uploaded unique scratch file, no newHTTP. Data-only deployment28738130 and preparation28738133(code822860db132f35a540b079b9bef87d7c156ce8d8)completed. Data28738133:0:0/37s, local acceptancePASS, Q9903archive sealed and scratch paths accepted. Preserve previous failedHTTP504/launcher/accesshistory. MED26-009twoCSVResourceWarnings fixed locally without scientific change/reexecution.
- `artifacts/bunya/next-phase-20260921/medical-decisions/accepted-dataset-paths.json` is the accepted data pointer. Cross-dataset overlap remainsUNKNOWN; internal18knownpairs/0cross-split is NOT external evidence. MED26-008also records publicSelfClean withinDerm7pt evidence, identitiesnotmatched, noautomaticexclusions.
- `docs/DERM7PT_PROTOCOL_DECISIONS_20260921.zh-en.md` is the concrete decision document. Officialtest authorMEL101versus primary100, allsplitprimary248; signcounts include1/1/5smallgroups. R101+Sdescriptive coverage/full12contrasts recommended, NOTAPPROVED. Hdataaudit4CPU8GiB30min0GPU2GiBcap orH0exactonly1CPU8GiB10min await userdecision. No medicalmodelauthorization.
- Source400lesion-firstseed43direction/fixed70test(61lesions)preserved; no samplelistfrozen. StandardResNet50identity/training and medicalGPUbudget stillpending. Aexisting10jobs pending latestsnapshot, noresubmit/nohardwait.


### 已归档提交摘要 / Archived submission summary

## 2026-09-22 实际提交 / Actual submissions

- 完整执行commit：`00f06dc549439db92e4842cb5ecd761f9b38bc43`，已推送fork/reproduction并通过Slurm部署；此前`227eedf5`科学实现与协议保持一致。
- 共同部署 **28781904**：COMPLETED0:0，6秒；实际release/launcher/config哈希及H/M独立可读性/环境检查PASS。请求1CPU/1GiB/5min；调度记录AllocCPUS2、billing1，不是请求扩容（MED26-010）。不等于M GPU适配或医学适用性通过。
- **H：28781914**，已取得sbatch成功回执；4CPU/8GiB/30min/0GPU/2GiB输出。
- **M：28781915**，已取得sbatch成功回执；1L40S/4CPU/16GiB/4h/5GiB输出，50轮/batch64/训练seed43，FP32/TF32和固定配置不变。
- 两者仅`afterok:28781904`，没有H→M或M→H依赖。独立输出`/scratch/user/uqcche38/hu-mcd/outputs/workstreams/<jobID>`；实际plan/config/protocol/launcher哈希及提交回执在`artifacts/bunya/medical-approved-20260922/`。
- 状态仅为**已提交、未收集、未验收**；本轮停止等待实验完成。用户通知后再收集。失败不自动重提或扩容。
- R101/S仅协议批准；医学发现、外部推理仍未授权。M完成后区分工程验收与研究适用性，报告七类/melanoma表现与训练曲线，再提出发现预算；不因低分重训。

Both H and M were submitted independently against the same verified release. Submission is not completion or acceptance. Wait for the user's completion notification before collecting results. R101/S approve protocols only; no medical discovery/external inference. Report class-level/melanoma metrics and learning curves separately from engineering acceptance before proposing discovery resources.
