# 当前状态与集中决定 / Current state and decisions

<!-- APPROVED-20260922:BEGIN -->
## 2026-09-22 最新批准 / Current explicit authorization

用户明确批准 **H＋R101＋S描述性＋M50/batch64/分类器训练seed43**，包含原部署预算。此批准仅在对应范围内替代下方历史提案的“未批准/待选择”；保留历史与未被替代的通用规则。主计划 `NEXT_PHASE_MAIN_CHAT_PLAN_20260921.zh.md` 继续定义总体目标和工作线边界，较新推荐本身不是授权。

| 稳定ID / Decision | 现批准及替代旧项 / Approved scope and supersession | 上限与排除 / Caps and exclusions |
|---|---|---|
| MED26-008/H | 完整H精确/相似候选检查获准，替代H/H0待选状态 |4CPU/8GiB/30min/0GPU/2GiB；不是H0，不删图 |
| MED26-002/R-MEL | R101官方test作者MEL分组，含1metastasis；替代R101/P100/E248待选 |仅评价范围协议；无外部推理授权 |
| MED26-002/S-GROUP、S-COVER、S-12 | 作者分组＋原标签、并集覆盖、全部概念×12描述性比较；替代对应待选 |仅分析协议，不含S+bootstrap、确认性p值、医学概念发现或外部推理；未来缓存表统计预算仍随该阶段批准 |
| MED26-002/M-SEED、M-EPOCHS | 单个分类器seed43、50epochs/batch64及完整M配置获准 |1L40S/4CPU/16GiB/4h/5GiB；不自动延长/重训，不改变采样seed授权 |
| Shared deployment |一次共用明确commit的CPU部署获准 |1CPU/1GiB/5min/0GPU；分别检查H/M，不创建二者之间的依赖 |

**M验收边界：** 完成后分别报告工程验收和研究适用性；报告七类及melanoma表现、训练/验证曲线，再提出后续发现预算。数值通过不表示医学适用，低分不触发自动重训。失败不自动重提、预算不扩容。提交后交接jobID/commit/config即停止等待，用户通知完成后才收集H/M结果。

The user explicitly approved H and M execution with the stated deployment budget. R101 and descriptive S are approved **protocols only**, excluding medical concept discovery and external inference. The main plan remains authoritative for unchanged scope/rules. Newer recommendations are not permissions. Separate engineering acceptance from research suitability; report class-level/melanoma performance and learning curves before proposing discovery resources. No automatic retry, expansion or low-score retraining.

Actual authorization, config hashes and submission receipts: `artifacts/bunya/medical-approved-20260922/`. Approved configuration copies: `configs/medical/*.approved.json`; historical `.proposed.json` are preserved. The actual resolved job config and full execution commit are additionally bound in each launch plan. Status below this block reflects historical proposals until specifically superseded here.
<!-- APPROVED-20260922:END -->


2026-09-22 Brisbane. Current source of truth: scheduler snapshot, acceptance receipts and explicit user decisions, reconciled in `artifacts/bunya/next-phase-20260922/authorization-ledger.json`. Historical failed jobs remain evidence; they do not supersede later successful acceptance.

| 类别 / Category | 当前事实 / Current facts |
|---|---|
| 已验收可复用 / Accepted reuse | 十类HU发现与HU+Random双向评价；GoldenMCD、ACE released-code-reference发现及评价。DermaMNIST-C28738133与Derm7pt28734478数据准备。依据final-20260917/delivery.json及各数据acceptance.json，未重复运行。 |
| A实际执行 / Actual A execution | 已有十作业全部结束：6COMPLETED0:0，4FAILED1:0（28731217airliner、28731220hummingbird、28731223container_ship、28731228beach_wagon）。尚未收集/验收结果，原因未定；原预算和失败历史保留，不重提。COMPLETED is not scientific acceptance. |
| 已明确批准 / Explicitly approved | A五条件、固定500图及5687区域；每类1L40S/4CPU/16GiB/20min、最多并发2，失败不重提。B400不同病灶seed43各1图方向，待模型/预算满足再冻结；源域70测试图全部保留，考虑61病灶相关性。两个数据准备授权已完成。 |
| 用户审阅 / User review | 9例基础视觉审阅已记录：正常显示、无明显损坏/无关内容；皮肤镜外观仅非专业印象。不覆盖标签/身份/重叠，不代表全部1011例；无删改要求。录入9月22日，实际审阅日期未提供。 |
| 本轮可直接完成 / Unblocked preparation | 已记录审阅、整理版本/验收/授权、更新报告；H/M worker与S纯计算函数已准备并完成本地软件检查。没有新医学实验提交。 |

<!-- DECISION-RATIONALE:BEGIN -->
## 决策依据补充：推荐不是批准 / Rationale, not approval

**用户再次明确H／R／S／M均未批准。本轮包含已授权代码/配置/软件检查准备，没有实验。** 稳定ID沿用父决策并细分；下次在本表和Obsidian对应块内更新，不另起重复记录。

| 稳定ID（前缀MED26-002，H为MED26-008） | 已核实原配置 | 当前提案为什么可能适用；证据边界 | 替代与资源／结论影响 |
|---|---|---|---|
| **R-MEL** | 作者loader把原位、不同厚度、未细分及转移性melanoma合为MEL。官方test101例，去掉1例转移性后100。 | 用可追溯的作者类别界定“源域melanoma概念的外部关联”，避免额外结果导向排除。**合并规则有直接证据；适合本问题是设计判断**，不是亚型等价或HAM源域组成相同的证明。 |100例明确研究非转移性，成本几乎不变、范围变窄；分亚型更细但样本小；248例跨划分增大探索范围，不等同独立官方test确认。 |
| **S-GROUP** | 作者既有原始细标签，也有可选GroupInfrequent分组；血管、色素沉着、消退结构进一步合并，原标签可保留。 | 以既有类别体系组织首次病例级分析，少些碎片化；**采用哪套标签仍是选择**，不能证明医学同质或样本充足。REG是作者标签，不自动等于正常／良性。 |保留原细分类：推理成本不变、分析单元更多且更稀疏；7个预先定义的临床二值对比是另一个问题，不能把regular默认为absent。R101里仍有1／1／5例小组。 |
| **S-COVER** | 项目B3提出的候选响应；未核实到作者规定的医学覆盖指标。Derm7pt提供病例级征象，不是真实征象像素面积。 | 每病例计算“归入概念的有效mask并集面积÷输入图面积”；并集避免同概念内重复计数，病例汇总避免把片段当独立样本。**计算性质有依据，语义有效性尚未验证**。 |出现与否、预先固定的activation聚合回答不同问题；病灶面积归一化需可信新轮廓。多数可复用后续缓存；覆盖受构图、背景、病灶大小、mask/分配/零特征影响，不等于重要性或真征象面积。 |
| **S-12** |不是论文原协议。五项三分类各与absent比较2次＋两项二分类各1次，**5×2＋2＝12**。 |预先展示完整矩阵，减少事后只挑漂亮对应；避免无序标签0/1/2相关。**12由所选规则推导，absent作为参照是操作性选择**；未直接比较regular与irregular。 |7个预定义临床二值对比、全部两两17项／概念、或只展示原分布。通常不新增GPU推理；改变分母与问题。K概念即12K个相关比较，不是12个独立临床标准；描述性结果不能变成显著性证据。 |
| **M-SEED** |作者启动脚本循环3次，未在所查训练脚本发现显式seed设置／传参；不能反推其确切随机状态。 |单次分类器适合**以这个拟合模型为条件的首个案例**，降低初始成本。没有证据说明训练seed43代表性好或一次足以支持稳定性结论。 |3个预先指定训练seed：按原提案上限最多12 GPU小时，而非4；后续概念稳定性还需分别拟合／评价，成本不止分类器3倍。不得选测试最佳seed。 |
| **M-EPOCHS** |作者默认100epochs、batch128、Adam1e-3，50/75%处降LR；M另改了标准预训练骨干、lr和batch64。 |**50只是未验证的有界工作量选择**，无学习曲线证明其最优、充分或收敛；预训练也不能证明“所以50够”。4小时是建议上限，不是实测依据。 |同M配置100epochs：12900steps／821500图次，50为6450／410750；预算另定，不能承诺恰好双倍耗时。预先定义early stopping需新增patience等规则。作者100×batch128约6500steps接近M的6450，也不能证明等价训练。 |
| **H（MED26-008/H）** |已有验收只证明各自内部检查；没有已验收跨数据集匹配记录，也未验证作者对此组合规定pHash/dHash阈值6。 |解决外部评价前的具体身份风险。全10015×1011检查因为分类器用全部源类别。**需要补证据有依据，算法、阈值和30分钟是操作性提案**。 |H：4CPU/8GiB/30min/0GPU；H0：1CPU/8GiB/10min仅精确，漏掉近重复；其他检索方法另需预算。阈值未校准，不匹配不证明独立；原5–15min估计不能当实测或充分性依据。 |

**两个43的授权不同：** `discovery_sampling_seed=43`属于已批准的400病灶选样方向（MED26-002/D11，实际清单未冻结）；`classifier_training_seed=43`是M的**未批准提案**，控制分类头初始化、训练顺序、增广。两者需分离RNG状态，不能因训练调用改变选样；相同／不同数字都不自动证明实验独立。训练用全部8215图，不是400图；bootstrap也不等于重复训练。

**状态：本表所有H/R/S/M子决策均未批准。** 我建议将作者提供的“分类规则”与本项目的“评价规则”和“预算缩减”分开决定；首轮推荐明确为解释一个固定分类器的条件性案例，不验证跨seed稳定性；50epoch为固定预算窗口，不声称已充分。

已核实来源与位置（本轮复用固定提交副本，线上固定URL刷新失败，未据此声称新在线核实）：

- Derm7pt `ce436877573c0a53cfa0d224bda20d9479b795aa`：`dataset.py:47–49,60–106,474–508`；本地 `artifacts/bunya/next-phase-20260921/sources/derm7pt-loader.py`。
- Corrected Skin Image Datasets `ac655ac5a5b93264c57d69d6d520778d915bfc89`：`train_and_eval_pytorch_corrected.py:24–26,115–116,152–158,256–266`、`run_224.sh:10–20`；本地`sources/correction-train.py`、`correction-run224.sh`。
- 范围与征象计数复用`medical-decisions/candidate-scopes.csv`、`raw-sign-counts.csv`、`proposed-contrasts.csv`；不重新采样、审计或计算模型输出。
- 每项英文依据／影响／替代／状态已合并进Obsidian的`MED26-HRSM-RATIONALE`块；来源哈希及机器记录在`artifacts/bunya/next-phase-20260922/rationale/`。

<!-- DECISION-RATIONALE:END -->

## 仍待选择，不能从推荐推定批准 / Pending choices

| 项目 / Item | 推荐与实数 / Recommendation and actual counts | 资源与依赖 / Resources and dependencies | 替代 / Alternative |
|---|---|---|---|
| **H 图片重叠检查 / Image overlap** | 全部10015source×1011derm；精确及相似候选，候选人工确认，不自动删图 / exact+perceptual candidates, not automatic identity claims | **4CPU/8GiB/30min/0GPU/2GiB输出**；估计5–15min，未实测；数据依赖已满足。本地实现/软件检查已完成，需批准后绑定新release，失败不重提。 | H0仅精确：1CPU/8GiB/10min/0GPU；或暂缓。H0不能排除近重复。 |
| **R 外部病例 / External cohort** | 官方test作者MEL分组**101例**，含1metastasis，保留原始亚型；9例中case827进入此候选范围。不因其同图引用删图。 | 本轮冻结协议不耗GPU、不授权推理；实际外部推理待模型/概念基及独立预算。 |100例明确排除metastasis；或全部划分248非转移性病例，仅探索、不称官方test独立确认。 |
| **S 概念与征象 / Concept–sign analysis** | 作者类别合并＋保留原标签；每病例概念有效区域并集覆盖；全部学习概念×12项预定非absent-vs-absent描述性对比；定向AUC、分布、分母，无p值/显著性声明。 | 规则选择不授权推理。未来基于已生成表格的描述统计拟纳入外部评价的CPU收尾，独立建议上限1CPU/4GiB/10min，仍待该阶段授权；无实测。无结果不计算。 | 保留原类别仅描述分布；或另选S+病例bootstrap2000/seed20260921，新增建议上限2CPU/4GiB/20min，未批准。患者独立性仍未知。 |
| **M 七类医学ResNet50 / Source classifier** | 全部**8215train**，**573val**选模型，冻结后**1227test**；标准R50+固定ImageNet初始化，seed43、50epochs、batch64、Adam1e-4、固定分段LR与原FP32/TF32设置 | **1L40S/4CPU/16GiB/4h/5GiB**，另部署1CPU/1GiB/5min。仅候选上限，医学训练未实测。具体方案见下；专用实现及本地软件检查已完成；未部署/未GPU验证，批准并绑定release后才提交。 | 暂不训练；提供可追溯匹配checkpoint。R18/作者customR50不是自动替代。 |

S的病例计数（R101）：pigmentnetwork absent31/typical12/atypical58；veil53/48；vascular67/regular18/irregular16；pigmentation38/regular1/irregular62；streaks41/regular1/irregular59；dots12/regular5/irregular84；regression53/48。小组1/1/5例应明确展示，不能因此事后合并。These are label counts, not concept results. The12contrasts have different denominators; source70images are a different dataset and are never pooled here.

**依赖 / Dependencies:** H独立于M/R/S；M不需要先决定外部R/S；医学概念发现需先验收模型并批准SAM/SSC预算，再冻结400病灶清单；外部应用需模型、概念基、H冲突处理、R/S及执行预算。批准H或R/S不等于批准M；批准M也不等于批准医学概念发现/外部推理。已批准的A不重复询问，不自动重提其失败作业。Current queue is empty; there is no remaining unsubmitted GPU task with complete authorization.

- M详细可审阅方案：`docs/MEDICAL_RESNET50_PROPOSAL_20260922.zh-en.md`；机器配置 `artifacts/bunya/next-phase-20260922/M-resnet50.proposed.json`。
- H/R/S细节及来源：`docs/DERM7PT_PROTOCOL_DECISIONS_20260921.zh-en.md`；原计数CSV复用，没有重选队列。
- 本轮审阅记录：`artifacts/bunya/next-phase-20260921/derm7pt-review/user-review-20260922.json`、原`manual-review.csv`。
- 私有报告仍为 `artifacts/bunya/ten-class-review/report/index.html`，没有新建第二套入口。

选择时可直接回复是否批准下方推荐组合，或仅指出H/R/S/M中要改的项目；代码准备不再作为科研选择项。选择尚未收到前，一律保持PROPOSED_NOT_APPROVED。提交后给出jobID、版本、资源和依赖即停止等待，用户通知完成后再收集验收。

<!-- FINAL-PROTOCOL:BEGIN -->
## 最终推荐、三项澄清与验收 / Final recommendation, scope and acceptance

**H／R／S／M仍未批准；本次批准的是代码、配置及必要软件检查准备。没有提交作业、读取远端医学图片、训练模型或计算医学结果。**

**1. 首轮目标 / First-round target — MED26-002/M-SEED.** 推荐将首轮定义为解释**一个固定医学分类器**的可追溯案例。全8215图训练分类器，573图按既定规则选出一个checkpoint并冻结；400不同病灶各1图是之后解释该模型的概念发现集。单训练seed43适用于“以此checkpoint为条件”的发现、预测干预及外部图片级关联，**不用于验证跨训练seed稳定性，不估计方法平均表现或训练方差**。分类器seed43仍待批准；选样seed43仍沿用已批准D11，两者独立RNG，不因数字相同合并授权。

The proposed first round explains **one fixed medical classifier**, not cross-training-seed robustness. Conclusions are conditional on that checkpoint, input protocol and fitted concepts. Training uses all8215 images; the later400-lesion sample has a separate explanatory role. Classifier seed43 remains unapproved and separate from approved discovery-sampling seed43.

**2. 工作量 / Workload — MED26-002/M-EPOCHS.** 保留最后一个不满batch（drop_last=False），每轮每张训练图恰好访问一次：

| 配置 / Configuration | 每轮更新 / Updates per epoch | 训练图次 / Training image visits | 总更新 / Total updates | 验证图次 / Validation visits | 最后batch / Tail |
|---|---:|---:|---:|---:|---:|
| 推荐50epochs／batch64 |129|410,750|6,450|28,650|23|
| 比较100epochs／batch128 |65|821,500|6,500|57,300|23|

后者训练图次和验证前向次数均为2倍，但只多50次更新（约0.78%）。更新次数接近**不等于**梯度噪声、BatchNorm统计、LR轨迹、耗时、显存或收敛等价。作者还使用不同stem、初始化、lr和三次重复；此表仅比较这两个工作量，不宣称完整训练配方可比。50轮仍是预算性窗口，**不是已验证最优或充分长度**；不继续为该数字追加依据搜索或试验。

The100/128 schedule doubles image visits with only50additional updates. Similar update counts do not make optimization or wall time equivalent. Fifty is a bounded operational window, not proven convergence or optimality; no extra experiment is proposed to justify it.

**训练后规则 / Post-training rules:**

| 观察 / Observation | 验收及后续处理 / Acceptance and handling |
|---|---|
| 完成50轮、8215×50图次/6450更新，绑定/行对应/有限值/独立普通模型与mask模型同batch重构检查通过 | 保存全部epoch轨迹；按573validation macroOVR AUC严格改善、同分选更早模型，先冻结SHA，再最终评估1227test一次。标记“完成待验收”；数值通过不等于临床适用或概念可解释。 |
| 超时、OOM、未完成计划、身份或行错配、非有限训练/选择分数 | 保留最后完整checkpoint、日志及失败记录；未完成不能充作已完成。不自动重提、改batch/精度、恢复或延长。定位后只提出具体修复/续跑选择。 |
| 同batch普通／全1mask／线性头重构门槛失败 | 保留数组，阻断该模型下游概念路径；本实现同时不继续最终test，避免把未通过新路径检查的模型当已验收；不放宽容差或自动再训练。 |
| 普通误分类、低分、验证曲线尚在改善或训练/验证差距大 | 如实报告学习曲线和已选epoch；不单凭它们认定代码错误，不事后换指标或增轮。即使50轮不足也先提交证据给用户判断。没有事先定义的“好分数即通过”阈值。 |
| 所选checkpoint在完整val或test只预测一个类别 | 记录实际分布并暂停该模型的下游验收，诊断是优化/类别不平衡还是实现问题；现象本身不是已确认代码bug。不据此选择另一个checkpoint。 |

Completion is distinct from acceptance. Low performance or incomplete apparent convergence is reported, without tuning or automatic extension. Identity/nonfinite/adapter failures block affected use; final argmax collapse requires review, not an automatic refit. No performance threshold is invented after seeing outcomes.

**3. 一句研究问题 / Research question — MED26-002/S-COVER、S-12:**

> 对于一个冻结的医学分类器及其固定概念，在预先确定的外部MEL病例中，各概念的图片覆盖率在某征象的指定非absent类别与absent类别之间是否呈现可描述的分布差异？
>
> For one frozen medical classifier and its fixed concepts, do image-level concept coverage distributions differ descriptively between each prespecified non-absent sign category and its absent reference in the predefined external MEL cohort?

概念覆盖是有效分配mask的并集面积/整张输入面积，不是真实征象面积。12项是5×2＋2个**预定类别对absent**的描述性比较；K概念即12K相关比较。提供分布、实际分母和固定方向AUC，不翻转负向结果、不计算确认性p值。它们最多提供**图片级关联证据**，不证明概念与征象语义等同、征象空间定位、因果关系或患者独立性；1/1/5例小组照实保留。缺失不是零，补空间单列。

**最终推荐组合 / Recommended package:** **H＋R101＋S描述性＋M50/batch64/训练seed43**。H检查身份风险；R101沿作者MEL范围保留转移性病例原标签；S先以完整描述矩阵回答上述有限问题；M建立一个可固定、可追溯的解释对象。选择它的理由是研究问题与预算一致，**不是参数已被证实最优**。

| 待明确批准 / Decision | 具体选择和上限 / Selection and ceiling | 不包含 / Excluded |
|---|---|---|
| **H / MED26-008/H** |10015×1011图片，原生/224canonical精确＋pHash/dHash64、Derm的D4检索、阈值6和每hash最近候选；4CPU、8GiB、30min、0GPU、2GiB输出。哈希算法/顺序已在配置明示，全部候选保留，首200只作审阅队列。 |不删图、不推断患者身份；H0替代仅精确1CPU/8GiB/10min。运行时间均未实测。 |
| **R101 / MED26-002/R-MEL** |官方test作者MEL101例，含1转移性；保留原亚型。若研究范围明确排除转移性可选P100。 |只冻结范围，不执行外部推理；不是疾病亚型等价假设。 |
| **S / MED26-002/S-GROUP、S-COVER、S-12** |作者分组＋原标签、mask并集覆盖、全部学习概念×12描述性对比，无bootstrap/确认性p值。未来缓存表格统计建议1CPU/4GiB/10min、0GPU，待该阶段一起授权。 |不授权医学推理，不默认S+、显著性或空间定位；S-raw仅分布为替代。 |
| **M / MED26-002/M-SEED、M-EPOCHS** |按完整配置：标准预训练R50、全8215train/573val/1227test、单训练seed43、50轮/batch64、Adam1e-4、既定FP32/TF32；1L40S、4CPU、16GiB、4h、5GiB。 |不含医学SAM/SSC、400图发现、外部推理或额外seed；100/b128需明确更改训练方案及预算，不能自动替代。 |
| **共用部署 / Shared deployment** |若H或M获批，拟共用1CPU/1GiB/5min、0GPU一次，复用同一新代码release；与现有部署流程一致。 |没有提交；批准后仍先核对相同任务是否已提交及实际release/launcher哈希。 |

H与M可以独立推进；R/S选择不阻塞M。医学概念发现与外部执行仍须模型验收及单独预算，不把本组合变成后续全链路无限授权。当前资源均为**上限，非实测**。待批准项只有H、R、S、M这四项；无需重新批准已确认数据和400病灶选样方向。

**准备状态 / Preparation status:** 新增`hpc/medical_protocol.py`、`medical_overlap.py`、`medical_classifier.py`、`medical_sign_analysis.py`，接入既有Slurm launcher/runtime。`configs/medical/*.proposed.json`仍是NOT_APPROVED；execute在建立SSH连接前验证批准及协议SHA，worker再次验证批准与Slurm计算节点。训练checkpoint拆为model/optimizer/RNG组件，兼容现有256MiB单文件collector上限。没有下载权重、读取医学图像或调用模型。

本地18项针对性软件检查通过（10项新增，8项既有回归）；范围包括合成数据/掩码、资源授权门槛、图次/LR、选择与test门槛、缺失/零/对比方向等。初次Python3.9 `int.bit_count`兼容错误已修正并保留失败日志。**这不等于Bunya/GPU训练或模型适配已验收**；GPU适配检查嵌入未来获批的同一个M作业，不加额外SAM probe。已固定本地源码提交`227eedf5a51c2c8dfd93b03f205e67347f4a248e`，未推送/未部署；两份launcher仅生成预览、未提交。具体源码哈希、检查日志与提交状态见`artifacts/bunya/next-phase-20260922/preparation/`。

<!-- FINAL-PROTOCOL:END -->
