# ACE / MCD 单类别基线准备 / Single-class baseline preparation

**Status: PREPARATION_ONLY — 2026-09-09.** 本线只完成本地源码审计和方案准备。
没有拟合、Slurm 提交、SSH/远端数据读取、采样、提交或推送。执行权限仍未取得。
机器可读方案：[golden_retriever_baselines.plan.json](../hpc/configs/golden_retriever_baselines.plan.json)。

**English summary.** MCD can use the frozen 400/50 identities, but needs distinct
unmasked `layer4` features and its own SSC/bases. ACE needs a derived 50-image
manifest, a 50-image training-only TCAV gradient-test manifest and a resolved
random/control-pool protocol. The released code requests **2,000 random images**;
its 20 CAV runs reuse the first 50 negative identities per concept. SLIC also
omits its highest label. These are confirmed upstream behaviors, not grounds for
silently changing the method. No baseline execution is authorized by this plan.
A/B can proceed independently; C must not duplicate B's Random benchmark.

## 1. 实际证据与依赖 / Evidence and dependencies

本地审阅 `docs/MAIN_CHAT_HANDOVER.md`、`docs/PAPER_REPRODUCTION_AUDIT.md`、
正式验收报告和其固定输入/源码证据。旧交接文档的“尚未部署”等记录是历史，
不能据此判定当前状态。完整结果报告位于
`artifacts/bunya/final-28208840/report/REPORT.zh-en.md`；它确认 Golden
作业 **28208840** 完成、400/50 全部处理、13 概念、completeness 0.67491。
该报告时点 ACE/MCD 尚未执行。**主线随后核查并确认 HEAD eeb27e9、队列为空，
sacct 截至此次查询无基线/评价提交。** 此为主线提供的已查事实，本线未自行SSH查询；
该空队列结论是当时的历史快照，不是本线对当前队列的查询。现已读取主线
Slurm **28214251** 的 inventory、launch_manifest、sacct和collection：CPU作业
COMPLETED/0:0，elapsed49秒，worker PASS，collection COMPLETE，均无记录错误。
已扫描的已知项目路径未发现baseline candidates；不表示其他路径也没有随机池。

| 对象 / Item | 本地确认 / Verified locally | 尚需 / Outstanding |
|---|---|---|
| 科研源码 | `run_ace.py`, `run_mcd.py`, `benchmark_methods.py`, `classes.py`, `concept_explainer.py`, `utils_ace.py`, `utils_general.py`, `utils_mcd.py` 与 upstream `168eb5bf1717ab46d1c84be86bd81f0c51861413` 逐字节相同 | 后续执行需绑定新 wrapper 的确切 commit，不把准备时 HEAD 当作未来执行版本 |
| 准备时 HEAD | `eeb27e9fb31122edbfda1109c69ab7e64b094941` | 无 commit/push 操作 |
| 固定输入 | `hpc/manifests/golden_retriever_seed43.json`，400 train / 50 validation，seed43 | ACE 派生清单尚未选择或生成 |
| 实际 dataset manifest | SHA256 `dafeafcaa64d2379a236500d443e18f8a27520b9a0ad0288e79bdab7c683cd74` | 每次运行继续核对原始 ID、实际输入映射、内容哈希 |
| ResNet50 | timm0.6.13；权重 SHA256 `14fe96d1f9fb311a60490082d2077e6e60427dcfe21839ddf934cce948f72b0f` | 路径以主线 Slurm 盘点为准，不猜测新路径 |
| ACE random/control pool | 28214251在明确扫描的已知项目路径未发现候选 | 其他位置未穷尽搜索；若提供已有池，再验明身份；不自动新建/选池 |
| ACE/MCD / 评价作业 | 主线此前确认无提交；STATUS.json记录B/C submitted=false；已收集28214251 accounting | 该证据不是实时全局队列；本次已知路径未找到可用基线缓存 |

依赖：`主线盘点 + C 协议决定 → 版本化输入/运行器 → 获授权的特征提取 → 拟合 → 基线评价`。
MCD 不依赖 ACE 随机池；A 剩余九类发现和 B Golden 评价不依赖 C 拟合完成。
B/C 只共享已确认的固定输入、模型/精度及评价记录格式；ACE 的随机负样本池与
B 的 **Random 10×10 网格**是两个不同对象。

## 2. 方法专属配置 / Exact method-specific settings

| 项目 / Setting | ACE benchmark | MCD benchmark |
|---|---|---|
| 发现输入 / Discovery | 固定400中的50，派生方案待定 | 原400张全部，顺序不变 |
| 层 / Layer | `global_pool` | `layer4` |
| 分割 / Segmentation | SLIC，`n_segments=[15,50,80]`, sigma1, compactness50；面积严格 >.001，去重 Jaccard严格 >.5 | 每图一个全1 mask；不运行 SAM/SLIC |
| 发现特征 / Features | tight rectangle `cropping_mode=1`，灰色填充.4588，`use_masks=False`, erosion1，未归一化特征 | 全图 `cropping_mode=0`, `use_masks=False`, erosion1；预计2048×7×7；空间向量分别L2归一化，移除全零训练向量 |
| 聚类 / Clustering | kmeans25，random_state43；未归一化 | SSC，tau1/gamma10；L1行范数严格大于.75分位数标记离群，再对inliers重算SSC |
| 簇过滤 / Filtering | ≥50片段、覆盖≥50%发现图片；CAV取距质心最近50片段 | min_size0, coverage0, max_samples=None |
| 概念 / Concepts | sklearn CAV，20轮；TCAV显著性p<.01 | 搜索 **3..19**，FO估维+PCA，每簇维度上限floor(2048/k)，首个completeness **>.5** 停止 |
| 验证 / Validation | 原50 ID和现有RGB输入映射；重新计算SLIC特征 | 同50；未归一化全图layer4特征用于评价投影 |

MCD 独立展示脚本 `run_mcd.py` 从 **5** 开始；benchmark 从 **3** 开始。
论文 §4 说明人类实验至少5概念。因此两者是不同用途，基线评价不得直接沿用
展示脚本默认值。到19仍不达标时保存全过程和 `threshold_reached=false`，
报告受影响结果，不扩大搜索范围或降低阈值。q=.75也不保证恰好25%离群，因严格
比较和分位数并列值；其比例不能直接与 HU-MCD q=1 的0离群比较。

精度沿用 Golden：cuDNN TF32=True，matmul TF32=False，benchmark=False，
deterministic=False，deterministic_algorithms=False，matmul precision=highest。
上游 CLI batch64；本计划**提议**batch8与 B/Golden 一致，但尚未执行/批准。
已有诊断证明跨batch可以产生TF32数值差异，必须记录该执行配置差异，不能声称
与上游batch64逐位等价，也不能通过关闭TF32来暗中改善结果。

## 3. ACE 输入选择：不动 Golden400/50

**建议方案 P：冻结清单训练顺序的前50行。** 此顺序已由seed43的候选排序和
合格图片补齐确定；编号 prepared filename 与该顺序一致。仅在获准后创建新的
`ACE50` 派生清单，记录parent manifest哈希、parent row、原始ID和实际输入哈希。
不重新排序原ID、不改父清单、不因ACE效果挑图。原 `load_class_images` 对编号文件
排序后的前50就是此方案；不会新增数据。

**替代方案 R：仅在原400内部固定算法/版本和seed43重新抽50。** 也不新增图片，
但会更换ACE输入身份和随机过程；没有必要为了“更随机”默认实施。当前未抽样。

**TCAV梯度测试的50图是第三种角色，不是CAV的20%holdout，也不是最终验证50。**
上游从目标训练目录 shuffle 后取50，没有排除发现图。给它只有ACE50的目录，
会导致全部重用这50张；给它原400目录，则可能部分重合。

- 代码参考方案：从原400冻结一次50图shuffle，记录与ACE50重合；保留上游行为。
- 明确的训练内holdout方案：若采用P，提议原400的第51–100行作为梯度测试50，
  与ACE50不交叉；这是上游未强制的输入协议变化，应先决定，不能静默实施。

两种方案都不使用最终validation50调概念或显著性。不要为了分离这些角色
重新分配/缩减 Golden400。独立清单应由manifest-aware wrapper传入；不要靠
覆盖原目录或把不同角色都伪装成一个50图目录。

## 4. 随机池、标签与控制 / Random pool, labels and controls

`compute_tcav_scores(mimic_segm=False,num_rdm_runs=20,max_samples=50)` 请求
`2*20*50 = 2000` 张全图随机样本。CLI“建议至少1000”不符合此实际请求。
加载器不足2000时可能只返回现有数量而不报错；能够运行不等于满足输入合同。
缓存后的可用非零特征数量也必须记录，不能只数文件。

每个保留概念至多50正样本（label1）+50负样本（label0），sklearn
`SGDClassifier(alpha=.01,max_iter=1000,tol=.001)`，默认hinge loss。
100条样本通常分80训练/20测试，`train_test_split(test_size=.2,random_state=42)`，
没有stratify或按原图分组：同一原图的不同片段可以跨该内部划分。模型准确率
仅为CAV二分类holdout诊断，不是最终ImageNet验证准确率。
另有一个随机对照概念（cluster label=-1）：50张随机图作为label1，与随机负样本label0训练。
若25簇都保留，总拟合数为 `(25+1)*20=520`，不是20次总拟合。

TCAV通过**目标类交叉熵梯度**与归一化CAV点积的负号比例计算；不是直接目标logit
梯度正号比例。p值来自 `ttest_rel` 的配对双侧检验，不是独立样本检验。
保存全部CAV权重、内部split身份、准确率、梯度测试图ID、每轮得分/p值和RNG状态。

### 待决定的固定池协议 / Pool decision before fitting

28214251已完成限定范围盘点，已知项目路径没有发现候选池；其他路径未经穷尽搜索。
池路径继续留空，因为没有选定池；以下仍是**提案，不是采样授权**。

1. 总体提议为ImageNet训练split，拒绝任何最终validation身份、重复内容和Golden400
   出现在负池。类别政策需明确：只排除golden_retriever（class-vs-rest），或排除
   全部十个目标synset（适合后续共用，但改变负总体）。不能默认为类别均衡或任意
   已有random文件夹就合法/可比。已有池若不满足所选政策，先报告，不静默修剪。
2. **上游代码参考**：冻结2,000张池；随机对照正50来自同池，保留实际CAV抽取行为。
   对照正样本可能与负样本身份重合；每个概念首轮的负50在后续19轮被重复使用。
   这可作为明确标注缺陷的代码复现，不能解释为20组独立随机负概念的标准ACE结果。
3. **另立改正协议（未授权）**：2,000张负池+不重合的50张随机对照，共2,050张；
   还需决定是否每轮重新从完整池抽50、如何使概念与随机对照的轮次负样本配对。
   此时应另名、另配置/结果，不替换代码参考。已验证的全图特征可跨协议复用，
   CAV/统计量不能直接复用。不能只多准备50图却声称循环和统计问题已解决。

无论选哪种，都保存源ID、synset、split、原图/实际输入SHA256、顺序、角色，
以及逐概念/逐轮正负样本交叉表。格式异常应列出；补样资格和顺序也须预先批准。
当前所有新清单均未生成。

## 5. 异常登记与最小必要检查 / Anomalies and focused checks

完整结构化记录在计划JSON `anomalies`，具有阶段、受影响范围/作业、证据、影响、
处理和状态；主线可按稳定ID `C-001..C-011` 合并统一记录。
除C-010引用共享inventory作业28214251外，`affected_jobs=[]` 指本线未拟合，
不是已有作业都未受影响的结论。

| ID | 发现及证据 / Finding | 处置 / Action |
|---|---|---|
| C-001 | `classes.py:457` 起CAV循环覆盖rdm_acts。AST抽取实际方法，classifier替换成仅记录身份的stub；2,000合成ID，20轮负ID集合全部相同，总共只50个，无拟合 | ACE科研协议待决定，不默改 |
| C-002 | `classes.py:46` SLIC使用range(max_label)。本地skimage0.19.3合成32×32图得到标签1..16，最大标签必漏 | 改循环会改变科研输入，需独立协议；无真实图片分割 |
| C-003 | 随机对照正50与负池共享身份总体，无排除 | 实际重叠未测；拟合前决定并记录 |
| C-004 | TCAV梯度图与发现图同源且不排除；仅ACE50目录会完全重合 | 明确第三清单，最终val不可充当调参控制 |
| C-005 | 配对t检验却未设计共享负组；交叉熵梯度与logit梯度不同；常数得分可能NaN p | 保留估计量定义，记录NaN/常数/收敛警告；受影响解释暂停 |
| C-006 | 图片加载静默跳过，cache键缺输入内容/模型/精度，SSC仅class/count，benchmark pickle身份也不足 | 可做工程wrapper修复：精确输入匹配、独立缓存/输出、验证metadata，不能直接复用同名文件 |
| C-007 | MCD搜索3..19 vs5..19，未达阈值仍会继续下游 | 显式benchmark3..19，保存每步和终止状态；未达标不悄悄扩搜 |
| C-008 | benchmark保留零特征；ACE零向量不匹配概念，HU-MCD通常argmax首概念；全零batch会除零。MCD缺席概念做空均值可能警告 | 非有限值若影响真实轨迹，暂停该结果；只属于被跳过概念的警告记录即可，不停A/B全部任务 |
| C-009 | `basis_of_ortho_complement` 满秩时返回未初始化1×d数组，tuple assert未检查预期条件；MCD矩形特征图坐标公式有风险 | 真实shape/rank/有限值/重构针对性gate；当前预计7×7不触发矩形问题，满秩是否触发未知 |
| C-010 | 28214251 inventory及collection已收齐；指定项目路径没有baseline candidates | 等待inventory已解除；附receipt/SHA，限定结论范围，不自动选池 |
| C-011 | 各方法端点break位置不同，ACE在概念内片段循环判断；严格>75%聚合、删除像素轴用于两方向 | 保存原轨迹/端点/样本数；不补点改指标，图注清楚解释 |

以上明确区分源码事实、合成复现和条件风险。没有推断论文结论错误。
如后续模型输入身份、研究规则或完整性异常，暂停受影响阶段；工程序列化、路径、
缓存身份验证、日志保存问题可局部修复，原历史不可覆盖。新增warning/小异常也写入
统一记录，不用吞掉warning或每个warning都全线停工。

## 6. 与 B 协同及缓存边界 / Coordination and reuse

- B HU-MCD应使用全部 **569** 个验证SAM片段，包含35零特征；534条过滤后的
  prototype数组不是benchmark输入。已完成结果的400/50、基和缓存不改变。
- 三种方法验证特征都保留零行；ACE `get_matching_concept_idx` 对零向量返回None，
  MCD按空间特征，不能为了统一表格强行套用HU-MCD的argmax首概念规则。
- MCD/HU-MCD的benchmark用 `norm_batch=True`；原prototype的False结果不能直接
  当成所有评价中间量。按每图正确边界重算缓存投影即可，不重跑SAM。
- 最终翻转推理共同使用原 `cropping_mode=0,use_masks=True,masking_mode=1,erosion=1`；
  不能改成发现时的−1。保留方法自己的mask生成/排序、complement排除和端点控制流。
  严格`count>.75*50`意味着至少38图有该步。保留实际像素覆盖、每步参与数及逐图预测，
  不通过插值、补全端点或改阈值制造主评价AUC。
- 原SAM mask/掩码global_pool特征/SSC不能给ACE SLIC或MCD全图layer4用。
  原400/50输入、权重、精度和已验证的通用跟踪/收集机制可以复用。
- ACE全图random特征可在两方向/不同CAV配置间复用，条件是**输入、顺序、层、
  预处理、权重、精度和batch元数据**都相同；发现的cropped特征不可冒充全图特征。
- MCD先保存450图的未归一化layer4 maps/logits，训练时另存L2空间向量和行映射。
  3..19搜索共享相同输入的初始及inlier SSC，不为每个k重算，保留k独立labels/bases。
  不将HU-MCD q=1的矩阵/离群掩码用作MCD q=.75缓存。
- 一个经过身份验收的ACE/MCD拟合同时服务C-Insertion和C-Deletion，不各方向重新拟合。
  B已完成或待完成的Random轨迹/预测在完整输入/预测签名相同后共享，默认不再跑Random。
  随机负池不需要Random网格结果，MCD不需要ACE负池。

## 7. 资源提案与执行门槛 / Estimated budgets and gates

以下均是**规划估计，不是已测耗时、不含排队、未获执行授权**。不以Golden
SAM耗时线性推算MCD SSC；没有安排任何SAM小probe。

| 阶段 / Stage | 提议申请 / Proposed allocation | 预计计算 / Estimate |
|---|---|---|
| ACE SLIC、50train/50val特征、2,000随机全图特征、50梯度图 | 1 L40S / 4CPU /16GiB /1h | 10–40min；SLIC每split最多约7,250候选，去重后变少；分batch提取 |
| ACE kmeans/CAV/TCAV | CPU4 /16GiB /1h，无GPU | 5–30min；至多520个小CAV，复用上阶段特征/梯度 |
| MCD450图layer4 | 1 L40S /4CPU /16GiB /30min | 2–10min；batch8 |
| MCD SSC两阶段及3..19搜索 | CPU8 /64GiB /4h，无GPU | 30–180min，未测且不保证该范围；最多19,600空间向量 |
| 将来ACE+MCD双向评价 | 1 L40S /4CPU /16GiB /1h | 10–40min；粗上限约2,600+2,000前向状态，不重复B Random |

19,600² coherence仅单矩阵就约1.43GiB(float32)/2.86GiB(float64)，还存在
输入副本、每worker `np.delete`、solver和谱/PCA内存；64GiB是保守申请，不是峰值实测。
限制joblib/BLAS线程为allocation，CPU拟合与GPU特征分开能避免SSC期间占用GPU。
若简化成每方法一个GPU作业更易部署，会在CPU阶段浪费GPU；不推荐默认这样申请。
超时/内存不足时保留已有缓存并报预算，不自动加资源/重提。

每个未来作业必须绑定execution commit、完整resolved config、父/派生清单hash、
权重/安装masking代码hash、环境、精度、batch及RNG状态；独立JOB_ID输出和cache，
保存阶段进度、失败attempt、异常、原始特征/行映射、拟合中间结果、重构结果和
本地collector receipt。正式评价前只做实际新路径必要检查，不重复通过的SAM probe。
计划JSON的路径和execution_commit留空是故意的，不可当成可执行config。

执行门槛：已收集的限定范围inventory和历史提交记录用于去重 → ACE协议及清单决定（仅影响ACE）→ 指定阶段
资源授权 → 固定代码/输入/输出配置 → Slurm执行。C当前只交付准备文档和JSON，
没有额外readiness脚本：当前未知的是池身份与科研决策，重复造一个无具体输入的
检查器并不能解决这些问题。

## 8. 主线需要用户判断的少数事项 / Decisions for the coordinator

1. **ACE协议包**：建议发现图先选冻结前50；同时决定TCAV训练内控制是否分离、
   负总体排除哪些类，以及保留已记录缺陷的上游代码参考还是另做改正协议。
   这是明确的科研/输入选择，不是“是否继续”的笼统确认。没有答案时只暂停ACE拟合。
2. **分阶段运行预算**：在盘点证实输入/缓存状态后申请上表具体任务；MCD可独立
   获准执行，不等待ACE池。若暂不投入基线拟合，先完成A/B并保留本准备方案即可。

English decision note: choose the ACE input/statistical protocol explicitly before
fitting. Then authorize the staged resource requests that remain necessary after
inventory. MCD is independent of ACE's random-pool decision. Keeping the released
bugs for a labeled code-reference baseline and correcting them in a separate
scientific variant are different choices; neither is silently selected here.

## 9. 主线 inventory 元数据合同 / Required inventory fact schema

计划JSON内 `inventory_receipt_schema` 为 JSON Schema draft2020-12，
`inventory_receipt_rules` 规定证据含义。共有三层：

- **执行证据**：Slurm job ID、compute node、开始/结束时间、worker commit/SHA、
  config SHA；scheduler另记录观测时间、account/用户、sacct查询起止范围、具体
  queries、队列和基线/评价job IDs、原始receipt文件大小/SHA。
- **检索范围**：展开的绝对roots（ROOT/data、ROOT/cache及选定legacy ROOT路径）、
  patterns random/acts/selfrepr、是否递归/跟随symlink、深度/数量/时间限制、排除路径
  和scope_complete。空结果只证明声明范围内未发现，不能外推整个scratch。
- **每个发现**：path/resolved_path/kind/exists，证据级别name_only/metadata_only/
  manifest_verified/content_verified；明细artifact及SHA；数量（文件、图片、可用
  解码图、特征行、零行、重复ID/内容、与Golden400/50重叠）、synset/split分布；
  producer job/commit、输入及有序行映射SHA、权重、层、预处理、精度、batch、
  clustering q/normalization、array shape/dtype；兼容结论及理由、尚需用户政策。

初次仅扫名称/metadata可把未测字段写 **null**；null不是0。无需为填全表强制新增
昂贵特征计算。未验明身份的缓存只能UNKNOWN/CANDIDATE，不能VERIFIED；
同名random路径不是已选负池。图片解码、hash、数组读取均在receipt的Slurm作业内，
不要为inventory直接反序列化不可信pickle。每个权限错误、损坏、warning、截断及
未验字段均应显式记录，在本地收集receipt后关联统一异常记录。

Inventory28214251 is now collected: COMPLETED/0:0,49 seconds elapsed on CPU,
worker PASS and collection COMPLETE. No baseline candidates were found at the
explicit known project paths. This does not exclude pools elsewhere. The earlier
empty-queue statement is historical; STATUS.json records B/C not submitted but
is not a raw global sacct/squeue transcript. No new schema or check was added.

交接时主线基础设施已推进至 `94369b79a78ca534433da622268c3a9afea9d293`
（准备时本地HEAD核对），Slurm **28214175** 部署记录COMPLETED/0:0；
共享inventory及A输入审计 **28214251** 已完成并收集。
本线未提交或部署本文件；未来baseline execution commit仍为空。
本次仅把已收集的inventory事实接入文档，未运行新任务或新增检查。JSON解析和八个科研源码hash复核通过；本地没有
JSON Schema meta-validator，因此不声称完整schema校验通过。

## 10. 已收集 inventory 证据 / Collected inventory evidence

作业 **28214251**，commit `94369b79a78ca534433da622268c3a9afea9d293`，compute node `bun155`；
开始 `2026-09-09T05:03:37.803828+00:00`，结束 `2026-09-09T05:04:23.620705+00:00`。
Slurm elapsed49秒是CPU作业的墙钟时长，不是TotalCPU实测值。

范围：`/scratch/user/uqcche38/hu-mcd` 及其 `data`、`cache` 的七个明确名称：
`random`, `random_acts`, `acts`, `acts_mcd`, `segms_val`,
`self_repr_matrices_mcd`, `self_repr_matrices_humcd`。另外列出data/cache直接子项，
读取`outputs/runs/*/summary.json`和`launches/*/launch_manifest.json`。
不是递归搜遍scratch或所有项目。worker只有发现匹配路径时才写`baseline_candidates`，
因此本次该字段缺省对应这组已知路径没有候选。Golden28208840的PASS/原commit得到再次记录。

以下路径相对仓库根；哈希针对实际收集文件。JSON `inventory.evidence` 还包含
字节数、artifacts及空errors/log证据；不要求这份旧worker输出满足后来准备的schema。

| Evidence path | SHA256 |
|---|---|
| `artifacts/bunya/workstreams-20260909/collected/28214251/20260909T050428.482463Z/inventory.json` | `59e03a84268c20e350ebb92c0fe37f3b28499d525132184df8ab4b1793bd7d25` |
| `artifacts/bunya/workstreams-20260909/collected/28214251/20260909T050428.482463Z/launch_manifest.json` | `74a82947955d84c0b9cd5b72eacb0d37d791349d56db811f2b80bc45b333998b` |
| `artifacts/bunya/workstreams-20260909/collected/28214251/20260909T050428.482463Z/sacct.psv` | `85ff15c3dc2cc6bdfa9ede0742a9f6ddc31a494ddf1ef4549d48012e9abb3222` |
| `artifacts/bunya/workstreams-20260909/collected/28214251/20260909T050428.482463Z/collection.json` | `de28226a0caa518cb7af91d42b80da9e1383f3bd1cdae5544077e7a964fc0180` |
| `artifacts/bunya/workstreams-20260909/collected/28214251/20260909T050428.482463Z/actual_config.json` | `73dffddbb7a119f9d650bbd605b3bc43afcaf6da60e12c5b7a74d79fc451ff4e` |
| `artifacts/bunya/workstreams-20260909/STATUS.json` | `b71c0cef3b7b5b1bf3c1e36fb131877aa92f7b584ccdae605f9c4c8dae310e67` |
| `artifacts/bunya/workstreams-20260909/initial-deploy.sacct` | `66113a4189f41c569e4532df29b1eb629bd4622345c9ff2c794b7c81e4cc74bf` |
| `artifacts/bunya/workstreams-20260909/A-input-audit.submission.json` | `14691618501bbc94f642978af710f77e4de590bc26269c70593fba3818f044cc` |

状态结论：**inventory等待已解除，ACE池选择及科研协议决策仍未解决**。
本次盘点没有为新负池提供自动采样授权，也没有验证任何ACE/MCD拟合缓存可直接复用。
