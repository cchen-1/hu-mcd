# H/M 医学结果验收 / Medical H/M acceptance — 2026-09-22

本轮仅收集、缓存复核及私有可视化；没有新模型预测、拟合、选样、远端计算作业或排除。最终工程凭据：`artifacts/bunya/medical-results-20260922/acceptance.json`。统一报告仍为 `artifacts/bunya/ten-class-review/report/index.html`，新增医学结果与成对审阅入口；旧概念展示及审阅保留。

This round collects, verifies saved evidence and renders private results only. No new inference, fitting, sampling, remote processing jobs or exclusions. The existing bilingual report remains the single entry. Engineering acceptance is distinct from medical suitability.

## 实际执行与身份 / Execution and identity

| 作业 / Job | 最终状态 / Final | Slurm耗时 / elapsed | 实际资源 / allocation | host MaxRSS |
|---|---|---|---|---|
| 部署 / deployment 28781904 | COMPLETED 0:0，复用已有部署验收 | 6s | requested1CPU, allocated2CPU, billing1; 1GiB | 52,280 KiB |
| H 28781914 | COMPLETED 0:0，检索工程验收通过，候选待人工 | 88s | 4CPU / 8GiB / no GPU | 1,376,376 KiB |
| M 28781915 | COMPLETED 0:0，工程验收通过，研究限制保留 | 782s | 1 L40S / 4CPU / 16GiB | 3,350,628 KiB |

- Full executed commit: `00f06dc549439db92e4842cb5ecd761f9b38bc43`; H/M scientific code unchanged from `227eedf5a51c2c8dfd93b03f205e67347f4a248e`. Actual configs, source input hashes, launcher and manifests are collected separately for each job. The later local report script is not misrepresented as executed training code.
- H protocol SHA256: `f6d997ff482216e2917e63ec023ad4eb48ea5befafca3b7a9e70fb27df28baed`; M: `15035163fc088076f537a43b4ab08988b586380641de064675d095262ba70b8b`.
- DermaMNIST-C exact archive SHA256 `f8d80f4fac5a0a1f53256e0c337456a8ab8020828a94a9eb8a1eb8ef5d748486`; accepted source manifest `42a57e7720d11e8c9a8f547f7ae1b96c34832c370ef32ffb2d6c4343fa0623b4`.
- Derm7pt ZIP SHA256 `89a4749e7e43d1c2e73876aaeb867af5b2624e929ee3d1ae7202268588566b54`; accepted cases CSV `17dd2fe063f88d6fcd3d8d07185c81cf2d1fea536a281eb4f70b7ac40298a044`.
- Initializer SHA256 `14fe96d1f9fb311a60490082d2077e6e60427dcfe21839ddf934cce948f72b0f`; selected medical checkpoint SHA256 **`07a6f65dbb23eb7c97510bef09c4b624a1a86007b27a64e324151fd1dfd2a655`**.
- Total CPU time (deployment/H/M): 1.168s / 82.046s / 1,284.498s. M worker elapsed774.324s excludes some Slurm startup/teardown; H worker elapsed83.280s. These measured times are not requested time limits.
- GPU utilization/peak VRAM were not returned by this Slurm accounting record. Host RSS is not VRAM. Both stderr and worker anomaly files are empty; that does not mean research limitations are absent.

## H / MED26-008

计划全部完成：10,015源图 × 1,011 derm图 = 10,125,165图片对，每种hash8方向，共81,001,320方向比较。原尺寸解码像素与224 bicubic规范化精确匹配均为0。NPZ保存数组，原始编码JPEG字节比较不适用。独立使用缓存重算全部距离、阈值、最近邻、方向和顺序，核对5913候选；并用本地已核验同版本NPZ独立复核精确key匹配。没有重新推理或改检索规则。

All planned comparisons completed. Independent cached-distance replay validates the bounded retrieval implementation, not the sensitivity of real-world duplicate detection. No exact native/canonical match was found; cropping, recompression or acquisition differences can escape this test. No patient identity can be inferred.

| 范围 / Scope | 阈值候选对 / threshold pairs | Derm病例 / cases | 源train / val / test 对数 |
|---|---:|---:|---|
| 全部1011 / all | 4291 | 366 | 3769 / 175 / 347 |
| R101 | 214 | 20 | 180 / 12 / 22 |

全部候选5913对，另含1622对仅最近邻补充；R101全部391对覆盖101病例，是算法强制最近邻的结果，不能解释为全部101例可疑。pHash/dHash距离≤6是未校准的操作阈值，不是同源概率。所有5913对都有原图并排和稳定pair ID，R101阈值214对另设优先入口。原始first200队列保留，其规则为metadata行/source ID顺序，不是最相似200对。人工表默认未审阅；不自动删除、不推断患者身份。当前没有确认重叠，也不能宣布外部集独立。

All5,913pairs are available, including1,622nearest-only supplements. R101's391total pairs cover all101cases because nearest neighbors are mandatory, not because all101are suspected duplicates. The priority view contains all214R101threshold pairs without visual cherry-picking. Exact match absence and unreviewed perceptual candidates do not establish external independence. Review cannot automatically authorize exclusion.

## M / MED26-002, MED26-012, MED26-013

50轮全部完成；每轮8215图恰好访问一次，129更新；合计6450更新、410750训练图片访问。验证573图选模，测试1227图只在checkpoint冻结后读取；首次严格最佳validation macro OVR AUC在第10轮：0.9816014213。保留末轮50checkpoint、优化器、RNG和50份实际顺序。训练seed43与未来400病灶抽样seed43用途不同，后者未在M使用。

FP32/no AMP; cuDNN TF32 enabled, matmul TF32 disabled; cuDNN benchmark/deterministic and deterministic algorithms disabled, highest float32 matmul precision. Runtime torch2.2.2+cu121/timm0.6.13/L40S. Seven-class head including bias; saved ordinary/all-ones-mask/head checks passed on the same first8validation images. Local algebra verification reuses saved features, not new predictions. This is not yet proof of arbitrary SAM-mask behavior or medical concept correctness.

| 指标 / Metric | 验证573 / validation | 测试1227 / test |
|---|---:|---:|
| Accuracy | 507/573 = 88.4817% | 1092/1227 = 88.9976% |
| Balanced accuracy | 57.5777% | 59.6981% |
| Macro OVR AUC | 0.981601 | 0.968671 |
| MEL recall | 17/22 = 77.2727% | 50/70 = 71.4286% |
| MEL OVR AUC | 0.968900 | 0.944104 |

Test per-class true-positive/support: AKIEC13/35, BCC29/44, BKL81/105, DF0/8, MEL50/70, NV907/948, VASC12/17. Full precision/recall/F1/OVR AUC/specificity, lesion/image denominators and both confusion matrices are in CSV. MEL precision50/89=56.1798%;39false positives,20false negatives. The70MEL images represent61lesions, not70independent patients. No patient-independence claim or unapproved confidence intervals.

**已观测限制 / Observed limitation (MED26-012):** DF在validation5图和test8图中均零召回，而且整个validation/test均没有DF最终预测。不是全模型恒定输出，也没有证据证明标签错位；DF test OVR AUC0.959393衡量排序，可与零argmax召回并存。类别不平衡和决策竞争可能有关，但原因尚未归因。完整混淆矩阵和身份复核保留。不事后改阈值、重训或改变选模。

**轨迹 / Trajectory (MED26-013):** 第10→50轮train CE0.302823→0.018966，val CE0.300582→0.436411，val AUC0.981601→0.971815；与后期过拟合/泛化退化相容，不证明唯一机制。50轮是预算窗口，不是收敛或最佳训练长度结论。训练accuracy未记录，不虚构。验证数据选出的10轮checkpoint保持冻结，test表现不用于重新选模。

Engineering acceptance passes identity, completeness, finite outputs, selection and sampled adapter/algebra checks. Research applicability is **conditional**: suitable to ask what this particular MEL classifier represents, with its mistakes retained; not a clinically validated classifier, robust seven-class model or seed-stable method. No post-hoc numerical pass threshold is invented.

## 下一批原提案：现已获批D0+D1+部署 / Original next-batch proposal — NOW APPROVED

2026-09-22用户已明确批准本节D0+D1+部署；外部推理仍未批准。执行及具体含义修正见 `MEDICAL_DISCOVERY_EXECUTION_20260922.zh-en.md`。此前“1%离群规则”是文档错误：q=1.0为100%分位点、有限行不剔除，参数不改；“cluster搜索”更正为既有片段数均值+标准差启发式。下面提问保留为批准前历史，不重复询问。

推荐先批准**仅源域医学HU-MCD发现与内部诊断**；Derm7pt外部推理继续单独等待。H候选尚未人工确认不阻止源域发现，但必须限制未来外部独立性解释。DF问题限制跨七类有效性，不自动禁止以MEL为目标的固定模型案例。

Recommended next scope: source-only medical HU-MCD discovery/internal diagnostics conditioned on the frozen classifier. External inference remains separately unauthorized. The unreviewed H candidates affect external independence claims, while zero DF recall limits seven-class generalization; neither is silently dismissed.

| 阶段 / Stage | 明确配置 / Configuration | 建议上限 / Proposed cap | 依赖 / Gate |
|---|---|---|---|
| D0身份与抽样冻结 / identity + sampling | 接受的1021train MEL图/533病灶；排序病灶ID后以独立NumPy RandomState(43)无放回选400病灶，再按选中顺序从各病灶按image ID排序的图片中用同一独立RNG均匀取1图；记录候选顺序、全部角色及哈希。70test MEL图/61病灶全部保留，用于解释诊断，不选模型/概念数。 | 1CPU / 4GiB / 10min / 0GPU / 0.5GiB输出 | **具体RNG/顺序方案是操作性提案，须此次批准，不将先前seed方向批准等同于该完整实现。**无删图/灰度筛选移植。 |
| D1单次完整发现 / one full discovery | 固定epoch10七类R50，MEL label4，global_pool2048含bias重构；400train/70held-out。SAM ViT-H既有hash a7bf3b…262e，points32/minregion256；native224不放大；batch8、FP32及现有TF32；mask/erosion、规范化、SSC既有片段数启发式确定cluster数、outlier_percentile=1.0（100%分位点，有限行无此项离群剔除）、mincluster50、自动维数均复用已验收HU路径，不调参。保留零特征、空区域、SSC原标签/拟合成员、子空间、最终分配、全图及片段特征、预测/重要性与完整mapping。 | **1L40S / 8CPU / 32GiB / 2h / 15GiB输出与新增缓存合计**；无扩容/自动重提 | D0通过；医学7类输入适配须完成本地检查并绑定完整提交；正式作业内先核对同batch全图/全1mask/头部代数，再继续整批；不是额外SAM小样本probe。任何身份/非有限/重构门槛失败即保留证据并停止。 |
| 部署 / deployment | 新完整提交、launcher/config/input/weight哈希校验；复用环境与权重，不装新模型或下载 | 1CPU / 1GiB / 5min / 0GPU | 仅部署；必须计入授权，不继承旧批次预算。 |

**预算依据，不是保证 / Budget evidence, not a guarantee:** 已验收Golden job28208840，400/50、L40S：总2016.4s；train SAM1433.5s、held-out SAM116.5s、SSC18.3s、assignment396.2s；主进程峰值约9.36GiB，子进程峰值约4.51GiB，GPU allocated峰值5840MiB。按50→70held-out线性延伸仅作粗参考，总约2084s（34.7min），不是医学实测。皮肤图的区域数、SSC规模未知；2h/32GiB是提案上限，不能自动扩大。M分类器13min不能估计SAM耗时。D1科学配置未因classifier低分改变；若无概念/高补空间率，照实报告，不自动改参数。

**需要决定 / Decisions requested:** 是否接受“以该冻结模型为条件、保留已知分类错误”的研究范围，并批准上表D0+D1+部署完整上限？备选只批准D0及D1中完整SAM/特征缓存阶段（同一2h上限，拟合暂停），根据真实区域数再定SSC预算；不会重做缓存。第二项独立人工工作：优先审阅R101全部214阈值候选，尤其180训练关联；人工疑似同源记录仍需核实，再决定外部协议是否处理，不能自动排除。

外部R101推理、S统计实际运行、更多训练seed、修正DF/重训、阈值/敏感性比较均不在该发现提案的自动授权中。

## 再现与私有交付 / Reproduction and private delivery

- Generator/acceptance: `scripts/analyze_medical_results.py` (local-only cached analysis); exact script hash and all asset hashes: `bundle/provenance.json`, `bundle/bundle-manifest.json`.
- Collection receipts, final `sacct`, all checkpoints/scientific arrays/logs remain under `artifacts/bunya/medical-results-20260922/`; immutable source results are not rewritten to say PASS.
- Overlay generator: `scripts/update_next_phase_status.py`, existing `ten-class-review/update_workstreams.py`. Updated report publication uses a staging directory and preserves the previous usable report on failure.
- Tables, native pair images, 3120×1104 learning-curve PNG and 3360×1440 confusion PNG (plus vector PDFs) are offline/private. Per-pair user forms are exploratory overlap review, not medical diagnosis or patient identity validation.
