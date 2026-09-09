# A/B/C execution report / 三线执行报告

**2026-09-09, Brisbane. A: nine formal discoveries submitted exactly once, two running and seven resource-dependent at the recorded snapshot. B: completed and independently accepted. C: preparation complete, no baseline fits submitted.** This is a time-stamped handover, not a claim that all nine discoveries or the full paper reproduction have finished.

**A 九类均已各提交一次，快照时两类运行、七类等待资源依赖；B 已完成并验收；C 完成准备、未提交拟合。** 以下区分已测结果、进行中工作与需决定的协议。

## Execution / 执行

| Work | Class | Job | Code | Snapshot state |
|---|---|---:|---|---|
| A | hummingbird | 28214893 | `982045f` | RUNNING |
| A | Siamese_cat | 28214894 | `982045f` | RUNNING |
| A | tailed_frog | 28214896 | `982045f` | PENDING |
| A | container_ship | 28214897 | `982045f` | PENDING |
| A | airliner | 28214983 | `982045f` | PENDING |
| A | police_van | 28214984 | `982045f` | PENDING |
| A | beach_wagon | 28215019 | `f73216b` | PENDING |
| A | zebra | 28215020 | `f73216b` | PENDING |
| A | ox | 28215042 | `f73216b` | PENDING |
| B | Golden HU-MCD + Random, both directions | 28214892 | `982045f` | COMPLETED, exit0, acceptance PASS |
| C | Golden ACE/MCD | — | versioned preparation | No fit submitted |

A uses one L40S,8CPU,32GiB,2h per class; at most two A jobs run concurrently. Resource-only afterany chains are:

- hummingbird28214893 → tailed_frog28214896 → airliner28214983 → beach_wagon28215019 → ox28215042
- Siamese_cat28214894 → container_ship28214897 → police_van28214984 → zebra28215020

Failures do not make independent classes scientifically dependent. The affected task is never automatically resubmitted. B was independent, oneL40S/4CPU/16GiB/1h requested, **3m28s actual Slurm elapsed**, compute nodebun124. Both first A jobs have passed the shared prerequisite/input/weight checks, and their clean release and model/training-image-load progress were collected. No extra SAM probe was run.

A 的两条队列链只限制并发；不是“前一类结果正确才能研究后一类”。首批两类已启动，当前清洁代码身份与阶段进度已收集。进行中的A缺少最终summary/概念数组是阶段未完成；原收集器返回partial/exit2，不能误判为实验失败。

Execution commits: `982045f0e392e2bea02334a48afb96e6dcd26fc9` and `f73216b3e4980f8ee74bfaed9eb8a9c583931175`. The latter only adds the explicitly approved validation adapter, tests and documentation; scientific core functions are identical. Deployment jobs28214828/28215000 completed. Original Golden28208840, its400/50lists, caches, failures and diagnosis remain preserved. Protected hpc/check_job.sh and outputs hashes remain unchanged; unrelated NotebookLM/tools files were not staged.

## Input acceptance / 输入验收

All nine classes have1300training candidates and50validationIDs. Candidate audit28214251 and nine separate publication jobs ran on Slurm compute nodes. Eight L validation images (beach_wagon1,zebra5,ox2) were privately converted to lossless RGB PNG: original decoded gray pixels equal each output channel at every location, with unchanged dimensions. Original IDs, sources, hashes and order were retained; actual mappings were separately saved. All nine400/50lists are frozen.

已独立核对原文件与实际模型输入：全十类4,000训练/500验证均为唯一文件哈希，训练/验证交集为0。此项不是视觉近重复检测。灰度候选数、本次补样跳过数、原始清单/实际输入映射及场景风险详见[Nine-class input protocol](NINE_CLASS_INPUT_PROTOCOL.md)。Zebra候选66/1300=5.08%，并含1张原仓库能加载、而本训练规则排除的三通道等值RGB图；不是所有灰度排除都等同于原加载器报错。

The [paper §4](https://openaccess.thecvf.com/content/CVPR2025/papers/Grobrugge_Towards_Human-Understandable_Multi-Dimensional_Concept_Discovery_CVPR_2025_paper.pdf) describes random400/image-class sampling. The inspected paper/supplement do not specify this explicit grayscale filter or supply the exact historic identities. The [pinned upstream loader](https://github.com/grobruegge/hu-mcd/blob/168eb5bf1717ab46d1c84be86bd81f0c51861413/concept_explainer.py) sorts filenames for discovery and skips loader failures. Our fixed candidate order, explicit exclusion/refill and manifests are a disclosed input protocol difference. Possible historical-vehicle/working-animal scene enrichment is unverified; no claim of measured bias, training conversion comparison or result-driven reselection is made.

## B results / Golden 双向评价结果

**280 output files hash-verified; 10,652 prediction states accepted.** HU-MCD276states/direction; Random5050states/direction. Every trajectory includes all50images, all global batch8 boundaries and input ordering were checked, and partial streamed logits exactly match final saved logits. Saved masks independently reproduce pixel fractions; top1accuracy/std and contributor counts were independently recomputed using the original strict>75% rule. No additional GPU evaluation was needed for acceptance.

Reuse: original13bases and all569raw validation segments, including35zero-feature segments. No SAM or concept refit. Original precision retained: cuDNN TF32true, matmulTF32false, cuDNNbenchmark/deterministicfalse, deterministic_algorithmsfalse, float32precisionhighest. The local acceptance script and receipts are in the evidence folder.

| HU-MCD step / 已处理概念步数 | Images / 参与图数 | Mean deleted/inserted pixels / 平均处理像素 | C-Deletion accuracy | C-Insertion accuracy |
|---:|---:|---:|---:|---:|
| 0 | 50 | 0.00% | 92.00% | 0.00% |
| 1 | 50 | 28.73% | 22.00% | 78.00% |
| 2 | 49 | 46.47% | 10.20% | 79.59% |
| 3 | 41 | 53.97% | 9.76% | 82.93% |

For all50images, removing each image's highest-ranked concept covers28.73%pixels on average and drops accuracy92%→22%; inserting it reaches78%. For an illustrative **nearby observed Random point** (step29,28.86%pixels; no interpolation), deletion is80% and insertion26%, also50images. Pixel budgets are close, not identical, so this is a descriptive comparison rather than a new matched-budget metric or significance claim.

对全部50图，首个概念平均处理28.73%像素；删除准确率92%→22%，插入达到78%。Random已有第29步平均处理28.86%像素时分别为80%/26%。这是相近但不相同像素预算的实测点展示，没有插值、调阈值或新造评价指标。它支持本类别下相关性排序的有效信号；不能代替十类平均、ACE/MCD比较、多seed不确定性或人类研究。

![C-Deletion](../artifacts/bunya/workstreams-20260909/collected/28214892/20260909T052534.334524Z/evaluation_data/sdc.png)

![C-Insertion](../artifacts/bunya/workstreams-20260909/collected/28214892/20260909T052534.334524Z/evaluation_data/ssc.png)

## Limitations and anomalies / 限制及异常

- The35zero features mechanically tie to concept index0 with zero local relevance under the released benchmark. This is preserved and explicitly separated from learned evidence; prototype filtering was not substituted.
- Only four HU-MCD curve points satisfy>75% availability:50,50,49,41contributors. Later raw states exist but are not eligible aggregate points. Changing cohorts and the upstream early endpoint mean the curve cannot be treated as a complete same50-image trajectory to100%pixels. No padded endpoints, interpolated AUC or substituted rules were introduced. Plot lines merely connect observed points.
- Random floor-divided grids leave border pixels unchanged in21/50images. Mean final insertion visibility is99.5064%, not100%; exact masks/coordinates are retained. The seed43 ordering is our recorded execution choice, not an identified historical paper RNG state.
- Random curves have small non-monotonic fluctuations; an image becoming correct after masking is possible. No smoothing or monotonic correction was applied. The saved std is across-image binary-response dispersion, not a multi-seed confidence interval.
- One NumPy bool8 deprecation warning was recorded in B. It did not change values or fail acceptance. All seven evaluator observation entries, original stderr and earlier failures are retained in[ANOMALIES.jsonl](ANOMALIES.jsonl). Input issues are closed only after hash/pixel acceptance; possible scene effects remain documented risks.

35个零特征的首概念并列分配、曲线参与人数变化、原始端点控制和Random边界残留均保留并披露；不因来自原仓库就假定合理，也不把它们直接解释成论文结论错误。全部已运行B结果已收集，不存在用缺失输出冒充通过的情况。A尚未完成的结果不在这一验收结论内。

## C preparation and next decisions / 基线准备及下一步判断

C preparation is complete: exact ACE/MCD settings, source risks, cache boundaries, inventory evidence, resources and alternatives are documented in[BASELINE_PREPARATION.md](BASELINE_PREPARATION.md) and the[versioned plan](../hpc/configs/golden_retriever_baselines.plan.json). The bounded Slurm inventory found no baseline/random-pool candidates at the explicitly checked project locations; it does not prove none exist elsewhere. No baseline images were newly sampled and no C fit was submitted.

**Recommended next authorization: MCD only, independently of ACE.** Frozen Golden400/50, pinned ResNet50, unmaskedlayer4, batch8/currentprecision, q=.75, original FO/PCA rule, search3..19 and first completeness>.5. These are MCD benchmark settings, not HU-MCD settings; batch8 is an explicit execution difference from the upstream CLI64. Proposed split:1L40S/4CPU/16GiB/30min for450whole-image feature maps (estimated2–10min), thenCPU8/64GiB/4h for SSC+search (estimated30–180min, unmeasured). Requested upper budgets:0.5GPUh and34CPUh across the two stages. Save features once and reuse across searches. Do not add baseline evaluation to that authorization automatically. Alternative: finish A before allocating C, or defer both C methods; no resource escalation/retry by default.

建议下一步只批准MCD发现基线：保持Golden400/50，分GPU特征与CPU拟合，避免拟合阶段占GPU。资源和固定参数如上；耗时只是估计。是否批准仍由用户决定，当前没有提交。

ACE needs a scientific protocol decision first. Confirmed upstream behavior includes dropping the highest SLIC label, reusing the first50negative identities through20CAVruns per concept, potential positive/negative control overlap and unseparated gradient-test images. **A code-reference run and a corrected ACE protocol must be separately named and cannot silently replace one another.** The plan proposes ACE discovery as the frozen first50of400; training-only TCAV roles and a2000-image negative/control pool remain explicit decisions. MCD/A/B do not depend on that pool. Do not rerun B Random for later baselines if the full prediction signature matches.

The figures above are local evidence links; result images/caches remain outside the public Git commit. / 图为本地证据链接，结果图片与缓存不进入公开代码提交。

## Durable evidence / 持久证据

All new evidence is local at`artifacts/bunya/workstreams-20260909/`: STATUS.json, per-task plans/submission receipts, deployment provenance, nine publication collections, actual-input-acceptance.json, validation-compatibility-acceptance.json, B-golden-faithfulness.collection.json, B-evaluation-acceptance.json, accept_evaluation.py, both initial A snapshots and collection_commands.json. The original Golden bilingual report/evidence archive remains under`artifacts/bunya/final-28208840`.

Nineteen focused tests passed (shared/evaluation14, input compatibility5); scientific core and protected output hashes checked unchanged. Two local capture files have a historical.json suffix despite plain stdout; equivalent.txt files were added and the naming issue recorded. Automatic push review initially paused deployment; read-only destination/public-content verification resolved it before a reviewed retry. No bypass was used; no credentials, image files, weights or caches were uploaded to GitHub.
