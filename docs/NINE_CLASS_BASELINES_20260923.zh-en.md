# Nine-class baseline execution / 九类基线执行 — 2026-09-23

User authorized the remaining nine classes' MCD and **ACE — HU-MCD released-code reference** discovery and C-Insertion/C-Deletion under the existing Golden protocols. Human review is independent, not a prerequisite. Golden results and all ten HU/Random results remain accepted and unchanged. Scheduler/accounting and local receipts showed no other baseline submission before this batch; the existing authenticated SSH master was reused.

用户已批准剩余九类MCD与ACE参考实现的发现及双向评价。人工审阅不作为前置条件；Golden及十类HU/Random已验收结果直接复用。本轮提交前核对实际队列、会计与本地回执，未发现九类基线已提交。

## Protocol and identities / 协议与身份

- Fixed class-specific400 training /50 validation lists and actual input mappings remain unchanged, including existing approved grayscale compatibility conversions. MCD uses layer4 spatial features, q=.75, k=3..19, stop strictly at completeness>.5; exhausted search retains k19 with threshold_met=false as source. One accepted machine-checked fit feeds both directions.
- ACE uses the first50 of each fixed400; gradient50 follows actual original post-control/CAV Python shuffle, validation50 isolated; random2000 is label-blind training RGB selected with independent preparation seed43, excluding that class's validation identities/hashes. Selection and all role overlaps are recorded. Repeated negatives,20CAV rounds,SLIC max-label omission, CE gradients, two-sided nominal p<.01, global TCAV ordering and original aggregation remain unchanged. Preserve every candidate, round, test, difference direction and filter reason. Passing nominal p does not imply higher-than-control mean or calibrated independent repetitions.
- MCD/HU discovery400 vs ACE50 and local-vs-global ranking remain method differences; using the same protocol across classes does not eliminate these comparison limits. MCD spectral initialization remains the original unspecified setting, not newly seeded.
- Each class reuses **its own** accepted Random predictions only under the complete exact signature (inputs, weights, preprocessing, batching, precision, implementation, seeds, grid/edges and directions). No cross-class Random substitution and no new Random inference.

固定每类400/50清单、实际输入映射及已批准灰度转换不变。MCD参数、搜索/停止条件沿用Golden；ACE保留全部已批准源码行为和记录，名义检验通过不等于高于随机控制。每类复用本类签名一致的Random；不跨类借用。方法间发现样本量与排序等既有差异继续披露，不声称同协议即可消除所有比较限制。

## Engineering changes / 工程适配

Golden-only guards/target207, candidate IDs and Random source28214892 are parameterized. The original scientific files are not edited. `baseline_multiclass.py` verifies each accepted discovery's code/input/model chain and Golden's shared environment/masking attestation. Each submitted stage binds an immutable commit, exact class config, input/weight hashes and unique output directory. Future stage hashes are resolved **inside Slurm** from exact producer IDs/task keys/commit/class/input/model/precision and successful manifest/artifact index. Resolved hashes/config are saved before consuming the cache; original worker checks then verify its owned files and scientific signature.

工程变化仅为类别身份/目标输出/命名/Random来源参数化，以及Slurm内阶段衔接。原算法文件未改；后续尚未产生的缓存哈希不能预先编造，而是在指定前序成功后于计算节点核验、记录并绑定。机器阶段门槛不等于最终本地科研验收；后者待用户通知所有批次完成后统一进行。

## Resources and scheduling / 资源与依赖

| Method/stage / 方法阶段 | Allocation ceiling / 上限 | Golden elapsed / Golden实测 |
|---|---|---|
| MCD features |1L40S,4CPU,16GiB,30min|18s|
| MCD fit |8CPU,64GiB,4h,noGPU|10m13s|
| MCD both directions |1L40S,4CPU,16GiB,30min|33s|
| ACE inputs |4CPU,8GiB,15min,noGPU|1m45s|
| ACE features |1L40S,4CPU,16GiB,1h|3m03s|
| ACE CAV |4CPU,16GiB,1h,noGPU|41s|
| ACE gradients + both directions |1L40S,4CPU,16GiB,1h|31s|

Nine classes ×7 stages =63 planned jobs plus one 1CPU/1GiB/5min deployment. GPU allocation ceiling27hours total; maximum2GPUs, one independent resource lane per method. Actual nine-class elapsed times are unknown. Stage afterok dependencies stop affected descendants on failure; afterany resource ordering lets later unrelated classes proceed after termination. No automatic retry or expansion. CPU fitting does not reserve a GPU. No SAM/probe/discovery rerun of HU-MCD.

九类共63个阶段作业及一次部署，GPU预约上限27小时、最多并发2张（每方法一条独立资源队列）；CPU拟合不占GPU。实测Golden仅供参考，不保证其他类别耗时。afterok控制科学依赖，afterany仅限流，不把某类失败当其他类的科学前提。失败不重提、不扩容。

## Delivery and follow-up / 交付与后续

Plans, actual submission receipts and scheduler snapshot: `artifacts/bunya/nine-class-baselines-20260923/`. Dry-run dependency numbers in `dry-run/` are placeholders, never actual jobs. Only `submissions/*.submission.json` and `batch.json` identify actual submissions. The coordinator exits without waiting. No new result curves/counts are claimed at submission; existing bilingual report lists pending stages alongside previously accepted results. Full collection, scientific acceptance and result inclusion occur only after the user's next completion notice. Preserve all warnings/failed attempts and unrelated medical/report work.

本轮只更新提交/准备状态，不等待作业结束，也不以已提交代替已完成、已验收。用户通知后再统一收集、验收并增量加入结果；所有警告、失败历史、现有审阅及无关医学工作保留。
