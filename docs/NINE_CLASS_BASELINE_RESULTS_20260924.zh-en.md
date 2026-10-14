# Nine-class baseline acceptance / 九类基线验收 — 2026-09-24

All 63stages: **COMPLETED / 0:0**, collected and independently accepted. Golden and all ten HU/Random results are reused. No new model computation, predictions, sampling or retry. / 全部63阶段结束并验收；复用Golden及HU/Random，无新模型计算、选样或重提。

| Class / 类别 | MCD concepts / 概念 | Completeness | ACE nominal-retained / 名义保留 | ACE above / below / 高于低于控制 |
|---|---:|---:|---:|---:|
| airliner | 8 | 0.5266910 | 18 | 18 / 0 |
| golden_retriever | 6 | 0.5585487 | 3 | 0 / 3 |
| hummingbird | 4 | 0.5204796 | 10 | 0 / 10 |
| Siamese_cat | 3 | 0.5672563 | 12 | 9 / 3 |
| tailed_frog | 6 | 0.5105925 | 12 | 12 / 0 |
| container_ship | 5 | 0.5182219 | 21 | 21 / 0 |
| police_van | 4 | 0.5122380 | 19 | 19 / 0 |
| beach_wagon | 4 | 0.5009013 | 22 | 22 / 0 |
| zebra | 4 | 0.5169276 | 10 | 10 / 0 |
| ox | 3 | 0.5213434 | 11 | 2 / 9 |

## Evidence and interpretation / 证据与解释

- Execution commit: `1e094cd7fa9f3e6990e6aad1294e85b96b82905e`. Explicit per-stage configs, input/weight hashes, Slurm dependency bindings and independent output directories verified. Original method source hashes and Golden scientific settings match; wrapper diffs are retained. / 固定执行版本、配置和身份链已验收，封装差异留档。
- All 400/50 MCD and 50/50 ACE discovery/evaluation inputs verified; gradient 50 and random 2000 roles/intersections replayed. Same full-image correctness per class across all four methods. / 固定输入与角色交集核对，四方法全图正确性身份一致。
- MCD raw 49 cells/image, nonzero preparation, both SSC caches, cluster/PCA mappings, first-crossing search, dimensions and saved reconstruction checked. Saved remote SVD/oblique checks reused; no refit. / 重用既有代数检查，无重拟合。
- ACE candidates/rounds/negative identities, nominal tests, analytic CE gradients, CAV dot products, nearest centroids and every actual mask trajectory replayed from saved arrays. / 从缓存重放分配、梯度、检验和实际掩码序列。
- Equal-TCAV ordering portability: Siamese_cat, airliner, container_ship, hummingbird, tailed_frog. Recorded valid order is authoritative; default local sorting may choose another tie order. No stable-sort method change. / 并列排序沿用实际顺序，未改变方法。
- ACE ten-class totals: 138 retained, 113 above control and 25 below. Repeated negative identity sets, paired two-sided nominal tests and their filtering effect remain disclosed. / 名义保留不等于高于控制或独立重复证据。
- ACE 50 vs HU/MCD 400; masks/rankings differ. Pool all 500 existing per-image trajectories at native step with n>375, without first applying class-level n>37.5 filters. No interpolation, AUC, smoothing or missing endpoint completion. / 汇总分母与类别贡献矩阵公开，不能作相同像素预算排名。
- Exploration forms stay unreviewed; new concept counts, reconstruction or curve performance do not establish human understandability. / 人工状态不因工程验收升级。

## Resources / 资源

GPU-stage allocated elapsed total **0.6516667 GPU hours**, measured from Slurm, versus the authorized27-hour allocation ceiling. Not utilization. MCD CPU fit 512–1544s; ACE feature 180–198s. No allocation enlargement or failed-job retry. / 实测累计GPU阶段分配时长约0.65小时，不是利用率；无扩容或重提。

## Files and rebuild / 文件与重建

- Result and acceptance root: `artifacts/bunya/nine-class-baselines-20260923/final-20260924/`.
- Existing private entry: `artifacts/bunya/ten-class-review/report/index.html`; current comparison: `baseline-progress.html`.
- Generate: pinned humcd-upstream Python `analyze_batch.py`; `render_baselines.py --class-name CLASS`; `scripts/research_hub/baseline_results.py STAGE` on a copy of the live report. The existing `build_research_hub.py` also calls this extension when the accepted summary exists.
- `validators/` preserves Golden-derived multiclass adapters; `common.py` verifies submitted→resolved configs against exact producer receipts and committed code bytes. `replay_semantics.py` performs no predictions.
- `analysis/` contains all250 ACE candidates, all round scores/intersections, both-method concept index, transitions, full warning inventory, curves and denominators. `visuals/CLASS/index.json` records every displayed identity, mask hash and selection rule; exact local images are reused.
- Stage publication only after local link checks and offline browser QA. Preserve previous report, HU images/IDs/review schema, unrelated medical pages and current `hpc/check_job.sh`/`outputs/`. No public upload.

## Remaining work / 后续

Formal human study and the paper input-treatment ablations remain unperformed. A corrected ACE protocol comparison would require a separately frozen and approved design; no correction was silently applied. / 正式人类实验及原论文输入处理消融仍未开展；ACE修正版比较须另行批准。
