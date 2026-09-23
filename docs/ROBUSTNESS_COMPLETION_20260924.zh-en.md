# ROB26-007 — fixed-slot completion / 保持槽位的补跑

The user's “好 那我们试试” explicitly approves trying the proposed engineering correction and completing the four failed classes. Scope is the existing five-condition protocol, fixed models/concepts/precision/input manifests; no SAM, refitting, threshold relaxation or expansion. Original failures remain. This is a one-time corrected submission, not permission to retry again.

用户批准尝试修正并补齐四类。复用原协议、固定模型/概念/精度/输入；不重跑SAM、不拟合、不放宽阈值、不改变科学规则。仅本次修正提交获授权，再失败需先报告。

## Concrete handling / 具体处理

- Reuse hash-bound accepted `identity0` and `erosion1` outputs from original failed jobs28731217/20/23/28. Their failure records are not overwritten.
- Recompute only complete `erosion2`, `dilation1`, `dilation2` conditions for each affected class; partial interrupted conditions are not claimed accepted.
- Preserve original batch membership, order and size, including a short final batch. For an empty intervention mask, place that same row's original input/mask in its original slot as an execution filler. **Discard its outputs**: scientific features/scores remainNaN, assignment−1, emptyflagtrue; retain the original baseline denominator.
- Frozen model is in eval mode; BatchNorm must use fixed running statistics. Inspected masked convolution/pooling paths are per-image, not batch normalization across current samples. No change to weights, precision or normalization.
- First run the exact historical failing batch with all original inputs at original size. Require existing feature/score/assignment tolerances. Save all control arrays before checks. Newly encountered batches needing fillers receive the same control once. Gate failure stops that class.
- Record every filled batch and its original, active and filler row IDs. Do not infer empty-region activations from filler outputs.

保留原batch槽位；空区域只用同一位置的原始输入维持执行形状，输出丢弃，科学记录仍为空/无效。入口针对历史失败batch检查，后续新出现空区域batch各检查一次。原identity与r1结果通过哈希复用；缺失三个条件重新完整计算，避免利用中断产物假装验收。

## Resource ceilings / 资源上限

Existing per-class ceiling:1L40S,4CPU,16GiB,20min,1GiBoutput. Four classes total at most80GPU-minutes; two resource lanes at most2L40S for this batch. Deployment1CPU/1GiB/5min. Prior six complete class jobs measured38–110seconds, while interrupted jobs reached98–352seconds; these are not timings for this correction and no speedup is guaranteed. No automatic time/memory/output increase.

沿用每类1L40S/4CPU/16GiB/20分钟/1GiB输出；四类最多80GPU分钟，本批最多两卡。部署1CPU/1GiB/5分钟。原成功六类38–110秒、失败作业98–352秒，仅作已有实测参照，不是假定本次耗时。提交后不等待，用户通知完成后再收集。

## Display correction / 图表修正

ROB26-008: previous heatmap used equal-aspect cells, compressing four columns; repeated long “Not accepted” strings and x labels overlapped. New renderer uses wide cells, percentages, two-line labels and a short missing symbol with explicit legend. Scientific values unchanged. Source `scripts/research_hub/robustness_figure.py`; integration remains in `scripts/build_research_hub.py`. Only the current robustness image/caption is repaired; other branches and review records are preserved.
