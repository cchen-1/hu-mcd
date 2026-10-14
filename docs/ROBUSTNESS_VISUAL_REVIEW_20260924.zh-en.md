## 2026-09-24 A线视觉解释补充 / Compact robustness comparison

已在原robustness.html热图后加入两张紧凑诊断总览及十类逐类对照图；40固定显示样例×5条件。R随机、S全条件稳定控制、D最高IoU有效切换、E空/零或补空间诊断，记录seed20260921+类别序号。每格叠加/selected-only，金色新增/品红移除，五条件同片段，保留完整ID、原图链接、概念参考成员与人工审阅CSV。图片输入、raw/effective掩码哈希和几何逐例验证；无新推理、远端操作或改实验。

Two diagnostic overviews and ten compact class sheets now accompany bilingual discussion beneath the figures. All40 examples have source hashes and actual224-input/mask alignment checks. All22748 perturbed region records underpin pooled transition/margin summaries. The maximum-IoU switch selection gives onlydilation1 cases (IoU0.965–0.990); this diagnostic selection is not a comparison of population risks. AI Airliner/Golden visual observations remain pending human confirmation. Lower baseline margin among switches is descriptive association, not causal proof or an independent-region test. No semantic-quality or segmentation-accuracy conclusion.

Generator: `scripts/research_hub/robustness_examples.py`, called by existing robustness completion publisher. Evidence: `artifacts/report-robustness-visual-20260924/{local-qa,browser-qa}.json`; report assets/CSV/provenance under `report/hub/robustness-examples/`. ROB26-009/010 retain selection limitations and resolved local rendering problems. Original hpc/check_job.sh and outputs/ preserved.
