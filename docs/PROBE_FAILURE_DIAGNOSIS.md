# Probe 28204575 失败根因与正式作业决策

更新：2026-09-09。**同型号 GPU 根因验证已完成；不是“尚待 GPU 验证”。**

## 结论

**这次失败由我们新增的 probe 检查设计导致：它将全一掩码、batch=2 的输出，与普通路径、batch=1 的输出比较，把受 cuDNN TF32 影响的跨 batch 数值差异误判为掩码实现不等价。**

诊断作业 **28206208** 于布里斯班时间 08:48:35–08:48:53 在 **bun124 / NVIDIA L40S** 完成，Slurm `COMPLETED`，exit `0:0`，总运行 18 秒。相同研究代码、模型权重、输入复现了原失败；纯普通路径的 batch 大小变化产生完全相同的逐元素误差。关闭 / 恢复 cuDNN TF32 使误差消失至原容差以内 / 精确返回。

因此这次已定位的失败不需要修改 HU-MCD 核心算法、400/50 数据清单或灰度兼容处理。**它也不能证明完整算法不存在其他问题，或此前验证集 learned-concept assignment 低仅由小样本造成。** 这些需要后续正式运行的科学记录判断。

## 有控制变量的证据

以下为同一张原始 RGB 训练图在同一 L40S 分配内的对照；所有值均从取回的原始 logits 数组重新计算并核验。

| 比较 | 最大绝对 logit 误差 | RMSE | 超原容差数 / 1000 |
|---|---:|---:|---:|
| TF32 默认：masked batch 2 首项 vs plain batch 1（复刻原检查） | **0.012950420379638672** | 0.0035049661595197754 | **751** |
| TF32 默认：plain batch 2 首项 vs plain batch 1 | **0.012950420379638672** | 0.0035049661595197754 | **751** |
| TF32 默认：masked vs plain，同为 batch 1 | **0，逐位一致** | 0 | 0 |
| TF32 默认：masked vs plain，同为 batch 2 | **0，逐位一致** | 0 | 0 |
| TF32 默认：masked vs plain，同为正式配置 batch 8 | **0，逐位一致** | 0 | 0 |
| 仅关闭 cuDNN TF32：复刻原检查 | **1.9073486328125e-6** | 6.049915487382801e-7 | **0** |
| 恢复原 TF32：复刻原检查 | **0.012950420379638672** | 0.0035049661595197754 | **751** |

额外控制：

- 掩码跨 batch 误差向量与纯普通路径跨 batch 误差向量 **逐位相同**，不仅仅最大值相同。
- 恢复 TF32 后的输出与最初默认输出 **逐位相同**。
- 默认 batch 8 vs batch 1 的普通路径最大差异为 0.012240886688232422（731 项超容差）；关闭 TF32 后降至 1.9073486328125e-6。同 batch 8 的掩码等价性始终成立。
- 第一张图的结果不受第二张图 mask 是全零还是全一影响；重复同一次 forward 输出逐位相同。
- 所有保存的 forward 均有限，top-1 均为 golden retriever（ImageNet index 207）。这只能说明该图的 top-1 稳定，不能推广为全数据集预测或 SSC 聚类均不受精度影响。
- 模型始终为 eval，所有 BatchNorm 均为 eval。模型全部参数 / buffers 的前后哈希完全一致。

原容差为 `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`。751/1000 是单张图的 1000 类 **logits 超容差的数量**，不是 75.1% 图片分类错误，也不是验证集概念分配失败率。

原跨 batch 差异的整体相对 L2 误差为 0.0003425733（约 0.0343%）；最大逐元素相对误差为 0.00283608（约 0.2836%）。不要把二者混用。

## 误差从哪里出现

对原检查的纯 batch 对照，诊断记录的中间层最大绝对误差如下。不同层数值尺度不同，不能把最大绝对值直接当成统一的误差放大比例。

| 观测位置 | TF32 默认，batch 2 vs 1 | TF32 关闭，batch 2 vs 1 |
|---|---:|---:|
| conv1 / bn1 / act1 / maxpool | 0 | 0 |
| layer1 输出 | 0.052544474601745605 | 0 |
| layer2 输出 | 0.2138357162475586 | 0 |
| layer3 输出 | 0.20262980461120605 | 0 |
| layer4 输出 | 0.04866480827331543 | 0 |
| global_pool | 0.002566516399383545 | 0 |
| fc | 0.012950420379638672 | 1.9073486328125e-6 |

同 batch 的掩码 / 普通路径在以上所有记录位置都逐位一致。**首次观测到的跨 batch 分歧在 layer1 输出**；本次未对 layer1 内逐个卷积或实际 CUDA kernel 做 profiling，因此不声称已识别某个具体 kernel。对当前故障的必要因果控制已完成，无需为了继续复现再增加一轮小样本 probe。

实际环境及精度开关：

```text
torch 2.2.2+cu121; CUDA 12.1; cuDNN 8902; timm 0.6.13; NumPy 1.26.4
cudnn.allow_tf32 = True
cuda.matmul.allow_tf32 = False
cudnn.benchmark = False
cudnn.deterministic = False
deterministic_algorithms = False
float32_matmul_precision = highest
```

`matmul.allow_tf32=False` 不等于卷积 TF32 已关闭。这里的干预只改变 `cudnn.allow_tf32`；其余开关保持原值。

PyTorch 官方关于 [batched/sliced computation](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#batched-computations-or-slice-computations) 和 [TF32](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#tensorfloat-32-tf32-on-nvidia-ampere-and-later-devices) 的说明支持这一机制。最终归因依据是上表在原 torch 2.2.2 环境里的受控实验，而非套用当前文档默认值。

## 上游代码、我们的调整与输入身份

原 probe 关键比较是：

```python
masked = explainer.model(batch)          # batch=2，第一项全一 mask，第二项全零 mask
ordinary = explainer.model(batch[0][:1])  # batch=1
np.testing.assert_allclose(masked[:1], ordinary, rtol=1e-4, atol=1e-4)
```

全一 mask 等价性应在 **相同输入张量与相同 batch 形状** 下比较。上面新增检查同时改变两个条件，是本次误判的程序位置 `hpc/reference_probe.py:66`。

`classes.py`、`concept_explainer.py`、`input_masking/resnet.py`、`input_masking/sal_layers.py`、`utils/utils_general.py`、`utils/utils_mcd.py` 在以下三版内容哈希完全一致：

- 上游 `168eb5bf1717ab46d1c84be86bd81f0c51861413`
- 失败 probe 的研究代码 `a5e7f843856d17b7647f8af6571319b45719f669`
- 取消的正式作业版本 `a12f9ab393de73cfddfcc8b8ae2af46ac7d00be8`

Bunya 已安装的两个 masking 文件也与上述源码一致。原日志的 `utils_mcd.py` tuple-assert 警告是另一项上游问题，不是实际触发的异常。

固定失败输入为训练清单第一张 **n02099601_10817.JPEG，原始 RGB**。输入文件哈希与冻结清单一致，GPU 输入张量哈希也与前一轮本地 CPU 对照一致。它不是获准转换的那张 L 验证图。

ResNet checkpoint SHA-256：`14fe96d1f9fb311a60490082d2077e6e60427dcfe21839ddf934cce948f72b0f`。

原 probe 已生成 38 × 2048 features 和 38 × 1000 logits，其中一个全零 feature 行由上游规则剔除。缓存 FC 重构在 WSL 最大误差 1.9073486328125e-6，在本次 Bunya CPU 为 3.814697265625e-6，均 **0 项超容差**；差异符合不同 CPU/BLAS 的微小浮点归约差别。没有发现保存的特征与模型 logits 错位。

## 对正式作业的明确建议

**可以继续正式复现的衔接；无须再次从 SAM 开始做小样本 probe。需要改的是校验和证据衔接，而非为这条断言改动科学算法。**

1. **修正边界检查**：普通路径使用完整的 `batch[0]`，再比较两侧第一项。两侧保持同 batch，继续使用原 1e-4 相对 / 绝对容差；把跨 batch 差异单独保存为诊断。全零 mask 的有限性检查仍保留。
2. **修正失败记录**：原 `probe_report.json` 到函数末尾才写，断言失败使已完成的分割计时、峰值显存、FC 摘要没有保存。应逐阶段保存，并在 assert 前保存误差和必要原始数组，异常退出也有记录。
3. **接通真实的前置证据**：28204575 保持 FAILED，不改历史状态。当前 launcher 只接受成功 probe，不能把 28206208 直接冒充为一个完整成功 probe。正式启动前需明确绑定“原 probe + 本次根因诊断”的版本、输入、模型、原始缓存与 SHA，验证补充证据，然后完成原断言之后未执行的代数 fixture / SSC primitive。这些可复用现有缓存，在正式作业预检里执行，无须重复 SAM。
4. **保持首次参考运行的数值协议可比**：建议保留本次原环境默认精度（卷积 TF32 开，矩阵 TF32 关），固定 batch=8，显式记录所有实际精度开关。不要仅为了消除错误断言而全局关闭 TF32。关闭 TF32 会真正改变 logits：本图同 batch 2 下开 / 关 TF32 的最大差异为 0.0385232，不能称为纯日志修复。若以后需要严格 FP32，应作为清楚命名的数值协议变更，而非静默修改本次参考。
5. **保留正式阶段检查**：训练 400、验证 50、seed 43、ViT-H 与清单不变；继续保存 masks、features/logits、SSC、bases、分配得分和重构误差。实际拟合后仍应校验 feature / relevance 重构，并诊断验证集是否大量落入 complement；本次单图等价性检查不能代替这些。

这些是基于已完成诊断的下一步修改要求。**上述为诊断完成时的状态；后续实现及实际提交状态见 FORMAL_RESUMPTION.md。**

## 资源与作业状态

- 原 28204575：bun125 / L40S，FAILED，28 秒；batch MaxRSS 5,625,080 KiB。ViT-H 分割及 batch 8 特征计算已执行，未进入代数 fixture / SSC primitive。没有 OOM 证据。
- 诊断 28206208：bun124 / L40S，COMPLETED，18 秒；batch MaxRSS 1,266,000 KiB；诊断自身 GPU allocated peak 260.31 MiB。**这是 ResNet 单图诊断的峰值，不能用来估算 ViT-H / 400 图正式资源。**
- 正式 28205921：08:10:06 因原 probe 的 afterok 前置条件失败被 Slurm 自动取消，elapsed 0，无节点分配。它没有开始正式复现，不能恢复运行；后续准备好新版本与前置证据后才提交新的正式作业。

原完整 ViT-H 显存峰值和计时摘要未保存，无法事后恢复。已有成功阶段是可用证据，但不等于完整资源认证；正式运行需保留实际阶段耗时与资源记录。

## 审计材料

本地私有结果目录：`artifacts/bunya/probe-failure-28204575/`。

- 原失败：`original-probe.out/.err`、`original-launch.json`、`original-accounting.psv`、`original-cache/`
- 源码对照：`core-source-comparison.json`
- 本地 CPU 对照：`local-fc-reconstruction.json`、`local-cpu-control.json`、`local-cpu-logits.npz`
- 完成的诊断：`diagnosis.json`、`diagnosis.out/.err`、`diagnosis-accounting.psv`
- 原始诊断数组：`logits.npz`、`layer_traces.npz`（108,774,099 bytes）、`input_tensor.npy`
- 原始数组独立重算：`validated_findings.json`、`validate_completed_probe_diagnosis.py`
- 提交身份：`diagnosis-submission.json`、`diagnosis-submitted.sbatch`
- 历史排队判断：`diagnosis-before-completion.md`

所有三份远端数组均按 `diagnosis.json` 的大小和 SHA-256 复核通过；独立重算核验了逐元素误差、TF32 恢复输出及三组关键逐层 trace。对 RMSE / norm 这类跨主机浮点汇总使用 1e-12 相对、1e-15 绝对重算一致性容差；最大误差、计数、原数组逐位比较的结论一致。这不是更改模型检查的原 1e-4 容差。

`diagnosis.json` SHA-256：`e7e44f3b407963c24413e030b9bece6d6a981e9c5222269f49b6a476faecf849`。

本次远端只执行 scheduler 查询和 SFTP 取回；计算/文件处理已经由 Slurm 诊断执行，本地复核在 WSL 完成。保留既有 `hpc/check_job.sh` 与 `outputs/`，不修改原 probe 或正式作业的历史证据。
