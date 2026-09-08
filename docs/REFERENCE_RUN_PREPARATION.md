# 单类别参考运行：输入兼容、ViT-H 与探测衔接

后续正式 400/50 作业与既有 probe 的依赖衔接见 `FORMAL_REFERENCE_RUN.md`；不再追加小样本探测。

更新：2026-09-09。用户已授权：保持训练 400 张及验证 50 个 ID，将唯一验证 L 图在私有目录转为 RGB PNG，准备 ViT-H，做短资源探测与关键重构检查，固定代码后推进单类别复现。训练灰度转换对比实验不在本次范围内。

## 已完成

- CPU Slurm 28204479（bun147，7 秒）：发布 `reference-golden-seed43-compat-v1`。
- CPU Slurm 28204480（bun147，15 秒）：从 SAM 官方地址准备 ViT-H。
- 本地数值与集成测试：包含非正交概念子空间、正负 relevance、complement、单独添加 bias、错误重构拒绝，以及科学结果的 producer→collector 校验。
- 核心拟合/分割算法未改写。run_smoke.py 新增可配置的科学记录与重构检查；此辅助集成必须使用本次新的代码提交，不能称为原 a5e7f84 的内容。

训练清单 400 项和验证清单 50 项均与前一版逐字节相同。
唯一调整为验证 ID `ILSVRC2012_val_00019590.JPEG`：

| 字段 | 值 |
|---|---|
| 实际输入文件 | `val_imgs/golden_retriever_val/0020_ILSVRC2012_val_00019590.png` |
| 原模式/尺寸 | L / 415×500 |
| 操作 | 将灰度值复制到 R、G、B，保存无损 RGB PNG |
| 像素检查 | 保存后重新打开；三个通道均与原 L 像素逐点相同 |
| 原文件 SHA256 | `ff191fd4f872734df5bab18cc8feef77d12d14ad163c85781cc685ad752e3614` |
| 实际输入 SHA256 | `7ec11f232a95e7029b072a9943760aee0d75d1f8fb057cf347dec4354000f7fc` |

共享 ImageNet 源文件和上一版准备目录未改动。
新目录 `/scratch/user/uqcche38/hu-mcd/data/reference-golden-seed43-compat-v1` 保存原始两份图片清单、两份实际输入清单和完整 `input_mapping.json`。
原模式异常保存在 `original_validation_issues`，此次输入兼容调整在 mapping 中逐项记录。

新 dataset_manifest SHA256：
`dafeafcaa64d2379a236500d443e18f8a27520b9a0ad0288e79bdab7c683cd74`。
input_mapping SHA256：
`4c326d6ea480b7ff41b6410bb39bd6efaee2057617aab7cbcef5297d7dc5d575`。

版本控制中的 `hpc/manifests/golden_retriever_seed43.json` 保存 400/50 个 ID、原始哈希、实际输入哈希和文件名，不含图片。
完整远端证据已回收到本地 `artifacts/bunya/reference-preparation/{28204479,28204480}/`。

## SAM ViT-H

官方来源：[facebookresearch/segment-anything 的 Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)。
下载 URL：`https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`。
路径：`/scratch/user/uqcche38/hu-mcd/models/sam_vit_h_4b8939.pth`。
大小：2,564,550,879 bytes。
SHA256：`a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`。
下载使用官方 HTTPS URL，核对 Content-Length 并计算完整哈希；该哈希是实测值，不冒充发布方签名。
模型结构和 strict state-dict 加载由后续 GPU 探测核查。

## 短资源探测

Slurm job **28204575**，基于已部署的 a5e7f84 科研核心，通过 launcher 传送单独记录哈希的 probe/numerics helper。
请求：gpu_cuda/debug，1×L40S，2 CPU，16 GiB，最多 5 分钟。
正式配置参数仍为 ViT-H、32 points、300 shortest-side、masked ResNet50/global_pool；batch size 初试 8。
探测仅使用固定训练清单前 2 张、验证清单第 1 张和上述转换图，合计 4 张，不修改正式清单。

检查范围：

- ViT-H 严格加载和逐图分割耗时、片段数；
- masked ResNet 特征与实际 1000 类 FC logits 重构；
- 全 1 mask 与同一张量无 mask 的输出一致；全 0 mask 输出为有限数；
- 小型非正交代数 fixture 中的 feature / signed relevance / complement / bias 重构；
- 至多 128 个片段的单进程 SSC primitive 耗时；GPU/RSS 峰值。

此探测不拟合正式概念，不产出论文指标，不以四张图片推断完整数据的确定耗时或内存上限。
实际拟合子空间的重构检查在正式运行中执行。

最后检查时探测因 Resources 排队；调度器估计开始时间为 2026-09-09 17:36:18（集群时间），不是保证。
A100 节点已分配全部 GPU；没有重复提交探测或更改其他人的作业。

## 科学记录与正式启动条件

`save_scientific_records: true` 时保存到每个 run 的 `scientific/`：

- `discovery.npz/json`：所有 fitted concept bases、complement、聚类 labels/outlier mask、保留的 cluster label、FC 权重/bias、模型配置；
- `training.npz` / `validation.npz`：实际特征、1000 类 logits、概念激活及 argmax 分配、固定最多 8 个片段的分解和局部 relevance；
- `*_segments.json`：每个片段的图片、索引及 mask 内容哈希；
- `*_checks.json`：全部片段的 FC 重构和抽样概念+complement 重构误差。

此阶段的局部 relevance 是片段级关键数值证据，不能称为完整空间热图、C-Insertion/C-Deletion 或人类研究结果。
原始 mask/activation 缓存仍按 job ID 保存；新科学文件进入 producer 校验索引。
collector 使用 `--include-scientific --max-file-mib 256` 回收并核对这些文件。

正式模式 `--mode reference` 必须显式提供：

- 精确提交与完整配置；
- `--probe-launch` 指向成功探测的 launch_manifest；
- `--probe-launch-sha256` 绑定实际探测记录。

计算节点启动时再次检查 probe 为 PROBE_COMPLETED/PASS，且数据 manifest、ViT-H 和 classifier 哈希与正式配置一致。
探测未完成或失败时不能启动正式 reference。这防止为了排队提前跳过资源/正确性检查。
下一步根据真实探测结果决定正式资源请求，并检查当前 QOS 上限；目前没有提交正式 400 张 GPU 实验。
