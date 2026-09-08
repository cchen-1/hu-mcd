# 400/50 单类别正式运行衔接

2026-09-09：用户已恢复 GitHub 推送认证，并明确只使用已有的一次 probe；没有具体故障就推进正式复现，不追加小样本试跑。

## 决定与执行顺序

- 原提交 `05d965d30b1fb6f2b1ee373da2956c63c3445257` 已正常推送到 fork 的 reproduction 分支。
- 检查时唯一 probe **28204575** 仍为 PENDING/Resources，尚无日志和结果。不能将排队误报成已通过。
- 正式作业通过 `afterok:28204575` 排队衔接。没有新增 probe。
- 正式作业只会在已有 probe 成功完成之后启动；`--kill-on-invalid-dep=yes` 使失败依赖不会留下永久等待的正式作业。
- 计算节点启动时核对 probe job ID、科研 commit、两个 helper 的哈希、PROBE_COMPLETED/PASS、数据和模型哈希以及已测试 batch size。届时记录完整 probe launch_manifest 的实际 SHA256。
- 这允许当前直接提交正式作业的队列依赖，而不伪造尚不存在的 probe 结果哈希。

## 固定配置

配置文件：`hpc/configs/golden_retriever_reference.json`。

| 项目 | 值 |
|---|---|
| 类别 | golden_retriever / n02099601 |
| 训练 | 已固定的 400 张，seed 43，不重新抽样 |
| 验证 | 原 50 个 ID，唯一 L 图使用已验证的私有 RGB PNG |
| SAM | ViT-H，32 points per side |
| 特征 | 已固定权重的 masked ResNet50 / global_pool |
| 聚类 | released SSC/PCA，自动簇数，min cluster size 50 |
| batch size | 8，与现有 probe 一致 |
| 保存科学记录 | 开启 |

正式作业的初始资源预算：gpu_cuda / short，1×L40S，8 CPU，32 GiB，12 小时时限。
这是首个完整运行的预算上限，不是基于尚未完成 probe 的实测耗时预测。
调度器只读核查表明用户账户允许 short，gpu_cuda 分区也允许 short，short 的 MaxWall 为 12 小时。
不为资源预算再增加小样本实验；正式运行的阶段耗时、内存、失败记录用于后续诊断。

joblib 子进程使用 `inner_max_num_threads=1`，避免 8 个 worker 各自再启动 8 个 BLAS 线程。
该改变限制 CPU 并发，不改变聚类、采样、特征、模型或评价公式。
本地双 worker 检查确认每个 worker 的 BLAS 线程数为 1。

## 正式提交

先部署本次提交的精确科研版本，然后调用：

```bash
python3 hpc/submit_release.py \
  --commit FULL_DEPLOYED_SHA --mode reference \
  --config hpc/configs/golden_retriever_reference.json \
  --after-probe 28204575 \
  --probe-commit a5e7f843856d17b7647f8af6571319b45719f669 \
  --cpus 8 --memory 32G --time 12:00:00 \
  --partition gpu_cuda --qos short --gpu l40s:1 --submit
```

probe 的科研 SHA 是 a5e7f84；它通过传送 helper 执行资源及数值检查，helper 的具体哈希已经固定。
正式运行使用包含科学记录和线程约束的新提交；不将两者混称为同一个版本。

运行过程中保留每个 job 独立的 masks/activation/SSC 缓存、阶段进度、输入身份、拟合基底、标签、分配与重构检查。
正式训练和验证产生的全部特征与 logits、分配，以及片段级重构样本保存至 `scientific/`。
这些记录用于判断验证分配问题；运行成功仍不等于完整论文评价已复现。
不会在这一轮加入训练灰度图转换对比实验。

45 项本地测试通过，包含依赖渲染、拒绝未完成/失败 probe、job/commit/helper 身份不符以及已有采样/兼容/结果回收测试。
实际提交和部署的 job ID、SHA、调度状态保存在 `artifacts/bunya/formal-reference/`，由主对话回报。
