# 指定 release 的部署与单类别数据准备

最新进度见 docs/REFERENCE_RUN_PREPARATION.md：唯一验证 L 图的兼容转换与 ViT-H 下载已完成，GPU 短探测 28204575 已提交，正式运行等待探测通过。以下早期状态保留用于追溯。

本轮以 a5e7f843856d17b7647f8af6571319b45719f669 为科研代码基线。
这个提交已部署；下面新增 launcher/worker/collector 仍是本地未提交修改，
不属于该 SHA。launcher 将自己的内容哈希单独记录，科研 checkout 不被修改。

## 已验证的结果（2026-09-08）

| Slurm job | 工作 | 结果 |
|---|---|---|
| 28190453 | 从 fork 准备精确提交 | COMPLETED，bun147，4 秒 |
| 28193321 | 使用新 launcher 核对 release、ImageNet 和环境 | COMPLETED，bun147，6 秒 |
| 28193518 | 严格 RGB 数据准备 | FAILED，发现选中灰度图，未发布数据目录 |
| 28193727 | 审计完整选样，不转换或替换图片 | COMPLETED，bun147，4 秒 |
| 28194446 | 审计全部 1300 训练候选并依 seed 43 补齐 | COMPLETED，bun147，8 秒 |
| 28194475 | 发布 400 张合格训练图，保留原验证清单 | COMPLETED，bun147，5 秒；验证格式仍阻塞 |

本地原有 19 项 collector/tracker 测试通过。新增 launcher、清单、哈希、
错误处理与 reference collector 测试通过。核心科研代码、用户原有
hpc/check_job.sh 与 outputs 内容未修改。

数据源已通过 compute job 确認为：
/scratch/licenseddata/imagenet/imagenet-1k/{train,val}/n02099601。
候选数量为 1300 train、50 val。用户确认 seed 43。

实际选中的 400 张训练图、固定全部 50 张验证图的清单和哈希在：
- artifacts/bunya/deployment-a5e7f84/selection-28193727/training_images.txt
- artifacts/bunya/deployment-a5e7f84/selection-28193727/validation_images.txt
- artifacts/bunya/deployment-a5e7f84/selection-28193727/selection_audit.json

本地复核：400/50 个唯一源文件哈希；两个 split 的路径交集和哈希交集均为 0。
这些是已确定的源图清单，尚不是通过全部格式检查并发布的运行输入目录。
清单记录 random.Random(43).sample(sorted filenames)、候选文件名哈希及 Python 版本。

选中图片中有 4 张训练灰度图：
- n02099601_5895.JPEG
- n02099601_4193.JPEG
- n02099601_3817.JPEG
- n02099601_353.JPEG

验证集中有 1 张灰度图：ILSVRC2012_val_00019590.JPEG。
其余所选图片通过解码/RGB 检查。原代码的 ImageClass 不会自动转换为 RGB。
后续用户已明确：训练按 seed 43 固定候选顺序跳过灰度图并补齐；验证清单不变。
完整候选审计发现 11/1300 张灰度图（0.846154%），本次跳过 4 张，取至第 404 个候选补齐 400 张。
所有灰度图预览未见明显特殊场景集中，据此继续准备；有限视觉检查不证明完全无偏。
已发布 /scratch/user/uqcche38/hu-mcd/data/reference-golden-seed43-rgb。
验证 50 张的路径、顺序和哈希保持原样，其中 1 张 L 图仍使 ready_for_reference=false。
未做灰度转 RGB 或对比实验。完整记录、清单、作业身份见 docs/TRAINING_SELECTION_SEED43.md。

已安装环境为 torch 2.2.2+cu121、timm 0.6.13。
两份 installed masking 文件与基线 input_masking 内容哈希完全相符。
已存在的 ResNet50 权重 SHA256：
14fe96d1f9fb311a60490082d2077e6e60427dcfe21839ddf934cce948f72b0f。
项目模型目录当前只有 SAM ViT-B，没有 ViT-H。

## Launcher

hpc/submit_release.py 默认为只显示计划；加 --submit 才提交。
所有资源参数和精确 commit 必须显式提供。研究代码来自：
/scratch/user/uqcche38/hu-mcd-git/releases/FULL_SHA/code。

例如，只生成只读检查计划：
~~~bash
python3 hpc/submit_release.py \
  --commit a5e7f843856d17b7647f8af6571319b45719f669 \
  --mode inspect \
  --dataset-root /scratch/licenseddata/imagenet/imagenet-1k \
  --cpus 1 --memory 2G --time 00:05:00 --partition general --qos debug
~~~

可用模式：
- inspect：有界目录元数据、环境版本、已安装 masking 与 classifier 权重检查。
- inspect-selection：保存完整拟选清单及所有格式错误，不发布输入目录。
- prepare-data：严格 RGB 校验、数量/哈希/重复校验全部通过后，原子发布 symlink 布局。
- audit-training：绑定历史选样 SHA256，检查全部训练候选、固定验证集并记录补齐和跳过信息。
- publish-training：绑定审阅后的审计 SHA256，再校验图片并发布 symlink；保留验证异常并阻止正式启动。
- reference：要求完整论文配置、可校验的 400/50 manifest 和已就绪模型；运行 pinned run_smoke.py。

数据选样使用用户确认的 seed 43。
prepare-data / inspect-selection 还必须指定 --prepared-root 与 --seed 43。
共享数据源只读。目标目录已存在时拒绝覆盖。
prepare-data 当前仍会因上述灰度图失败，不应重复提交。

所有远程工作先验证 Slurm allocation 和 bunNNN compute hostname。
提交通道在登录节点仅调用 sbatch；没有在登录节点执行 Python、Git fetch、
目录扫描、hash、模型加载、图像处理或安装。Git 准备也在 CPU allocation 内完成。
用户登录/MFA 保持手动；SSH ControlMaster 失效时拒绝新登录。
连接在提交过程中失败时先查 squeue，不自动重试提交。

启动记录写到 runtime_root/launches/JOB_ID/launch_manifest.json，
含实际科研 SHA、辅助代码哈希、已安装 masking 身份、资源请求及状态。
科研输出继续用 outputs/runs/JOB_ID，缓存用 cache/runs/JOB_ID。

## Reference collector

未来 reference 作业使用：
~~~bash
python3 hpc/collect_results.py JOB_ID \
  --profile reference --run-layout \
  --expected-commit FULL_RESEARCH_SHA --include-scientific --max-file-mib 256 --watch
~~~

collector 同时检查 launcher 和 producer 中的 job/commit，
要求科研 checkout 干净，仍校验 resolved config、文件索引与结果哈希。
新 reference profile 的 job name 是 humcd-reference，日志前缀是 reference。
旧 smoke/gpu-probe 用法保持可用。

## 尚未完成的条件

1. 训练筛选和 400/50 symlink 目录已完成。需确认固定验证集中 1 张 L 图的通道处理，不能静默替换或转换。
2. 准备和核对 SAM ViT-H；资源探测后确定 batch size、CPU/RAM/time。
   reference.template.json 中相关字段故意保留 null，launcher 会拒绝未填写的配置。
3. 科学结果记录与 feature/logit/relevance 重构检查仍未补齐。
   a5e7f84 没有这些功能；若修改科研代码，必须明确使用新的 commit，不能称其仍是 a5e7f84。
4. C-Insertion/C-Deletion、局部 relevance 与基线评价仍未执行。

未提交 GPU 作业或 400-image 参考实验。当前进度是：
精确版本部署、launcher 和 400 张训练数据准备已验证；固定验证集的 1 张 L 图仍需处理决定。
