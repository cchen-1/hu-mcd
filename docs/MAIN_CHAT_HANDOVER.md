# Main-chat handover — 2026-09-08

## 本次范围与交付状态

用户在侧对话授权：整理、检查现有复现代码，提交并推送到自己的 fork，
然后回到主对话推进部署和实验。本次不执行 Bunya 部署，不提交 Slurm 作业。

本次代码版本包含：本地结果 collector、每次运行独立输出/缓存目录、
配置与代码/输入文件记录、阶段进度、Git release 准备工具及现有诊断记录。
提交的精确 SHA 和推送结果见侧对话最终消息；本文件随该提交保存。

- 工作目录：`/home/chen/projects/hu-mcd`
- 分支：`reproduction`
- fork：`https://github.com/cchen-1/hu-mcd.git`
- upstream：`https://github.com/grobruegge/hu-mcd.git`
- 已核对的 upstream commit：`168eb5bf1717ab46d1c84be86bd81f0c51861413`
- 本地保留但不随本次提交发布：`outputs/`、`artifacts/` 中的实验结果、
  数据衍生图片和缓存，以及独立的 NotebookLM 文件。
- 用户原有的 `hpc/check_job.sh` 修改原样纳入提交；原有 outputs 文件内容未改动。

## 用户刚提供的信息

Bunya 上可用的 ImageNet1k 路径：

```text
/scratch/licenseddata/imagenet/imagenet-1k
```

路径来自用户。本次没有登录远端验证它的读取权限、目录结构、解包状态、
标签映射或 train/validation 划分。不要假设它与 Imagewoof 的目录结构相同。
后续在 Slurm compute 作业内检查并建立选样清单；保持共享数据源只读。

用户表示目前已登录 Bunya。SSH 登录和 MFA 始终由用户完成。
后续实操前检查 `ssh -O check bunya`，复用现有 ControlMaster；
它可能在主对话恢复时已过期，不自动发起新登录。

## 研究方向

侧对话建议优先推进单类别论文配置的参考复现，用户希望回主对话继续实操：

- golden_retriever（ImageNet synset `n02099601`）；
- 原始 ImageNet training split 中固定种子随机选取 400 张；
- 对应 validation split 中 50 张，检查与训练清单不重合；
- SAM ViT-H，32 points per side；
- 保持已固定的 released-code 行为，包括最小 cluster size 50、
  masked ResNet50/global_pool、自动 SSC/PCA 等；
- 先分析该类别，再决定扩展十类和 ACE/MCD 对照。

这是下一里程碑的方案，不代表完整论文已经复现，也不证明此前零验证
分配只由数据量导致。Supplement 的 golden-retriever 参考是 13 clusters、
completeness 0.67；不要强制设成 13 或围绕 0.67 调参以获得表面一致。
原论文的完整采样 ID 和历史环境未提供，逐位一致不是已确定的验收条件。

## 已有能力与待接通处

1. `hpc/prepare_git.sbatch` 与 `hpc/submit_git_prepare.sh`
   能在 CPU Slurm 作业中从 fork 取得指定 commit，产生
   `/scratch/user/uqcche38/hu-mcd-git/releases/FULL_SHA/code`。
   旧版 commit f22adf2 的准备作业 28159694 已有成功记录。
   **此次新提交尚未在 Bunya 准备。**
2. `hpc/phase3_smoke.sbatch` 仍然写死旧目录
   `/scratch/user/uqcche38/hu-mcd`，并使用 10/5、ViT-B smoke 配置。
   **不要直接把该脚本当成新版代码或论文规模 launcher。**
   下一步要让 launcher 显式选择 prepared release，复用既有模型/环境，
   并引用新的参考配置。
3. `run_smoke.py` 已接入 `utils/run_tracking.py`。
   Slurm 下要求 `--run-id "$SLURM_JOB_ID"`，输出和缓存分别放到
   对应根目录的 `runs/JOB_ID`；记录 resolved config、Git commit/dirty
   state、选定源码和实际读取图片的 hashes。启动/阶段完成/异常/结束写进度。
   新 producer 已有本地测试，尚无新版本远端完整执行的证据。
4. `hpc/collect_results.py --run-layout` 已支持新目录和身份/哈希检查，
   通过 `--watch` 轮询。现有 profile 仍要求 smoke 的 job name、
   日志命名和结果文件布局。新 launcher 改动这些时，必须一并适配 collector。
   它只用 sacct、squeue、SFTP；不做远程文件处理，不提交或取消作业。
5. 原有数学路径未因本次整理而修改。拟合的 bases、cluster labels、
   segment-to-image mappings、局部 relevance、环境/模型完整身份等额外
   科学记录，仍需按审计文档补齐。
6. `outputs/` 和下载快照留在本地，不在 Git 中。诊断脚本依赖那些历史
   文件，因此它是特定缓存的复核工具，不是 fresh clone 即可运行的实验入口。

## 建议主对话的执行顺序

- 阅读 `docs/PAPER_REPRODUCTION_AUDIT.md` 和 `docs/VALIDATION_DIAGNOSIS.md`。
- 以侧对话最终消息中的已推送 SHA 为起点，检查当前主对话是否又产生新改动；
  若有，不覆盖它们，不假设旧 SHA 包含新改动。
- 完成指定 release 的 launcher、参考配置、collector 路径的衔接，
  先用本地测试验证身份、配置、路径和进度对应。
- 在后续获得运行指示的范围内，用 Slurm 检查 ImageNet 结构、
  准备确定的图片清单和 ViT-H 权重，核对安装的 masking 代码与模型身份。
- 做必要的 feature/logit 重构与评估边界检查，然后安排单类别参考实验。
  新增计算或文件处理均在 Slurm allocation 内完成。
- 根据真实耗时/内存安排资源；不要把 10-image ViT-B smoke 的耗时线性外推到
  ViT-H/400-image。SSC 的 dense coherence matrix 随片段数平方增长。
- 保存并收集日志、manifest、原始科学结果和图。后续再推进
  C-Insertion/C-Deletion、ACE/MCD；人工可理解性研究另列范围。

## 本次验证

- collector / run tracking 的 19 项 unittest 全部通过；
- `run_smoke.py`、tracker、collector、缓存诊断脚本的 Python 语法通过；
- 所有现有 HPC .sh/.sbatch 的 bash 语法通过；
- 提交前检查 whitespace、文件范围与原有结果保留情况；
- 不把本地测试等同于远端部署成功，不把运行 PASS 等同于科学复现成功。
