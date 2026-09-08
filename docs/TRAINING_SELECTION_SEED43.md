# Golden retriever：seed 43 训练选样与格式审计

后续输入兼容处理和 ViT-H / GPU 探测进度见 `REFERENCE_RUN_PREPARATION.md`。本文件保留训练筛选阶段的历史状态。

记录更新：2026-09-09。科研 release 为
`a5e7f843856d17b7647f8af6571319b45719f669`。
本轮新增 launcher/worker 属于尚未提交的本地辅助代码，内容哈希独立写入作业记录；
没有改写该 release，也没有提交 GPU 训练或灰度转 RGB 对比实验。

## 已完成与证据

| 项目 | 实测结果 |
|---|---|
| 类别 | golden_retriever / n02099601 |
| 完整训练候选 | 1,300 |
| 灰度图 | 11 / 1,300 = 0.846154% |
| 灰度类型 | 11 张 PIL L；RGB 三通道逐像素完全相等的图片为 0 |
| 其他格式或解码异常 | 0 |
| 最终训练图片 | 400，全部 RGB，路径和 SHA256 各自唯一 |
| 本次选样扫描数量 | 前 404 个候选 |
| 本次跳过灰度图 | 4 张；无其他跳过原因类别 |
| 验证图片 | 原清单 50 张，路径、顺序、命名和 SHA256 保持一致 |
| 训练/验证交集 | 路径 0，内容 SHA256 0 |
| 验证格式异常 | 1 张 L，保留原文件并报告 |

灰度定义为 PIL 灰度模式，或 RGB 图片每个像素的三通道完全相等。
没有使用低饱和度阈值；没有把彩色低饱和度照片自动归为灰度。
统计分母是该类别 train 目录中全部 JPEG/JPG/PNG 候选，而不是拟选的 400 张。

| Slurm job | 工作 | 节点 | 状态 / 耗时 |
|---|---|---|---|
| 28194446 | 全量解码、哈希、选样、预览图生成 | bun147 | COMPLETED / 8 秒 |
| 28194475 | 核对审阅过的审计哈希，再次检查输入，发布私有 symlink 目录 | bun147 | COMPLETED / 5 秒 |

两个作业均请求 1 CPU、2 GiB、5 分钟，general/debug，无 GPU。
执行前已确认 SSH master 仍有效；通过现有连接复用，不发起新登录或 MFA。
远端仅在登录节点调用 sbatch/sacct；所有目录扫描、图像处理、哈希和目录发布均在 Slurm 计算节点。
结果通过 SFTP 回收，本地进行统计复核和预览图查看。

## 固定候选顺序与补齐

对排序后的全部 1,300 个训练候选执行
`random.Random(43).sample(candidates, len(candidates))`，顺序向后取首 400 张合格图片。
审计显式验证这个完整顺序的前 400 项与历史作业 28193727 的原始选样完全相同；
不假定不同 sample 长度在任意候选规模下都具有此前缀性质。
还核对历史候选文件名哈希、已选训练图片内容和全部验证图片内容。

冻结的历史清单 SHA256：
`c860dc749e572f770ebc7b4ae703fcbc3a60c4ae0cc8687f77f6cae4c2f04e79`。

| 候选位置 | 跳过文件 | 原因 |
|---|---|---|
| 133 | n02099601_5895.JPEG | grayscale_mode；unsupported_mode:L |
| 165 | n02099601_4193.JPEG | grayscale_mode；unsupported_mode:L |
| 263 | n02099601_3817.JPEG | grayscale_mode；unsupported_mode:L |
| 363 | n02099601_353.JPEG | grayscale_mode；unsupported_mode:L |

补入第 401–404 个候选：
`n02099601_3901.JPEG`、`n02099601_5366.JPEG`、
`n02099601_2029.JPEG`、`n02099601_7916.JPEG`。
原选样中的 396 张合格 RGB 图片全部保留，顺序保持。

## 初步场景检查及筛选影响

本地查看了计算节点生成的全部 11 张灰度图预览，以及选中训练集前 16 张 RGB 图片的上下文预览。
灰度图涵盖室内休息、头部/面部特写、庭院及其他户外场景；有些为黑白肖像风格。
没有发现明显集中于某个罕见特殊场景。头部特写和休息姿态在 RGB 上下文预览中也存在。
这是有限的人工视觉检查，不是全量场景标注或统计检验，不能证明筛选无偏。

删除灰度候选必然缩小色彩呈现的多样性，因此应把“跳过灰度后补齐”作为复现偏差记录。
根据全量占比低于 1%、本次仅替换 4/400 张，以及未见明显特殊场景集中，按用户已批准规则继续准备训练集。
没有开展、也没有自动安排灰度转 RGB 对比实验。

## 验证集异常与启动阻塞

`ILSVRC2012_val_00019590.JPEG`：PIL L，415×500，原清单第 20 张。
SHA256：`ff191fd4f872734df5bab18cc8feef77d12d14ad163c85781cc685ad752e3614`。
预览显示户外草地上的狗，文件可以解码；问题是原 pipeline 所需的三通道输入格式。
此文件没有删除、替换或转换。全部 50 张验证图按原清单建立 symlink。

发布的 manifest 明确记录 `ready_for_reference: false` 和 `validation_issues`。
launcher 的正式运行输入检查会在模型计算前拒绝此状态，防止把数据准备成功误报为正式实验已就绪。
后续需要用户确定如何处理这张固定验证图的输入通道；该决定独立于是否开展灰度对比实验。

## 文件位置及身份

远端已发布目录：
`/scratch/user/uqcche38/hu-mcd/data/reference-golden-seed43-rgb`。
其中训练目录为 `golden_retriever/`，验证目录为 `val_imgs/golden_retriever_val/`。
目录名中的 rgb 指训练筛选；验证图仍保留原 L 文件。
这些是指向共享许可数据源的 symlink，没有修改源文件，也没有覆盖之前的审计。
它不是 scratch 清理之外的持久数据副本；本地清单用于以后校验或重建。

已审阅训练审计 SHA256：
`1875d627b346e0ed06e95af01eaba742e46b4d23dca89b85ea5d7e1c31c1a865`。
已发布 dataset_manifest SHA256：
`962bc882e20ceef6785ced92a4f1245f8ad93318646fbb0e944810e1f2c95b1f`。

本地证据保存在 `artifacts/bunya/deployment-a5e7f84/`：

- `training-audit-28194446/training_audit.json`：完整候选顺序、模式、尺寸、文件哈希、选样决定、最终清单及验证异常。
- `training-audit-28194446/candidate_order.txt`：完整 1,300 项候选顺序。
- `training-audit-28194446/skipped_training.json`：本次跳过文件、候选位置及原因。
- `training-audit-28194446/training-grayscale-01.jpg`：全部 11 张灰度图预览。
- `training-prepared-28194475/training_images.txt`：实际发布的 400 张训练清单。
- `training-prepared-28194475/validation_images.txt`：实际发布的原 50 张验证清单。
- `training-prepared-28194475/dataset_manifest.json`：发布的完整输入与审计记录。
- 两个目录中的 `launch_manifest.json`、`sacct.psv`、`job.out`、`job.err`：辅助代码身份、release、资源和执行证据。

以上本地 artifacts 受现有 .gitignore 排除，图片衍生预览没有进入 Git。
本轮辅助实现位于 `hpc/audit_training_candidates.py`、`hpc/submit_release.py` 和 `hpc/release_worker.py`。
新模式 `audit-training` 必须指定冻结清单路径及 SHA256；
`publish-training` 必须指定已审阅审计路径及 SHA256 和新目标目录。
两个模式只分配 CPU；所有上传的辅助代码内容哈希随作业保存。

## 验证与后续

38 项 HPC unittest 通过，其中新增 7 项覆盖全量/选样统计区别、补齐顺序、
冻结验证集、身份变化拒绝、内容重合跳过、RGB 灰度识别、损坏图片、
发布前输入变化以及未解决验证格式时的启动阻塞。
原 hpc/check_job.sh 和 outputs 中合计 35 个文件的内容哈希保持不变。

当前已完成训练数据准备。正式单类别参考运行仍需：

1. 明确这张固定验证图的通道处理，保持验证 ID 不变并记录处理方式。
2. 准备核对 SAM ViT-H 权重，通过资源探测确定配置；目前仅已核实 ViT-B 和 ResNet50 缓存。
3. 补齐既有审计指出的科学中间结果记录与重构检查；若改变科研代码，应使用新提交身份。
4. 再提交单类别论文配置运行与后续评价。当前没有提交新的 GPU 实验。
