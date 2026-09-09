# Nine-class input protocol / 九类输入协议

The user approved all five flagged training selections and all eight validation compatibility adjustments on2026-09-09. Preserve each audited seed43 first400eligible list, and all50 original validation IDs. No training grayscale conversion comparison is authorized.

用户已批准：维持seed43候选顺序，跳灰度补齐400张后冻结；场景偏差只列为未验证风险。八张验证L图在私有目录复制灰度值为三个通道、无损PNG，逐图验证尺寸与像素；原图/ID/清单保留。审计、发布、转换与验证均在Slurm计算节点。

| Class | Candidates | Grayscale | Fraction | Skipped to400 | L validation |
|---|---:|---:|---:|---:|---:|
| Siamese_cat | 1300 | 11 | 0.846% | 6 | 0 |
| airliner | 1300 | 8 | 0.615% | 4 | 0 |
| beach_wagon | 1300 | 16 | 1.231% | 5 | 1 |
| container_ship | 1300 | 15 | 1.154% | 3 | 0 |
| hummingbird | 1300 | 0 | 0.000% | 0 | 0 |
| ox | 1300 | 28 | 2.154% | 11 | 2 |
| police_van | 1300 | 6 | 0.462% | 4 | 0 |
| tailed_frog | 1300 | 3 | 0.231% | 3 | 0 |
| zebra | 1300 | 66 | 5.077% | 22 | 5 |

## Difference from paper and released code / 与论文、原仓库的差异

Paper §4 implementation details specify400 images randomly sampled per class from ImageNet1k training. In the inspected paper and supplement, no explicit grayscale-exclusion rule or historical sample IDs were found. The paper statement does not establish how its authors handled non-RGB files. Source PDFs and text are archived under artifacts/bunya/final-28208840/report/support; this is a bounded document observation, not proof of the authors' undocumented choices.

Released commit168eb5bf1717ab46d1c84be86bd81f0c51861413: ConceptExplainer._load_images sorts filenames; discovery does not request shuffle. It catches image-loader exceptions, prints a warning, and keeps scanning until the requested count. ImageClass.load_image opens without RGB conversion and requires array shape[2]==3: an L image has no third axis and fails. RGB files with exactly equal channels are accepted upstream, whereas our explicit training policy excludes them. Our numbered prepared filenames implement the seed43 frozen order while retaining the original source IDs. Therefore this is a documented paper-scale reproduction with a different controlled input-selection protocol, not recovery of the paper's original400identities or an exact raw-directory execution of upstream.

论文说随机抽400张，未在已核对正文/补充中明确灰度筛选，也没有原抽样ID。原仓库按文件名排序，加载异常时警告并跳过；没有主动将L图转RGB。原仓库可接受三通道完全相等的RGB灰度图，而当前训练规则也排除它们。候选随机顺序、显式资格规则、补样日志和冻结清单属于我们的输入协议；不可描述为论文原样抽样。验证兼容调整保持像素/ID，避免原加载器静默丢图，单独记录工程输入调整。

Qualitative thumbnail review noted historical-looking aircraft/vehicles in some airliner/police_van/beach_wagon grayscale images, many patterned closeups among zebra, and working/display scenes among ox. We have not measured enrichment relative to colour candidates, so cannot conclude subtype or scene bias exists. No result-driven reselection, balancing, threshold change or training conversion was made. Counts above are full-candidate facts; scene-distribution effects remain unverified risks accepted by the user.

全候选灰度数量与本次实际跳过数量不同：筛选只扫描到补齐400张；保留完整候选顺序与每张决策。数据证据：Slurm28214251，commit94369b79a78ca534433da622268c3a9afea9d293，collected/28214251/20260909T050428.482463Z/inputs。全十类原始文件SHA核对为4000/500无重复且训练/验证不交叉；没有声称完成视觉近重复检测。
