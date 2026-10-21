# 阅读与证据

理解模型的计算过程，先读 01—07 的教学 HTML 或对应 Notebook；它们已全部交付。下列阶段 Markdown 报告提供完整表格与来源，预测卡片保留当时的预测和决定。卡片中的“下一步”“待审”属于历史时点；已收窄的旧解释在旁边标注更正，不能当作当前已验证结论。

各文件明确区分本机 Mac M4 Max 的 MPS 运行、CPU 核验、保存数据的派生计算与预测。对应设备、精度和适用范围以各报告的记录为准。

| 文件 | 内容 | 性质 |
|---|---|---|
| `00_orientation.html` | 仓库导读与载荷账本 | 不是实测 |
| `01_environment_and_attention.html` | 人工数组上的分项计时与完整注意力读取对照：怎样取 K/V，选择成本是否抵消少读的收益 | 读取下列实测结果，CPU 小例子 |
| `g0_mac_m4max.md` | G0：构造数组，分别测索引打分、选位、取 K/V 及短链；另记录平台支持和带宽，最后据此预测 | 实测 + 预测 |
| `g1a_mac_m4max.md` | G1a：在同一主 Q/K/V 上比较完整注意力与索引选择后注意力，计时止于 V 加权输出 | 实测 |
| `g1a_supplement_mac_m4max.md` | G1a 补测：分别改变 A/D 执行顺序、缓存预留容量和完整注意力写法，检查比较是否受这些条件影响 | 实测 |
| `02_qwen_integration_and_decode.html` | 真实 Qwen 处理固定随机 token，数值检查与整步测速；解释缓存如何建立、恢复和连续增长 | 读取下列 G1b 实测结果，tiny 模型 CPU 小例子 |
| `g1b_mac_m4max.md` | G1b：比较原库/封装/全选的数值；历史先由真实模型计算，再测固定长度单步与固定输入的 128 步 | 实测 |
| `g1b_numerics_mac_m4max.json`、`g1b_summary_mac_m4max.json`、`g1b_manifest_mac_m4max.json` | G1b 精简数据与 manifest（含 model / tokenizer revision、权重哈希） | 实测 |
| `03_what_is_lost_when_reading_less.html` | 读编号—数值资料后各自回答，比较四种历史读取方式；展示选位、错答和资料改写 | 读取下列 G2-lite 实测结果，tiny 模型 CPU 契约重放 |
| `g2lite_mac_m4max.md` | G2-lite 报告：基线可用性、四方法对照与配对四格、覆盖率、代表案例、反事实改写、成本 | 实测（48 份探索样本，无区间统计） |
| `g2lite_summary_mac_m4max.json`、`g2lite_answers_mac_m4max.json`、`g2lite_manifest_mac_m4max.json` | G2-lite 精简表格与案例、逐题回答与解析、三遍 run 的 manifest（含 model / tokenizer revision、配置版本、数据 digest） | 实测 |
| `04_can_an_indexer_learn.html` | 实验课04：第14层索引器第一次训练，冻结特征上的教师模仿与局部输出 | 新 G3a 48/16 资料；不是 G2-lite 重跑或整模型能力评测 |
| `g3a_mac_m4max.md`、`g3a_summary_mac_m4max.json`、`g3a_cpu_audit.json` | 单层结果、逐资料/逐组指标、真实选位、独立 NumPy 核验 | 源自新的 G3a 原始记录 |
| `05_training_all_layer_indexers.html`、`g3b_mac_m4max.md` | 完整模型分别提供每层特征，27 层各自训练、第 14 层复用；比较固定输入上的名单与局部读出 | 旧开发集描述，不代表真实稀疏生成 |
| `06_from_teacher_features_to_sparse_states.html`、`g3c_mac_m4max.md` | 接回训练参数后，各方法输入相同 token、维护各自缓存；比较逐层 Q/K、读出和词表分数 | 4份旧开发资料的控制序列，不是答题成绩 |
| `07_trained_sparse_quality_and_cost.html`、`g4lite_mac_m4max.md`、`g4lite_answers_mac_m4max.json` | 32 份新编号取值题、160 条自由回答；另用固定控制输入测成本，保留 96 条成本记录 | 真实自由生成成绩与同token成本分列 |
| `cards/G0_mac_m4max.md`、`cards/G1a_mac_m4max.md`、`cards/G1b_mac_m4max.md`、`cards/G2lite_mac_m4max.md` | 关卡卡片：运行前预测、运行后观察、解释、决策 | 预测写于运行前 |
| `g0_*.json`、`g1a_*.json`、`g1a_supplement_*.json` | 上述报告的精简数据与 manifest | 实测 |
| `g0_roofline_mac_bw390.{md,json}` | 以实测 390 GB/s 回填的载荷情景 | 预测 |
| `provenance/` | 跑分时的基准源码快照、跑分后 diff、原始 run 归档与哈希清单 | 证据链，`scripts/check_provenance.py` 校验 |
| `p0_score_kernel_mac_m4max.md`、`p1_attend_kernel_mac_m4max.md` | 旧 Metal 打分与取行注意力原型；文件中的 P0/P1 是阶段名，不是索引位置设置 | 已停止，不接入正式模型 |
| `bootstrap_validation.md`、`public_publication.md` | 启动工具与发布记录 | 不是 GPU 结果 |

重新生成：`python scripts/g0_report.py`、`python scripts/g1a_report.py`、`python scripts/g1a_supplement_report.py`、`python scripts/g1b_report.py`、`python scripts/g2lite_report.py`。无 GPU 从归档重生 G1a 报告的步骤见 `provenance/README.md`。

G3a 报告重建：`python scripts/g3a_report.py`（标准库，不导入 torch）。完整特征保留在本机原始 run；精简归档含数据、参数、指标及原位置名单。

G3b报告重建：`python scripts/g3b_report.py`（标准库，不导入torch）。参数来源与身份见 `g3b_parameters.json`。

第四课3a CPU教学补充：[完整20步中间记录](lesson04_cpu_trace.json)，调用同一正式索引器、教师与主注意力读取实现。12个临时参数、无固定集合、只选一个位置，不修改G3a正式数据或索引参数。

默认执行的差别：01—03 读取保存结果并运行小规模 CPU 检查；04 另执行上述 20 步 CPU 更新；05—07 读取原记录与作图。都不自动加载真实 0.6B 模型、重训正式索引器或启动 GPU 基准。长记录在 HTML 中可展开查看，完整数据入口不变。
