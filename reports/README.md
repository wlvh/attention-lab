# 阅读与证据

各文件明确区分本机 Mac M4 Max 的 MPS 运行、CPU 核验、保存数据的派生计算与预测。对应设备、精度和适用范围以各报告的记录为准。

| 文件 | 内容 | 性质 |
|---|---|---|
| `00_orientation.html` | 仓库导读与载荷账本 | 不是实测 |
| `01_environment_and_attention.html` | 第一份实验课：G0 与 G1a 的三个问题，已执行的 Notebook 导出 | 读取下列实测结果，CPU 小例子 |
| `g0_mac_m4max.md` | G0 报告：环境、能力、带宽、索引路径原语成本、回填预测（含公式更正） | 实测 + 预测 |
| `g1a_mac_m4max.md` | G1a 报告：单层 A 与 D 六进程微基准、通过判定、剖析 | 实测 |
| `g1a_supplement_mac_m4max.md` | G1a 补测：交错顺序、容量四格、折叠手写 A 对照 | 实测 |
| `02_qwen_integration_and_decode.html` | 第二份实验课：G1b 把索引注意力装进 Qwen3-0.6B，整模型一步为何不等于单层乘层数 | 读取下列 G1b 实测结果，tiny 模型 CPU 小例子 |
| `g1b_mac_m4max.md` | G1b 报告：三段数值对照、Prefill、固定 N 单步与连续 128 步整步耗时、内存、剖析、判定 | 实测 |
| `g1b_numerics_mac_m4max.json`、`g1b_summary_mac_m4max.json`、`g1b_manifest_mac_m4max.json` | G1b 精简数据与 manifest（含 model / tokenizer revision、权重哈希） | 实测 |
| `03_what_is_lost_when_reading_less.html` | 第三份实验课：G2-lite，少看历史会丢掉什么——四种读取方式在 T1 / T2 上的质量对照、覆盖率、代表案例、反事实改写 | 读取下列 G2-lite 实测结果，tiny 模型 CPU 契约重放 |
| `g2lite_mac_m4max.md` | G2-lite 报告：基线可用性、四方法对照与配对四格、覆盖率、代表案例、反事实改写、成本 | 实测（48 份探索样本，无区间统计） |
| `g2lite_summary_mac_m4max.json`、`g2lite_answers_mac_m4max.json`、`g2lite_manifest_mac_m4max.json` | G2-lite 精简表格与案例、逐题回答与解析、三遍 run 的 manifest（含 model / tokenizer revision、配置版本、数据 digest） | 实测 |
| `04_can_an_indexer_learn.html` | 实验课04：第14层索引器第一次训练，冻结特征上的教师模仿与局部输出 | 新 G3a 48/16 资料；不是 G2-lite 重跑或整模型能力评测 |
| `g3a_mac_m4max.md`、`g3a_summary_mac_m4max.json`、`g3a_cpu_audit.json` | 单层结果、逐资料/逐组指标、真实选位、独立 NumPy 核验 | 源自新的 G3a 原始记录 |
| `cards/G0_mac_m4max.md`、`cards/G1a_mac_m4max.md`、`cards/G1b_mac_m4max.md`、`cards/G2lite_mac_m4max.md` | 关卡卡片：运行前预测、运行后观察、解释、决策 | 预测写于运行前 |
| `g0_*.json`、`g1a_*.json`、`g1a_supplement_*.json` | 上述报告的精简数据与 manifest | 实测 |
| `g0_roofline_mac_bw390.{md,json}` | 以实测 390 GB/s 回填的载荷情景 | 预测 |
| `provenance/` | 跑分时的基准源码快照、跑分后 diff、原始 run 归档与哈希清单 | 证据链，`scripts/check_provenance.py` 校验 |
| `bootstrap_validation.md`、`public_publication.md` | 启动工具与发布记录 | 不是 GPU 结果 |

重新生成：`python scripts/g0_report.py`、`python scripts/g1a_report.py`、`python scripts/g1a_supplement_report.py`、`python scripts/g1b_report.py`、`python scripts/g2lite_report.py`。无 GPU 从归档重生 G1a 报告的步骤见 `provenance/README.md`。

G3a 报告重建：`python scripts/g3a_report.py`（标准库，不导入 torch）。完整特征保留在本机原始 run；精简归档含数据、参数、指标及原位置名单。
