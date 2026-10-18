# 阅读与证据

模型成绩与性能数字来自保存的Mac M4 Max/MPS运行，或明确标注的情景预测；课堂CPU算例另标，不冒充MPS实测。默认直接阅读已保存HTML，不执行模型。三课的版本、Notebook和未验范围见[课程入口](../notebooks/README.md)。

| 文件 | 内容 | 性质 |
|---|---|---|
| [00_orientation.html](00_orientation.html) | 仓库导读与载荷账本 | 不是实测 |
| [01_environment_and_attention.html](01_environment_and_attention.html) | 第一份实验课：G0 与 G1a 的三个问题，已执行的 Notebook 导出 | 读取下列实测结果，CPU 小例子 |
| `g0_mac_m4max.md` | G0 报告：环境、能力、带宽、索引路径原语成本、回填预测（含公式更正） | 实测 + 预测 |
| `g1a_mac_m4max.md` | G1a 报告：单层 A 与 D 六进程微基准、通过判定、剖析 | 实测 |
| `g1a_supplement_mac_m4max.md` | G1a 补测：交错顺序、容量四格、折叠手写 A 对照 | 实测 |
| [02_qwen_integration_and_decode.html](02_qwen_integration_and_decode.html) | 第二份实验课：G1b 把索引注意力装进 Qwen3-0.6B，整模型一步为何不等于单层乘层数 | 读取下列 G1b 实测结果，tiny 模型 CPU 小例子 |
| `g1b_mac_m4max.md` | G1b 报告：三段数值对照、Prefill、固定 N 单步与连续 128 步整步耗时、内存、剖析、判定 | 实测 |
| `g1b_numerics_mac_m4max.json`、`g1b_summary_mac_m4max.json`、`g1b_manifest_mac_m4max.json` | G1b 精简数据与 manifest（含 model / tokenizer revision、权重哈希） | 实测 |
| [03_what_is_lost_when_reading_less.html](03_what_is_lost_when_reading_less.html) | 第三份实验课：G2-lite，少看历史会丢掉什么——四种读取方式在 T1 / T2 上的质量对照、覆盖率、代表案例、反事实改写 | 读取下列 G2-lite 实测结果，tiny 模型 CPU 契约重放 |
| `g2lite_mac_m4max.md` | G2-lite 报告：基线可用性、四方法对照与配对四格、覆盖率、代表案例、反事实改写、成本 | 实测（48 份探索样本，无区间统计） |
| `g2lite_summary_mac_m4max.json`、`g2lite_answers_mac_m4max.json`、`g2lite_manifest_mac_m4max.json` | G2-lite 精简表格与案例、逐题回答与解析、三遍 run 的 manifest（含 model / tokenizer revision、配置版本、数据 digest） | 实测 |
| `cards/G0_mac_m4max.md`、`cards/G1a_mac_m4max.md`、`cards/G1b_mac_m4max.md`、`cards/G2lite_mac_m4max.md` | 关卡卡片：运行前预测、运行后观察、解释、决策 | 预测写于运行前 |
| `g0_*.json`、`g1a_*.json`、`g1a_supplement_*.json` | 上述报告的精简数据与 manifest | 实测 |
| `g0_roofline_mac_bw390.{md,json}` | 以实测 390 GB/s 回填的载荷情景 | 预测 |
| `provenance/` | 跑分时的基准源码快照、跑分后 diff、原始 run 归档与哈希清单 | 证据链，`scripts/check_provenance.py` 校验 |
| `bootstrap_validation.md`、`public_publication.md` | 启动工具与发布记录 | 不是 GPU 结果 |

重新生成：`python scripts/g0_report.py`、`python scripts/g1a_report.py`、`python scripts/g1a_supplement_report.py`、`python scripts/g1b_report.py`、`python scripts/g2lite_report.py`。无 GPU 从归档重生 G1a 报告的步骤见 `provenance/README.md`。

当前版本：01/02保存内容为 `ff31d3a`（PR #4校正版）；03保存内容为 `a3832a8`，使用旧v2运行 `af62df4` 与报告生成代码 `72b848d`。PR #5已在 `d2fc4ad` 合并；当前配置v3仍NOT_RUN。整页第三课布局未验，已有图像检查不替代整页验收。#6–#9的诊断保留在各自固定提交，不加入这里的冻结主结果。
