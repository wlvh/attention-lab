# 阅读与证据

所有数字来自本机 Mac M4 Max 的 MPS 实测，或由实测回填的预测；每个文件都写明了自己是哪一种。不要凭空填性能值。

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
| `g2lite_replay_header_mac_m4max.md` | G2-lite 单变量重放：把表头钉进近期窗口 B 的名单（BH，读取数与 B 相同）并以缩短同样名额的 B（BM）作对照，与归档的第二遍 B 配对，只回答「B 在窗口内仍答错是否因为表头不在名单里」 | 实测（24 份 T1 资料 × 2 档 ρ，诊断变体，无区间统计） |
| `g2lite_replay_header_summary_mac_m4max.json`、`g2lite_replay_header_manifest_mac_m4max.json` | 重放的配对表、集合计数、四格与逐请求回答；重放 run 的 manifest 与参照 run 的标识 | 实测 |
| `g2lite_replay_filler_mac_m4max.md` | 表头重放的续篇：把钉住的 33 个表头位置换成紧接表头的 33 个填充句位置（BF），与 B、BH 同 run 配对，分离「表头的内容」与「靠前的位置」 | 实测（同一 24 份 T1 资料 × 2 档 ρ，诊断变体，无区间统计） |
| `g2lite_replay_filler_summary_mac_m4max.json`、`g2lite_replay_filler_manifest_mac_m4max.json` | 续篇的配对表、集合计数、四格、逐请求回答与钉住块覆盖率；run 的 manifest 与参照 run 的标识 | 实测 |
| `g2lite_replay_header_split_mac_m4max.md` | 表头重放的续二：只钉表头格式句的 token 位置（BG，19 个）对比只钉前半句的 token 位置（BA，14 个），其余名额给最近的候选，与 B、BH 同 run 配对，分离表头文本里起作用的是哪一半 | 实测（同一 24 份 T1 资料 × 2 档 ρ，诊断变体，无区间统计） |
| `g2lite_replay_header_split_summary_mac_m4max.json`、`g2lite_replay_header_split_manifest_mac_m4max.json` | 续二的配对表（四个变体各自的窗口与内/外）、集合计数、全部变体两两四格、逐请求回答与钉住块覆盖率；run 的 manifest 与参照 run 的标识 | 实测 |
| `g2lite_replay_format_split_mac_m4max.md` | 表头重放的续三：只钉格式句前段「格式为「编号 X：Y」，」的 token 位置（BG1，9 个）对比只钉后段「Y 是编号 X 对应的四位数值。」的 token 位置（BG2，10 个），与 B、BH、BG 同 run 配对，分离格式句里起作用的是哪一段 | 实测（同一 24 份 T1 资料 × 2 档 ρ，诊断变体，无区间统计） |
| `g2lite_replay_format_split_summary_mac_m4max.json`、`g2lite_replay_format_split_manifest_mac_m4max.json` | 续三的配对表、集合计数、全部变体两两四格、逐请求回答与钉住块覆盖率；run 的 manifest 与参照 run 的标识 | 实测 |
| `cards/G0_mac_m4max.md`、`cards/G1a_mac_m4max.md`、`cards/G1b_mac_m4max.md`、`cards/G2lite_mac_m4max.md`、`cards/G2lite_replay_header_mac_m4max.md`、`cards/G2lite_replay_filler_mac_m4max.md`、`cards/G2lite_replay_header_split_mac_m4max.md`、`cards/G2lite_replay_format_split_mac_m4max.md` | 关卡卡片：运行前预测、运行后观察、解释、决策 | 预测写于运行前 |
| `g0_*.json`、`g1a_*.json`、`g1a_supplement_*.json` | 上述报告的精简数据与 manifest | 实测 |
| `g0_roofline_mac_bw390.{md,json}` | 以实测 390 GB/s 回填的载荷情景 | 预测 |
| `provenance/` | 跑分时的基准源码快照、跑分后 diff、原始 run 归档与哈希清单 | 证据链，`scripts/check_provenance.py` 校验 |
| `bootstrap_validation.md`、`public_publication.md` | 启动工具与发布记录 | 不是 GPU 结果 |

重新生成：`python scripts/g0_report.py`、`python scripts/g1a_report.py`、`python scripts/g1a_supplement_report.py`、`python scripts/g1b_report.py`、`python scripts/g2lite_report.py`、`python scripts/g2lite_replay_report.py`（表头重放；续篇、续二、续三用 `--run … --out … --json … --manifest … --title … --card …` 指定自己的 run 与输出；都需要 `runs/` 里有参照 run，见 `provenance/README.md`）。无 GPU 从归档重生 G1a 报告的步骤见 `provenance/README.md`。
