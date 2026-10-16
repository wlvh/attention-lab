# 阅读与证据

所有数字来自本机 Mac M4 Max 的 MPS 实测，或由实测回填的预测；每个文件都写明了自己是哪一种。不要凭空填性能值。

| 文件 | 内容 | 性质 |
|---|---|---|
| `00_orientation.html` | 仓库导读与载荷账本 | 不是实测 |
| `01_environment_and_attention.html` | 第一份实验课：G0 与 G1a 的三个问题，已执行的 Notebook 导出 | 读取下列实测结果，CPU 小例子 |
| `g0_mac_m4max.md` | G0 报告：环境、能力、带宽、索引路径原语成本、回填预测（含公式更正） | 实测 + 预测 |
| `g1a_mac_m4max.md` | G1a 报告：单层 A 与 D 六进程微基准、通过判定、剖析 | 实测 |
| `g1a_supplement_mac_m4max.md` | G1a 补测：交错顺序、容量四格、折叠手写 A 对照 | 实测 |
| `cards/G0_mac_m4max.md`、`cards/G1a_mac_m4max.md` | 关卡卡片：运行前预测、运行后观察、解释、决策 | 预测写于运行前 |
| `g0_*.json`、`g1a_*.json`、`g1a_supplement_*.json` | 上述报告的精简数据与 manifest | 实测 |
| `g0_roofline_mac_bw390.{md,json}` | 以实测 390 GB/s 回填的载荷情景 | 预测 |
| `provenance/` | 跑分时的基准源码快照、跑分后 diff、原始 run 归档与哈希清单 | 证据链，`scripts/check_provenance.py` 校验 |
| `bootstrap_validation.md`、`public_publication.md` | 启动工具与发布记录 | 不是 GPU 结果 |

重新生成：`python scripts/g0_report.py`、`python scripts/g1a_report.py`、`python scripts/g1a_supplement_report.py`。无 GPU 从归档重生 G1a 报告的步骤见 `provenance/README.md`。
