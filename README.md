# Attention Lab · 注意力读取取舍实验室

**在同一份预训练权重、同一任务、同一设备下，比较完整注意力与“先筛选历史、再做主注意力”的速度、答案质量、内存和新增训练成本。**

MacBook 本地优先；Codex 实施；Python 脚本执行；Jupyter 分段解读；HTML 直接阅读。

> 当前是**文档与启动工具仓库**，不是已经完成的注意力实现。没有本项目的 Mac / CUDA 实测、Qwen 评测或索引器训练结果。原始附件中的实验和性能预测另行存档，不算本项目成绩。公开远端由用户于2026-10-16建立，发布状态以远端提交与文件核验为准。

## 第一次打开

先读 [START_HERE.md](START_HERE.md)。直接交给本地 Codex 的任务书在 [CODEX_HANDOFF.md](docs/CODEX_HANDOFF.md)。

| 你要做什么 | 入口 |
|---|---|
| 理解为何做这个项目、为何不再从零训练 | [背景与问题演进](docs/BACKGROUND.md) |
| 看当前完整执行要求 | [有效方案](docs/PLAN_CURRENT.md) |
| 看 v3 哪些地方被修正、为何修正 | [决策与修订记录](docs/DECISIONS.md) |
| 在 Mac 设置环境、运行、停止、恢复 | [Mac 操作手册](docs/MAC_RUNBOOK.md) |
| 实施模型封装、索引器、缓存与对照 | [计算与数据接口](docs/IMPLEMENTATION.md) |
| 理解测速、任务、统计的验收 | [测量协议](docs/MEASUREMENT.md) |
| 把真实代码与结果转成实验课 | [Notebook 交付规范](docs/TEACHING_NOTEBOOKS.md) |
| 控制本地资源与 $100 云预算 | [预算与停止规则](docs/BUDGET.md) |
| 知道现在完成了什么 | [PROJECT_STATUS.md](PROJECT_STATUS.md) |
| 查原件与引用 | [来源记录](docs/SOURCES.md)、[教材锚点](docs/REFERENCE_MAP.md) |

## 内容布局

```text
AGENTS.md                    Codex 的项目约定
START_HERE.md                一页入口与发布方式
PROJECT_STATUS.md            不依赖聊天的接力状态
docs/                        当前方案、背景、方法与操作知识
docs/archive/                原始 v3、原 roofline、被替代方案（不可当当前要求）
docs/reference/              原始教材 HTML
docs/tasks/                  可直接实施的分关任务书
src/attention_lab/           正式实验实现预留目录；目前只有包标识
scripts/                     已有：预测核算、源文件校验、环境清单、GitHub 发布
configs/                     Mac 首轮预期配置（不是已有执行器）
tests/                       启动工具测试，不等于注意力正确性测试
notebooks/                   00 导读；后续实验课按关卡增加
reports/                     导读 HTML 与启动工具验证记录
runs/                        本地原始实验输出，不自动提交大数组
```

## 现在能运行的命令

以下只用 Python 标准库，不安装模型，不收费，不启动 GPU 基准：

```bash
python3 scripts/check_sources.py
python3 -m unittest discover -s tests -v
python3 scripts/roofline.py --rho 0.03125 --r 32
python3 scripts/environment_inventory.py --output runs/inventory.json
```

预测脚本默认**不填设备带宽**，只输出字节账本。需要复现 v3 的 4080 假设时明确传入 `--bw 716.8e9`；这不是 Mac 的带宽，更不是实测生成速度。

## 克隆公开仓库并开始

仓库由所有者明确选择公开：`wlvh/attention-lab`。不需要再创建仓库，也不运行旧私有建仓脚本。

```bash
git clone https://github.com/wlvh/attention-lab.git
cd attention-lab
```

在 Mac 本地 Codex 中打开此目录，发送：

```text
读取 AGENTS.md、PROJECT_STATUS.md、docs/CODEX_HANDOFF.md，
按 docs/tasks/01_G0_G1A.md 从 G0 环境检查与 G1a 最小微基准开始。
保留执行前预测和真实日志，正式实现写 .py，逐段教学交 .ipynb 和 HTML。
不要把导读当作 GPU 实验结果；首轮不下载全部模型、不训练、不租云。
```

公开仓库不包含令牌、机器身份和非项目聊天记录；运行日志发布前需检查。模型权重、大数组、凭证留在忽略目录。仓库公开不代表自动授予新的开源许可；见 [RIGHTS.md](RIGHTS.md)。不自动开启 Pages。

## 研究成功的含义

本设备不划算、质量损失太大、证据不足，都是可接受结果。成功意味着能够根据可追溯的预测与对照作出设计判断，而不是一定证明稀疏注意力更快。

原始 [PLAN_v3.md](docs/archive/PLAN_v3.md) 保留不改；执行时以当前方案和 [AGENTS.md](AGENTS.md) 为准。原件、讨论结论、官方资料核对和本地工具验证各有独立来源，参见 [SOURCES.md](docs/SOURCES.md)。

最新缩减任务：[G3a 单层索引器可学习性](docs/tasks/06_G3A_SINGLE_LAYER.md)；[结果报告](reports/g3a_mac_m4max.md)；[实验课04](reports/04_can_an_indexer_learn.html)。默认阅读使用保存结果，不启动训练。
