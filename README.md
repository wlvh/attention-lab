# Attention Lab · 注意力读取取舍实验室

**在同一份预训练权重、同一任务、同一设备下，比较完整注意力与“先筛选历史、再做主注意力”的速度、答案质量、内存和新增训练成本。**

MacBook 本地优先；Codex 实施；Python 脚本执行；Jupyter 分段解读；HTML 直接阅读。

> **当前已交付三份实验课。** G0/G1a/G1b 已完成；Mac 速度主线按停止规则收口。G2-lite 与审查修订后的第三课已随 [PR #5](https://github.com/wlvh/attention-lab/pull/5) 合并到 main（`d2fc4ad`）。原始模型结果仍是 v2；当前配置 v3 为 NOT_RUN，G3 未授权。#6–#9 只保留为同批案例的递进诊断。

## 第一次打开

先读 [START_HERE.md](START_HERE.md)，或直接打开下面的已保存 HTML；不需要运行 Notebook、下载模型或启动 GPU。三课的版本与源码边界见[课程入口](notebooks/README.md)，当前状态与诊断索引见[交接当前段](docs/HANDOFF_2026-10-18.md#current-state-20261018)。

1. [第一课：环境与单层读取成本](reports/01_environment_and_attention.html)——G0/G1a，PR #4 措辞校正版。
2. [第二课：Qwen接入与整步解码](reports/02_qwen_integration_and_decode.html)——G1b，PR #4 措辞校正版。
3. [第三课：少看历史会丢掉什么](reports/03_what_is_lost_when_reading_less.html)——旧v2结果、PR #5修订解释与展示；整页布局仍标未验。

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
src/attention_lab/           正式实现：选择器、注意力、Qwen封装、资料与汇总
scripts/                     报告生成、来源校验与历史启动工具
configs/                     实验配置；v3控制修复未运行，旧v2在来源快照
tests/                       启动、数值契约与报告回归；依赖和跳过项分别记录
notebooks/                   00导读，以及已保存输出的01/02/03实验课
reports/                     三课HTML、报告、逐题结果与来源归档
runs/                        本地原始实验输出，不自动提交大数组
```

## 阅读时可选的文件检查

以下只用 Python 标准库，不安装模型，不收费，不启动 GPU 基准：

```bash
python3 scripts/check_sources.py
python3 scripts/check_provenance.py
python3 scripts/check_repository.py
```

预测脚本默认**不填设备带宽**，只输出字节账本。需要复现 v3 的 4080 假设时明确传入 `--bw 716.8e9`；这不是 Mac 的带宽，更不是实测生成速度。

## 克隆公开仓库并开始

仓库由所有者明确选择公开：`wlvh/attention-lab`。不需要再创建仓库，也不运行旧私有建仓脚本。

```bash
git clone https://github.com/wlvh/attention-lab.git
cd attention-lab
```

新读者克隆后，在 Mac 本地 Codex 中打开此目录；已有本地项目直接复用，发送：

```text
这是既有项目接管。读取 AGENTS.md、PROJECT_STATUS.md、
docs/HANDOFF_2026-10-18.md 的当前段，以及 notebooks/README.md。
保留环境、缓存、runs和worktree，不重跑G0/G1a/G1b或G2-lite。
以已保存第三课的真实案例开始逐段教学；新实验和诊断归档另行授权。
```

公开仓库不包含令牌、机器身份和非项目聊天记录；运行日志发布前需检查。模型权重、大数组、凭证留在忽略目录。仓库公开不代表自动授予新的开源许可；见 [RIGHTS.md](RIGHTS.md)。不自动开启 Pages。

## 研究成功的含义

本设备不划算、质量损失太大、证据不足，都是可接受结果。成功意味着能够根据可追溯的预测与对照作出设计判断，而不是一定证明稀疏注意力更快。

原始 [PLAN_v3.md](docs/archive/PLAN_v3.md) 保留不改；执行时以当前方案和 [AGENTS.md](AGENTS.md) 为准。原件、讨论结论、官方资料核对和本地工具验证各有独立来源，参见 [SOURCES.md](docs/SOURCES.md)。
