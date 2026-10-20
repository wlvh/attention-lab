# Attention Lab · 注意力读取取舍实验室

**在同一份预训练权重、同一任务、同一设备下，比较完整注意力与“先筛选历史、再做主注意力”的速度、答案质量、内存和新增训练成本。**

MacBook 本地优先；Codex 实施；Python 脚本执行；Jupyter 分段解读；HTML 直接阅读。

> 已完成本机实现、单层与整模型对照、索引训练和新题质量/成本评测。正式研究交付已进入主干，现有 00 导读与 01—07 实验课；当前工作是教学说明修订。旧方案和任务书保留历史，不是重新运行实验的指令。

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
src/attention_lab/           注意力、缓存、索引训练与真实推理的正式实现
scripts/                     结果重建、来源核对、教学辅助与历史工具
configs/                     各阶段已登记的实验配置
tests/                       工具、CPU 契约及有明确设备条件的检查
notebooks/                   00 导读及已执行的 01—07 实验课
reports/                     课程 HTML、阶段报告、精简结果与来源归档
runs/                        本地原始实验输出，不自动提交大数组
```

## 阅读与重放

直接阅读 [课程目录](notebooks/README.md) 或 [第七课结果全貌](reports/07_trained_sparse_quality_and_cost.html)；想从一次参数更新进入代码，可读 [第四课](reports/04_can_an_indexer_learn.html)。默认使用保存结果和小规模 CPU 检查，第四课另有 12 参数、20 步 CPU 更新，不加载真实 0.6B 模型。

在现有环境中，以下命令只核对文件、重建旧报告或计算字节量：

```bash
.venv/bin/python scripts/check_sources.py
.venv/bin/python scripts/g4lite_report.py
.venv/bin/python scripts/roofline.py --rho 0.03125 --r 32
```

预测脚本默认不填带宽。历史 4080 假设需显式传 `--bw 716.8e9`，它不是本机实测。正式训练和 GPU 基准有独立脚本，不属于默认阅读步骤。

现有公开仓库为 `wlvh/attention-lab`。在本机继续使用已有目录、环境、缓存和 runs；在另一台机器需要源码时才克隆。不要运行旧建仓脚本，不自动重新执行 G0 或训练。新会话先读 [PROJECT_STATUS.md](PROJECT_STATUS.md)，再按当前用户任务选择相关文件。

仓库公开不等于重新许可原始材料，见 [RIGHTS.md](RIGHTS.md)。不自动发布 GitHub Pages。

## 研究成功的含义

本设备不划算、质量损失太大、证据不足，都是可接受结果。成功意味着能够根据可追溯的预测与对照作出设计判断，而不是一定证明稀疏注意力更快。

原始 [PLAN_v3.md](docs/archive/PLAN_v3.md) 保留不改；执行时以当前方案和 [AGENTS.md](AGENTS.md) 为准。原件、讨论结论、官方资料核对和本地工具验证各有独立来源，参见 [SOURCES.md](docs/SOURCES.md)。


已完成的连续质量路线：单层学习（第四课）→各层训练（第五课）→真实状态对照（第六课）→新题质量与成本（第七课）。32 份新 T1 中 A 为 21/32、D0 为 0/32、训练后 D 为 17/32；当前约 2K 设置下 D 更慢，缓存数组也更多，不能称为实用提速方案。

PR #16、#17、#18 已按顺序合入 main，最终合并提交为 `1edde2d`，对应研究 Issue 已关闭。当前教学修订沿用 [PR #19](https://github.com/wlvh/attention-lab/pull/19)，不自动合并。研究范围与历史停止决定见 [接力状态](PROJECT_STATUS.md)。
