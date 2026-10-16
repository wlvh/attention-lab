# 接力状态

更新日期：2026-10-16。状态：**G0_DONE_ON_MAC / G1A_NOT_RUN**。

## 远端与发布

用户创建的公开仓库：`wlvh/attention-lab`，默认分支 `main`。首次入口提交为 `95885aa2c24d692c595c3ea88024a6c836ffdea6`。本轮工作在本地分支 `task/g0-g1a-mac`，尚未提交，等待用户决定 commit 与 PR。

## 已准备

当前有效方案、背景与决策记录、原始 v3/roofline/教材、Mac 操作手册、Codex 接力指令、Notebook 交付规范、预算、测量协议、分关任务书，以及预测核算、来源和链接校验工具。

## 本轮完成：G0（Mac M4 Max，2026-10-16）

- 环境：`~/Projects/attention-lab/.venv`，Python 3.12.10，torch 2.14.1，transformers 5.20.0；MPS 可用；依赖锁 `requirements/lock-macos-arm64.txt`。
- 能力：fp32/fp16/bf16 matmul、top-k、index_select、mask softmax、SDPA（含 `enable_gqa=True` 与 bool mask）、小索引矩阵 KL 反向、`torch.compile` 逐元素函数均在 MPS 上通过并与 CPU 对照一致；fp64 不支持。
- 带宽：只读约 440 到 456 GB/s，copy 约 400 到 413 GB/s，注意力形状扫描 382 到 419 GB/s（标称 546）。热读与轮换工作集无差别。
- 关键发现：索引路径成本由原语决定。`torch.gather` 配展开下标 673 µs/层（32K），拉平 `index_select` 27 µs/层；索引键预转置后打分从 137 降到 58 µs/层。最优链式每层 97 / 115 / 166 µs（8K / 16K / 32K）。
- 回填预测：以 390 GB/s 代入 roofline，叠加实测链式成本，eager 下 8K 持平偏负、16K 持平偏正、32K 正收益；详见 `reports/g0_mac_m4max.md` 第 6 节。预测写于运行前，见 `reports/cards/G0_mac_m4max.md`。
- 证据：`reports/g0_mac_m4max.md`、`reports/g0_probe_mac_m4max.json`、`reports/g0_inventory_mac_m4max.json`、`reports/g0_roofline_mac_bw390.{md,json}`、`scripts/g0_probe.py`、`scripts/g0_report.py`；原始日志在本地 `runs/`。
- 消耗：约 13 分钟设备活跃时间，现金 0。无模型下载，无训练，无云。

## 尚未完成

G1a 单层 A 与 D 微基准（含密集整层实际时间、子集注意力时间、`enable_gqa` 内核剖析）；G1b 起全部；CUDA 轨道未启动；G1 到 G5 的任何速度或质量结论。

## 下一步

在 `task/g0-g1a-mac` 分支按 `docs/tasks/01_G0_G1A.md` 的 G1a 部分实现 `src/attention_lab/`，遵守 G0 卡片给出的三条实现约束（索引键 `[G, r, N]` 连续；拉平 `index_select` 取 K/V；top-k 只对候选域一次）。先过正确性契约，再测 8K / 16K / 32K。

```bash
source .venv/bin/activate
python scripts/g0_probe.py            # 复跑 G0，约 10 秒
python scripts/g0_report.py           # 由最新 run 重新生成报告
```

## 接力格式

每次记录分支、代码 SHA、run_id、实际命令、证据路径、失败/未测项、剩余预算与下一条命令。未完成进程的 PID 与主机路径保存在本地忽略目录，不直接写进公开报告；公开接力只提供去标识化状态。
