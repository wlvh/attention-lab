# 接力状态

更新日期：2026-10-17。状态：**G0_DONE / G1A_DONE（通过，continue）/ PR1_MERGED（`0798aa7`）/ NOTEBOOK_01_DONE / G1B_UNLOCKED（进行中，分支 `task/g1b-qwen-mac`）**。

## 远端与发布

用户创建的公开仓库：`wlvh/attention-lab`，默认分支 `main`。首次入口提交为 `95885aa2c24d692c595c3ea88024a6c836ffdea6`。G0 在分支 `task/g0-g1a-mac` 提交为 `ae1c0a8`；G1a 的源码、测试、基准、报告、卡片、补测与 Notebook 01 随同一分支经 PR #1（两轮 GPT 6 Pro 评审）于 2026-10-17 合并进 `main`，合并提交 `0798aa7`。G1b 在分支 `task/g1b-qwen-mac` 上进行。

## 已准备

当前有效方案、背景与决策记录、原始 v3/roofline/教材、Mac 操作手册、Codex 接力指令、Notebook 交付规范、预算、测量协议、分关任务书，以及预测核算、来源和链接校验工具。

## G0（Mac M4 Max，2026-10-16，已提交 `ae1c0a8`）

MPS 能力、带宽、提交开销与索引路径原语成本；关键发现是 `torch.gather` 配展开下标比拉平 `index_select` 慢 25 倍、索引键预转置后打分快一倍。详见 `reports/g0_mac_m4max.md` 与 `reports/cards/G0_mac_m4max.md`。

## 本轮完成：G1a（Mac M4 Max，2026-10-16 14:51 到 15:22 UTC）

- 实现（`src/attention_lab/`）：`selection.py`（F、候选域、k0 规则）、`reference.py`（dense_reference、dense_mask_reference）、`sparse.py`（拉平 index_select 取行、子集注意力、gather_attention）、`full.py`（A 的 fold / enable_gqa / repeat / explicit 变体）、`indexer.py`（分组索引器，索引键 `[B,G,r,C]` 连续）、`decode.py`（一步解码的 A 与 D，D 只对候选域打分并做一次 top-k）、`runinfo.py`（来源记录）。
- 正确性：`tests/test_attention_contracts.py` 10 项契约，CPU float64 与 MPS float32 / bfloat16 全部通过，容差预设未改：全选对应完整注意力、同子集 gather 对应 dense mask、未选行填 NaN 不影响结果、未写入槽与未来位置不影响结果、单查询长历史不套方阵因果遮罩、原位置不重编号、top-k 与蛮力一致且只扫候选域、top-k 只调用一次且不用 `torch.gather`、索引表布局与分块构建、k 规则与边界。全仓 30 项测试通过。
- 微基准（`bench/microbench.py`）：G=8、H=32、d=128、r=32，N 8K/16K/32K，ρ 1/32 与 1/8，bf16 与 fp32，36 层轮换与热工作集，预热 20、正式 50，前后同步；5 个独立 eager 进程 + 1 个 eager+compile 进程。
- 结果（bf16、eager、轮换，每层 µs，6 进程中位数）：8K A 110 / D_full 195（1.78）；16K A 203 / D_full 222（1.10）；**32K A 405 / D_full 297（0.74，6/6 进程 ≤ 0.90）**，D_read 233（0.58）。32K ρ=1/8：D_full 411（1.02，不通过）、D_read 337（0.84，通过）。fp32 下 16K ρ=1/32 与 32K 两档通过，但 fp32 的 A 自身低效（32K 1426 µs），不作 bf16 依据。剖析：6 个 (dtype, N) 的 D 都没有 N×128 的 matmul；A 的 SDPA 调度为 `_scaled_dot_product_attention_math_for_mps`。compile：36 格全部 1 graph、0 break、逐位相同、compile/eager 0.81 到 1.12，不改格局。
- 判定：**通过，continue**。预测对照：方向全部成立；32K ρ=1/8 的 D_full 预测 0.81、实测 1.02（预测错误）；链式时间高于分段之和（与"重叠"预测相反）；`enable_gqa` 在 32K 慢 65%；容量 N+2048 切片视图的代价在进程间 0% 到 40% 不一致，G1b 须复测。
- 证据：`reports/g1a_mac_m4max.md`（由 `scripts/g1a_report.py` 生成）、`reports/g1a_summary_mac_m4max.json`、`reports/g1a_manifest_mac_m4max.json`、`reports/cards/G1a_mac_m4max.md`（预测写于运行前）；原始 `runs/g1a-mac-20261016T145200Z-p1` … `151108Z-p5`、`151610Z-compile`（本地，含 microbench.jsonl、summary.json、profiles.txt、manifest.json），日志 `runs/g1a-formal.log`。
- 消耗：约 30 分钟设备活跃时间，现金 0。无模型下载，无训练，无云。

## PR #1 评审第一轮（GPT 6 Pro，2026-10-17）的处理

- **证据链**：跑分时的 `bench/microbench.py` 快照（SHA-256 `4c1e5b66…`，与六份 manifest 一致）与一行 diff 归档在 `reports/provenance/`，原始 run 打包 3.1 MB；`scripts/check_provenance.py` 进 CI。报告可从归档无 GPU 逐字节重生。
- **测试**：`assert_close` 改为按参照精度比较；新增三条"必须失败"的回归测试；CI 新增安装 CPU torch 的任务，契约测试被跳过即失败。本机 33 项测试、72 子测试通过。
- **G0 公式**：净值改为"A 理想读取 − 实测最优链式"，报告与卡片附更正；方向不变，D 更有利。
- **措辞**：G1a 的"MPS 无重叠"因果改为"已观察到、原因未分离"；p95 口径、memory_peaks 快照、固定顺序写入边界。
- **补测**（`reports/g1a_supplement_mac_m4max.md`，run `g1a-supp-20261016T170030Z-formal`，commit `4dd95df`）：交错顺序下 bf16 32K ρ=1/32 的配对 D_full/A 为 0.67（0.60–0.72），40 cycle 全部 ≤ 0.90，结果不依赖固定执行顺序；32K ρ=1/8 为临界档（配对中位 0.89–0.90，进程中位 1.02）；容量 N 与 N+2048 同数据交错，A 与 D 的松弛/紧凑均在 0.99–1.01，本次条件下未复现容量惩罚；折叠手写 A 比 sdpa_fold 慢 1.1–2.6 倍，未找到更快的合法 A。附带：fp32 的 A 在 32K 跨运行漂移 1426→918 µs，bf16 的 A 跨运行偏离最大约 11%。
- **教学**：`notebooks/01_environment_and_attention.ipynb` 已从全新内核执行并导出 `reports/01_environment_and_attention.html`，围绕三个问题：取子集两种写法为何差 25 倍、少读 KV 为何仍可能更慢、分段相加为何不等于链式。

## PR #1 评审第二轮（2026-10-17）的处理

复审确认第一轮的卡点关闭，指出三处收尾：CI 测试步骤无 `pipefail`，pytest 失败会被 `tee` 掩盖，已加 `set -euo pipefail` 并用故意失败的测试在本地验证退出码；共用的 `runinfo.refuse_unsafe_env` 仍按字符串比较 `"0"`，已改为数值解析并由 `g0_probe.py` 复用，新增 `tests/test_runinfo_env.py`；Notebook 与 G0 卡片把 8K 处 A 实测 110 µs 与 86 µs 情景值的快慢写反，已更正为"A 比情景慢，预测偏差来自 G0 链式成本范围不完整"。补测结论按复审收窄：顺序效应为"结果不依赖固定顺序"，容量为"本次条件下未复现"，32K ρ=1/8 标为临界档，bf16 的 A 跨运行偏离写实际数字。未重跑任何基准，未改旧 manifest。

## 尚未完成

G1b 起全部（模型封装、静态缓存、连续 128 步、重编译与内存增长；任务书 `docs/tasks/02_G1B_MODEL.md` 已解锁，`03_G2_TO_G5.md` 仍为 BLOCKED）；CUDA 轨道未启动；G2 到 G5 的任何质量结论。Notebook 01 与已执行 HTML 已随 PR #1 交付。

## 下一步

1. G1b（分支 `task/g1b-qwen-mac`，按 `docs/tasks/02_G1B_MODEL.md`）：只下载 Qwen/Qwen3-0.6B 固定 revision；封装每层 `self_attn`，先做未改模型 / 封装完整路径 / 封装加静态缓存的三段数值对照；再接入 G1a 的索引读取；整模型 bf16 测速 N0 ∈ {8192, 16384, 32256}、ρ=1/32、固定 N 单步与连续 128 步；预测先写进 `reports/cards/G1b_mac_m4max.md`；Notebook 02。
2. 不重跑 G0/G1a，不进入 G2/G3，不训练，不租云。

```bash
source .venv/bin/activate
python -m unittest tests.test_attention_contracts -v     # 契约测试，约 1 秒
python bench/microbench.py --modes eager --tag pN        # 一个独立 eager 进程，约 5 分钟
python scripts/g1a_report.py                              # 由全部非 quick run 重新生成报告
```

## 接力格式

每次记录分支、代码 SHA、run_id、实际命令、证据路径、失败/未测项、剩余预算与下一条命令。未完成进程的 PID 与主机路径保存在本地忽略目录，不直接写进公开报告；公开接力只提供去标识化状态。
