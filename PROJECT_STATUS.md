# 接力状态

更新日期：2026-10-17。状态：**G0_DONE / G1A_DONE（通过，continue）/ PR1_MERGED（`0798aa7`）/ NOTEBOOK_01_DONE / G1B_DONE（工程完成；性能 `correct_but_not_faster`，收口速度主线）/ NOTEBOOK_02_DONE / P0_FUSED_SCORE_DONE（停止门通过）/ P1_FUSED_ATTEND_DONE（停止门未通过：比值 0.82 / 0.83 > 0.80）/ FUSED_KERNEL_LINE_STOPPED（用户决定，2026-10-17）/ PR2_MERGED（`e682e9b`）**。`task/p0-fused-score-kernel`（PR #3）是已停止的独立实验原型，目标分支改为 main，待归档评审；可合并的记录不等于内核完成生产验收。

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

## 本轮完成：G1b（Mac M4 Max，2026-10-16 20:21 到 20:46 UTC，分支 `task/g1b-qwen-mac`）

- 模型：只下载 Qwen/Qwen3-0.6B 固定 revision `c1899de289a04d12100db370d81485cdf75e47ca`（权重 sha256 `f47f7117…`，596,049,920 参数；28 层、16/8 头、head_dim 128 从 config 读取并与参数形状核对）。主体冻结，索引器未训练（随机，P0，r=32）。
- 实现（`src/attention_lab/qwen_wrap.py`）：`LabAttention` 封装每层 `self_attn`（复用原 q/k/v/o、q_norm/k_norm、模型 RoPE 与输出路径，不改安装库）；`LabCache` 静态 K/V `[B,G,C,d]` + 索引表 `zT [B,G,r,C]`，每层一次 append、`advance`、`rollback`、`clone`；`StepPlan` 每步在设备上建一次（k0 请求内固定、合法长度每步加一）；稀疏步调用 G1a 的 `sparse_decode_step_kernel`；`LabRunner` / `NativeRunner` 驱动；`tiny_qwen3` 供无下载契约。
- 正确性：`tests/test_qwen_wrapper.py` 12 项（tiny 随机 Qwen3 CPU float64 + MPS fp32/bf16；真实 checkpoint CPU fp32 三段对照）。正式数值 `bench/model_numerics.py`（run 3 `g1b-numerics-20261016T204139Z-formal3`，98 项门控）：float32 全部按预设通过（max|Δ| ≤ 6.4e-4，容差 1e-2 / 1e-3）；bf16 一次前向与未改模型逐位相同；**封装逐 token 路径与 transformers 自身 DynamicCache 逐 token 路径在 MPS 上逐位相同（fp32 与 bf16、三个长度、词表分数与缓存）**；bf16 对一次前向的 17 项绝对预设未过（相对 2.05e-2 到 2.41e-2 对 2e-2；K 1 到 2 个 ulp 对 0.25），归因为库自身逐 token 舍入，修订判据（≤ 2 × 自身噪声底）在 run 2 前登记并全过；run 1 记录保留。
- 测速（`bench/model_decode_bench.py`，run `g1b-mac-20261016T202847Z-formal`，commit `c95e097`，375 s）：bf16、batch 1、N0 8192 / 16384 / 32256、ρ=1/32、容量 N0+128、四条路径（原生 DynamicCache / 原生 StaticCache / 封装 A / D）各自缓存。整步中位 ms（eager）：封装 A 13.32 / 16.00 / 21.21，D 16.11 / 16.84 / 19.68，原生静态 14.19 / 16.74 / 22.65，原生动态 24.15 / 37.49 / 74.68（序列口径 16.7 / 23.9 / 39.3）。**D/A 配对中位 1.210 / 1.054 / 0.930（32K 范围 0.872–0.986，7/50 循环 ≤ 0.90）**；块编译（MLP+RMSNorm，四路径共用）后 1.225 / 1.103 / 0.934。128 步序列平坦（斜率 −31 到 +2 µs/步），k 恒等于 k0，分配器快照恒定（DynamicCache 的载荷增长 14.68 MB 不反映在快照中），无重编译。剖析（只记录 CPU 活动）：D 无全长 matmul，非视图 aten 算子比 A 多 422，算子自时间之和 4 到 8 ms 对墙钟 13 到 21 ms，基底归因未分离。
- 判定：**工程完成；性能 `correct_but_not_faster`**（无设置每循环 ≤ 0.90；32K 临界 7% 收益）。预登记的修正（块编译）不改格局；剖析定位到 D 链核数量，同形状探针显示少两个核只到 0.915（`runs/g1b-dev-probe-kernelcount-20261016T2045Z.txt`，非正式），未跑正式一轮。按规则**收口速度主线**：不以速度为前提进 G2，不训练，不换更大模型。详见 `reports/cards/G1b_mac_m4max.md`（预测写于运行前，含预测对照）与 `reports/g1b_mac_m4max.md`。
- 教学：`notebooks/02_qwen_integration_and_decode.ipynb` 从全新内核执行并导出 `reports/02_qwen_integration_and_decode.html`："把索引注意力装进真实 Qwen：为什么单层快不等于整模型快"。
- 消耗：设备活跃时间约 10 分钟，现金 0，下载 1.5 GB。累计 G0 + G1a + G1b 约 55 分钟。
- **PR #2 评审收尾补丁（2026-10-17）**：显式阶段状态（`LabState.phase`，`forward()` 必须给出阶段；tiny 模型测试含单 token 资料、chunk = 1、8193 按 2048 分块）；数值比较两侧 `torch.isfinite` 先于聚合（反例测试注入 NaN）；manifest 记录 `HF_HUB_OFFLINE`；数值重跑 run 4 `g1b-numerics-20261017T085459Z-formal4`（98 项与 run 3 逐项相同）；32K 最小对照 run `g1b-mac-20261017T085522Z-phasefix`（A 21.76 / D 19.25 ms，D/A 0.884，判定不变，报告附录）；报告输入与跑分时源码快照归档在 `reports/provenance/`（`check_provenance.py` 覆盖，干净目录可重生报告）；卡片、状态、范围说明的措辞与算术更正（fp32 缓存余量 1.56 倍、剖析归因未分离、DynamicCache 载荷增长与分配器快照、判据修订时间、`correct_but_not_faster` 含义、bf16 三层表述）。判定不变。

## 本轮完成：P0 预检（用户授权半天，2026-10-17 07:35 到 07:40 UTC，分支 `task/p0-fused-score-kernel`）

- 问题：D 链的打分段（z 列写入 + u 投影 + 候选打分，PyTorch 路径是 3 个 bmm + 1 个 copy_）做成一个 Metal 核能否接近带宽下界。任务书 `docs/tasks/04_FUSED_D_KERNEL.md`，停止门写在开工前。
- 实现：`src/attention_lab/metal_score.py`（`torch.mps.compile_shader`；v1 每线程一列 2 字节读取，v2 每线程 4 列 8 字节向量读取 + 共享内存归约，两版保留）、`tests/test_metal_score.py`（4 项）、`bench/p0_score_kernel.py`、`scripts/p0_report.py`。
- 结果（两个独立进程，28 层轮换，bf16）：32K 处 fused v2 一次启动 **69.7 / 70.0 µs/层**（有效 236 GB/s；PyTorch 四算子链 102，只打分的 bmm 62.6），对链比值 0.69 / 0.68；8K 47.9（0.48–0.50），16K 56.4（0.58–0.63）。第一版核 209 µs（79 GB/s）不过门，重设计一次后通过。数值：fused fp32 分数对 float32 参照 ≤ 3.6e-7（PyTorch bf16 路径 6e-3），z 列逐位相同，top-k 集合一致 99.6% 到 99.7%。
- 判定：**P0 停止门通过**（三条、两进程）。单靠这一段整模型 32K D/A 只能从 0.930 到约 0.89（估算）；要稳定越线需要 P1（取行注意力核）与 P2（选择核）。**P1 未授权，未开始。** 不改模型路径、不改 G1a 核。证据：`reports/p0_score_kernel_mac_m4max.md`、`reports/cards/P0_fused_score_mac_m4max.md`、`runs/p0-formal.log`。
- 消耗：设备活跃时间 < 1 分钟，现金 0。

## 本轮完成：P1（用户授权，2026-10-17 07:55 到 07:58 UTC，分支 `task/p0-fused-score-kernel`）

- 问题：D 链的取行注意力段（两次拉平 `index_select` + 子集 SDPA）做成按行表直接读缓存的 Metal 核，S = 1076 时能否 ≤ 60 µs/层且比值 ≤ 0.80。
- 实现：`src/attention_lab/metal_attend.py`（每 simdgroup 一行、32 lane × 4 维、两个查询头共用读取、在线 softmax、分区合并核；交错布局与排序行表作诊断）、`tests/test_metal_attend.py`（6 项，对 `dense_mask_reference`）、`bench/p1_attend_kernel.py`、`scripts/p1_report.py`。
- 结果（两个独立进程）：S = 1076 处最优 P8 **38.5 / 39.8 µs**（113 GB/s 有效），PyTorch 路径 47.0 / 48.1 µs（93 GB/s），比值 **0.82 / 0.83**；S = 324 处 27.9 µs（0.85–0.86）；S = 580 处 36.0 µs（0.96–0.99）。数值：对 float32 参照 ≤ 2.4e-7，bf16 输出 100% 逐位等于 RNE。交错布局、排序行表均在 ±2% 内无影响。
- 判定：**时间与数值通过，比值门未通过 → 按预登记规则停止**。取行段受随机访存效率限制而非核数量限制，PyTorch 的 index_select 已接近同一下界；用局部节省推出的三段全融合整模型 32K D/A 约 0.84 到 0.88 是情景推算，不是实测也不是严格上限（G1a 到 G1b 已显示局部节省不会原样进入整模型；范围说明的"乐观 0.65"前提不成立）。P0 在 fp32 中算分数，旧路径含 bf16 中间结果，top-k 集合一致率 99.6% 到 99.7%，不是逐位等价替换。G1b 的收口结论维持；不建议 P2。
- 消耗：设备活跃时间不到 1 分钟，现金 0。累计 G0 + G1a + G1b + P0 + P1 约 57 分钟。

## 尚未完成

速度主线按 G1b 收口，融合核路线在 P1 后由用户决定停止（原型与记录在 PR #3，不接入模型路径）；以下均未启动、需用户决定与授权：CUDA 复核轨道（4080 主机）；> 32,768 的上下文；G2 到 G5 的任何质量结论（`docs/tasks/03_G2_TO_G5.md` 仍为 BLOCKED）。本机磁盘在 G1b 运行时只剩约 28 GiB 可用（97%），后续大 run 前需留意。

## 下一步

1. PR #2 已合并（`e682e9b`）。PR #3（`task/p0-fused-score-kernel`，已停止的独立实验原型）目标分支已改为 main，待用户评审归档；不自行合并。
2. 融合核路线已由用户决定停止（2026-10-17）：P0 通过、P1 比值门未过，整模型比值的情景推算约 0.84 到 0.88（不是实测）；P2 到 P4 不做。剩余方向由用户另行决定：(a) 以教学与边界结论收口；(b) CUDA 复核轨道；(c) 更长上下文。任何一项都需要明确授权，不自动开始。
3. 不重跑 G0/G1a，不训练，不租云。

```bash
source .venv/bin/activate
python -m pytest -q tests                                  # 全部契约（含 G1b 封装；真实 checkpoint 项只在本地快照存在时跑）
python bench/model_numerics.py --tag formalN               # G1b 三段数值对照，约 25 秒（需本地快照）
python bench/model_decode_bench.py --compile-blocks --tag formal   # G1b 整模型测速，约 6.5 分钟
python scripts/g1b_report.py                               # 由非 quick run 重新生成 G1b 报告
python bench/microbench.py --modes eager --tag pN          # G1a 单进程微基准，约 5 分钟
python scripts/g1a_report.py
```

## 接力格式

每次记录分支、代码 SHA、run_id、实际命令、证据路径、失败/未测项、剩余预算与下一条命令。未完成进程的 PID 与主机路径保存在本地忽略目录，不直接写进公开报告；公开接力只提供去标识化状态。
