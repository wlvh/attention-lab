# 接力状态

更新日期：2026-10-17。状态：**G0_DONE / G1A_DONE（通过，continue）/ PR1_MERGED（`0798aa7`）/ NOTEBOOK_01_DONE / G1B_DONE（工程完成；性能 `correct_but_not_faster`，收口速度主线）/ NOTEBOOK_02_DONE / P0_FUSED_SCORE_DONE（停止门通过）/ P1_FUSED_ATTEND_DONE（停止门未通过：比值 0.82 / 0.83 > 0.80）/ FUSED_KERNEL_LINE_STOPPED（用户决定，2026-10-17）/ PR2_MERGED（`e682e9b`）/ PR3_MERGED（`158bad9`）/ PR4_MERGED（`5f55729`）/ G2LITE_DONE（T1 对照完成；T2 `baseline_too_weak`；分支 `task/g2-lite-mac`，PR 待评审）/ G2LITE_REPLAY_HEADER_DONE（B 窗口内失败由「表头不在名单里」解释；分支 `task/g2-lite-replay-header`，PR #6 待评审，不合并）/ G2LITE_REPLAY_FILLER_DONE（起作用的是表头的内容，不是靠前的位置；分支 `task/g2-lite-replay-filler-pin`，PR #7 待评审，不合并）/ G2LITE_REPLAY_HEADER_SPLIT_DONE（表头里起作用的是格式句，前半句单独无效；分支 `task/g2-lite-replay-header-split`，PR #8 待评审，不合并）/ G2LITE_REPLAY_FORMAT_SPLIT_DONE（格式句里起作用的是后段「Y 是编号 X 对应的四位数值」，前段单独无效；分支 `task/g2-lite-replay-format-split`，PR 待评审，不合并）**。`task/p0-fused-score-kernel`（PR #3）是已停止的独立实验原型，目标分支改为 main，待归档评审；可合并的记录不等于内核完成生产验收。

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

## 本轮完成：G2-lite（Mac M4 Max，2026-10-17 12:38 到 13:08 UTC，分支 `task/g2-lite-mac`，基于 main `158bad9`）

速度主线收口后的独立缩减任务（任务书 `docs/tasks/05_G2_LITE.md`；`03_G2_TO_G5.md` 仍为 BLOCKED）：用真实问题检验「少看历史」丢掉什么。

- 实现：`src/attention_lab/decode.py` 在 `select_rows_kernel`（D0）旁加 `select_rows_recent`（B：F ∪ 最近 k 个候选）与 `select_rows_uniform`（C：F ∪ 候选域内均匀抽取 k 个，numpy SeedSequence 按（种子, 请求, 步）播种，回退重放不变），共用 `gathered_decode_step_kernel`（拉平 index_select + 子集注意力，D0 的核序列不变）；`qwen_wrap.py` 的 `LabState.selector ∈ {index, recent, uniform}`、每步名单记录；`g2lite_data.py`（资料生成，前缀切点逐 token 核对闭合）、`g2lite_eval.py`（冻结的解析与判分、聚合）；`bench/g2lite_quality.py`（三遍：A 基线 → 可用格子四方法 → 改写）；`scripts/g2lite_report.py`。契约 `tests/test_g2lite.py` 11 项（同 F 同名额同读取数、B 为最近窗口、C 由种子决定、全选退化为 A、污染未选行 B / C 逐位不变、B / C 不打分、数据几何与改写）；全仓 78 项通过。
- 数据设计：冒烟暴露 v1 问题措辞歧义（模型复述编号），校准集（种子 777，与正式种子不同）证明 v1 密集表（169 / 681 条）超出 0.6B 能力（A ≤ 3/12）；改为 v2：每份 16 条「编号 X：Y」记录穿插无数字中文填充句，尾部两条记录放在受控 token 距离上（倒数第一条 32–60，倒数第二条 150–250），2K 前缀 2037–2060 token、8K 8181–8204；全部校准数字在卡片附录与 `configs/mac_g2lite.json` 的 `revision_history`；解析与终止规则未变；预登记的一次提示修正未使用。
- 结果（Qwen3-0.6B bf16 MPS，ρ ∈ {1/32, 1/8}，k0 = 64 / 256（2K）、256 / 1024（8K），读取数 68 + k0）：基线 A：T1/2K 12/12、T1/8K 10/12（可用）；T2/2K 8/12、T2/8K 1/12（`baseline_too_weak`，不进入对照）。T1 对照：**B 2/12、2/12（2K）与 2/12、3/12（8K），C 0/12 ×4，D0 0/12 ×4**；配对四格没有任何「A 错而稀疏读取对」。B 窗口外 0/34，窗口内 9/14（五个窗口内答错的请求都在大名额或 8K，同一资料上名额变大反而变错出现两次）。失败形态几乎只有一种：答成问题里的编号（C 47/48、D0 48/48、B 窗口外 33/34）；A 在 T1 上从不如此。覆盖率：C 等于 k/M（偏差 ≤ 0.1 pp）；D0 为 k/M 的 1.2–2.2 倍、中间层更成簇，但层组完整覆盖比例 ≤ 2%。反事实：A 改值跟着改 3/3、改无关记录不变 4/4；B / C / D0 改值后 24/24 不变（仍答编号）；移近改写后 B 按资料 3/4、按（资料, ρ）4/8 变对，C / D0 15/16 仍错。「四种方法都对」不存在。
- 判定：T1 对照完成——在 3%–13% 的读取比例下丢掉的是检索本身（退化为复述问题），未训练索引与均匀随机同为零；T2 `baseline_too_weak`。预测对照见卡片：基线主情景、B 窗口外全错、C 接近零且覆盖率 = k/M、D0 不优于 C、三类案例、反事实签名成立；「B 窗口内全对」「C 错误为编造数」「D0 覆盖率 ≈ k/M」失效。不解锁 G3 / G4。
- 证据：`reports/g2lite_mac_m4max.md`、`reports/g2lite_summary_mac_m4max.json`、`reports/g2lite_answers_mac_m4max.json`（逐题回答与解析）、`reports/g2lite_manifest_mac_m4max.json`、`reports/cards/G2lite_mac_m4max.md`（预测写于运行前，附校准附录与预测对照）；原始 `runs/g2lite-*/`（本地），归档 `reports/provenance/g2lite_runs_20261017.tar.gz`（含代表案例的名单位图），`check_provenance.py` 覆盖（正式 run 全部在提交 `af62df4` 干净工作树上），报告可从归档无 GPU 逐字节重生。
- 教学：`notebooks/03_what_is_lost_when_reading_less.ipynb` 从全新内核执行并导出 `reports/03_what_is_lost_when_reading_less.html`（每个案例：资料与问题 → 运行前预测 → 四种方法实际回答 → 各自选择的位置 → 改动一处资料再检查 → 哪些解释得到支持、哪些仍不确定）。
- 消耗：正式三遍 313 s，加冒烟与校准共约 11 分钟设备活跃时间；现金 0；分配内存峰值 3.4 GB。累计约 70 分钟。

## 本轮完成：G2-lite 单变量重放——表头（Mac M4 Max，2026-10-17 14:07 到 14:10 UTC，分支 `task/g2-lite-replay-header`，基于 `task/g2-lite-mac`）

用户授权的卡片「下一步 (a)」：只回答一个问题——第二遍里近期窗口 B 在被查询记录已落入窗口时仍答错（14 个窗口内请求 9 对 5 错，错的全部复述编号），是否由「表头那句『格式为「编号 X：Y」』不在名单里」解释。

- 实现：`decode.select_rows_recent_pinned`（钉住的候选位置先占名额，其余给最近的未钉住候选；F、k、读取数与 B 相同，不打分）；`qwen_wrap.LabState(selector="recent_pinned", pinned_positions=…)`；资料记录 `header_span`（24 份都是 token 位置 [3, 37)，候选域内 33 个）；`bench/g2lite_quality.py` 的两个诊断方法 **BH**（F ∪ 表头 ∪ 最近 k0 − 33 个候选，读取数与 B 相同）与对照 **BM**（F ∪ 最近 k0 − 33 个候选，BH 的窗口、没有表头），run 标 `diagnostic_replay`，旧报告脚本自动排除；配置 v2 文件逐字节不变。契约 +3 项（81 项、139 子测试通过）。预测先写进 `reports/cards/G2lite_replay_header_mac_m4max.md` 并提交（`e3296ec`），再冒烟、再正式运行。
- 结果（run `g2lite-20261017T140840Z-replay-header`，24 份 T1 资料 × B / BH / BM × ρ ∈ {1/32, 1/8}，B 重跑与第二遍 48/48 逐位相同）：钉住 33 个表头位置使 2K · 1/32 的窗口由 93 缩到 60 个资料 token，两份 back_near 资料的记录行被窗口起点切开，所以**配对窗口内集合是 12 个请求（第二遍 B 7 对 5 错）**。**BH 12/12（5 个错误全部变对，7 个对的全部保持）；同窗口不含表头的 BM 7/12，与 B 相同；窗口外 34 个请求三者都 0 对。** 同窗口只差表头（BM 对 BH）：5 个 BM 错 BH 对、0 个反向。只缩短窗口不加表头（B 对 BM）：45/48 回答相同，配对集合里 2 个反向改变（T1-2K-08@1/8 变对、T1-8K-09@1/8 变错）。
- 判定：**(a) 得到支持**——表头在名单里是 B 答对窗口内记录的充分条件；(b) 窗口切入句中、(c) 问题 K/V 由部分读取算出、(d) 窗口越大越易放弃，各自作为充分解释被排除；窗口起点附近内容的次要效应（2/12 反向改变）与 (a) 内部「是格式说明的内容还是靠前的位置」未分离。追加观察：有表头时 8K 的 16 个窗口外请求里 9 个不再复述编号而答成窗口里最后一条记录的值（2K 不变），即 G2-lite 的「退回到复述问题」至少在 8K 上部分是表头缺席的后果。G2-lite 的判定不变；可选的「表头移到末尾」未执行（措辞须改，不是干净的单变量）。
- 证据：`reports/g2lite_replay_header_mac_m4max.md`、`g2lite_replay_header_summary_mac_m4max.json`、`g2lite_replay_header_manifest_mac_m4max.json`、卡片（预测写于运行前，附预测对照：主预测我押的 S2 错、实测 S1）；原始 run 归档 `reports/provenance/g2lite_replay_runs_20261017.tar.gz`（含全部请求的名单位图）；`reports/provenance/g2lite/` 存 G2-lite 四个源文件的运行时快照与本关 diff，`check_provenance.py` 按快照链校验，旧 G2-lite 报告仍可从归档逐字节重生。正式 run 未用 `caffeinate` 包裹（要求在启动后到达），起止连续、无休眠迹象。
- 消耗：设备活跃时间 104 s（冒烟 5 s + 正式 99 s），现金 0，分配内存峰值 3.40 GB。累计约 72 分钟。

## 本轮完成：G2-lite 单变量重放续篇——钉住填充句而非表头（Mac M4 Max，2026-10-17 17:12 到 17:14 UTC，分支 `task/g2-lite-replay-filler-pin`，基于 `task/g2-lite-replay-header`）

用户选择在本会话继续表头重放的下一个单变量：把 BH 钉住的 33 个表头位置换成紧接表头的 33 个填充句位置 [37, 70)（24 份都是无数字的中文填充句，在第一条记录之前；运行时再检查），其余一切不变，与 B、BH 同 run 配对，分离「表头的内容」与「靠前的位置」。

- 实现：`bench/g2lite_quality.py` 诊断方法 **BF**（`recent_pinned` 钉住 [37, 70)，读取数与 B 相同，窗口与 BH 相同）与运行时填充句检查；`scripts/g2lite_replay_report.py` 按 run 里的变体决定列（上一关三个输出从归档逐字节重生已核对）；契约 +1 项（82 项、139 子测试通过）。预测先写进 `reports/cards/G2lite_replay_filler_mac_m4max.md` 并提交（`5d9b9c7`），再冒烟、再正式运行（`caffeinate -i`，日志在 `runs/`）。
- 结果（run `g2lite-20261017T171237Z-replay-filler`，24 份 T1 × B / BH / BF × ρ ∈ {1/32, 1/8}，B 与第二遍、BH 与上一关各 48/48 逐位相同）：**配对窗口内集合 12 个请求上 B 7、BH 12、BF 7；第二遍的 5 个错误 BH 全修好、BF 只修好 1 个（并把 B 原本对的 1 个改错）；BF 对 BH 四格 7 / 0 / 5 / 0。** BF 在 48 个请求上生成的 token 序列与上一关「只缩短窗口、什么都不钉」的对照 BM 全部相同——多读那 33 个填充位置没有改变任何一个生成 token。窗口外 BF 0/34，形态与 B 相同（33/34 编号回声）。
- 判定：**起作用的是表头的内容，不是靠前的位置**（预登记情景 C）。上一关的解释 (a) 收窄为「名单里没有表头那段文本」。仍未分离：表头文本里是格式那半句还是前半句；窗口起点附近内容的次要效应（2/12 反向改变，BM 与 BF 相同）。五条预测全部成立。
- 证据：`reports/g2lite_replay_filler_mac_m4max.md`、`g2lite_replay_filler_summary_mac_m4max.json`、`g2lite_replay_filler_manifest_mac_m4max.json`、卡片（预测写于运行前，附预测对照）；原始 run 归档 `reports/provenance/g2lite_replay_filler_runs_20261017.tar.gz`；`reports/provenance/g2lite_replay/` 存表头重放的 harness 快照与本关 diff，`check_provenance.py` 按链校验。
- 消耗：设备活跃时间 103 s（冒烟 4.5 s + 正式 98.6 s），现金 0，分配内存峰值 3.39 GB。累计约 74 分钟。

## 本轮完成：G2-lite 单变量重放续二——表头拆半钉住（Mac M4 Max，2026-10-17 17:35 到 17:39 UTC，分支 `task/g2-lite-replay-header-split`，基于 `task/g2-lite-replay-filler-pin`）

用户授权的卡片「下一步 (i)」：表头文本里起作用的是哪一半——只钉格式句「格式为「编号 X：Y」，Y 是编号 X 对应的四位数值。」的 token 位置（**BG**，[18, 37)，候选域内 19 个）对比只钉前半句「下面是一份工作记录，其中穿插着若干条编号记录，」的 token 位置（**BA**，[3, 18)，位置 3 在 F 里，候选域内 14 个），各自把其余名额给最近的候选，读取数与 B 相同（窗口分别比 B 少 19 / 14），与 B、BH 同 run 配对。

- 实现：`bench/g2lite_quality.py` 诊断方法 BG / BA、`HEADER_CLAUSES` 与 `header_clause_span`（用 fast tokenizer 的 offset mapping 求两段子句的 token 跨度，运行时核对两段拼起来等于资料里的表头、分割表头跨度、各自解码回原文，否则拒绝运行）；`scripts/g2lite_replay_report.py` 泛化到窗口不同的多个变体（配对集合 = 目标在每个变体窗口内的请求；每个变体各自的窗口与内/外；变体两两四格；前两关六份输出从归档逐字节重生已核对）；契约 +1 项（83 项、139 子测试通过）；`check_provenance.py` 快照链延长（填充句重放的 harness 快照 + 本关 diff）。预测先写进 `reports/cards/G2lite_replay_header_split_mac_m4max.md` 并提交（`3aa1ec8`），再冒烟、再正式运行（`caffeinate -i`，日志在 `runs/`）。预登记几何：2K · 1/32 窗口 B 93 → BH 60 / BG 74 / BA 79，被 BH 切开的两份 back_near 资料在 BG / BA 窗口内；主集合仍是四者都在窗口内的 12 个请求。
- 结果（run `g2lite-20261017T173652Z-replay-header-split`，24 份 T1 × B / BH / BG / BA × ρ ∈ {1/32, 1/8}，B 与第二遍、BH 与前两关各 48/48 逐位相同）：**主集合 12 个请求上 B 7、BH 12、BG 12、BA 7；第二遍的 5 个错误 BG 全修好（0 个改错，BG 对 BH 四格 12 / 0 / 0 / 0），BA 只修好 1 个并改错 1 个——与上一关对照 BM 相同的两个反向改变；BA 在 48 个请求上生成的 token 序列与 BM 在 47 个上相同（第 48 个是几何差别）。** B 窗口内的全部 14 个请求 BG 14/14（BH 13/14 是几何）。窗口外四者 0/34；8K 的 16 个窗口外请求里 BH 9 个改答窗口里最后一条记录的值，BG 3 个（BH 的子集），B / BA 全部复述编号。
- 判定：**格式句单独足够，前半句单独不够**（预登记情景 G，我的主判断）。前两关的解释收窄为「B 在窗口内仍答错是因为名单里没有格式句的 token」；靠前位置、填充句、前半句三者都没有改变过任何一个生成 token。预登记的次要形态预测（BG 在 8K 窗口外「答成记录值」≥ 5/16）失效（3/16，BH 9/16），这个差别来自两个半句的组合或 BH 少读的 14 个窗口 token，未分离。其余预测成立。G2-lite 判定不变，`03_G2_TO_G5.md` 保持 BLOCKED。
- 证据：`reports/g2lite_replay_header_split_mac_m4max.md`、`g2lite_replay_header_split_summary_mac_m4max.json`、`g2lite_replay_header_split_manifest_mac_m4max.json`、卡片（预测写于运行前，附预测对照）；原始 run 归档 `reports/provenance/g2lite_replay_header_split_runs_20261017.tar.gz`（含全部请求的名单位图与清理后的日志）；`reports/provenance/g2lite_replay/` 新增填充句重放的 harness 快照与本关 diff，`check_provenance.py` 按链校验。
- 消耗：设备活跃时间 127 s（冒烟 5.5 s + 正式 121.3 s），现金 0，分配内存峰值 3.40 GB。累计约 76 分钟。

## 本轮完成：G2-lite 单变量重放续三——格式句拆半钉住（Mac M4 Max，2026-10-17 18:18 到 18:22 UTC，分支 `task/g2-lite-replay-format-split`，基于 `task/g2-lite-replay-header-split`）

用户选择在本会话继续续二的「下一步 (i)」：格式句里起作用的是哪一段——只钉前段「格式为「编号 X：Y」，」的 token 位置（**BG1**，[18, 27)，候选域内 9 个）对比只钉后段「Y 是编号 X 对应的四位数值。」的 token 位置（**BG2**，[27, 37)，候选域内 10 个），各自把其余名额给最近的候选，读取数与 B 相同（窗口分别比 B 少 9 / 10），与 B、BH、BG 同 run 配对。

- 实现：`bench/g2lite_quality.py` 诊断方法 BG1 / BG2、`FORMAT_CLAUSE_PARTS`，钉住跨度的检查泛化为 `clause_part_span`（两段拼起来等于整段、分割整段跨度、各自解码回原文，否则拒绝运行；表头两段的检查行为不变）；`scripts/g2lite_replay_report.py` 登记新变体（前三关九份输出从归档逐字节重生已核对）；契约 +1 项（84 项、139 子测试通过）；`check_provenance.py` 快照链再延长（续二的 harness 快照 + 本关 diff）。预测先写进 `reports/cards/G2lite_replay_format_split_mac_m4max.md` 并提交（`4b0cb1a`），再冒烟、再正式运行（`caffeinate -i`，日志在 `runs/`）。
- 结果（run `g2lite-20261017T181951Z-replay-format-split`，24 份 T1 × B / BH / BG / BG1 / BG2 × ρ ∈ {1/32, 1/8}，B 与第二遍、BH 与前三关、BG 与续二各 48/48 逐位相同）：**主集合 12 个请求上 B 7、BH 12、BG 12、BG1 8、BG2 11；第二遍的 5 个错误 BG2 修好 4 个（第 5 个 T1-8K-11@1/8 答成不在资料里的近似数 8339，目标 9339），BG1 只修好 1 个——且那是缩窗口对照都出现过的反向改变，BG1 的生成序列与 BA 47/48 相同。** 两者都没有改错 B 原本答对的。窗口外五者 0/34；8K 的 16 个窗口外请求里改答窗口里最后一条记录的值的个数随钉住的表头文本量单调：BH 9、BG 3、BG2 2、BG1 0。
- 判定：**后段单独足够（预登记情景 T，边界值 11/12），前段单独不够（边界值 8/12）**；四关链条：整句表头 = 格式句 = 12/12 ＞ 后段 11/12 ≫ 前段 ≈ 前半句 ≈ 填充句 ≈ 只缩窗口 ≈ B。「编号」一词次数不能解释效应（两段各一次）；后段独有「对应的四位数值」（与问题几乎逐字重合）与定义语义仍分不开，要分离需改字。五条预测全部成立（主预测在边界上）。G2-lite 判定不变，`03_G2_TO_G5.md` 保持 BLOCKED。
- 证据：`reports/g2lite_replay_format_split_mac_m4max.md`、`g2lite_replay_format_split_summary_mac_m4max.json`、`g2lite_replay_format_split_manifest_mac_m4max.json`、卡片（预测写于运行前，附预测对照）；原始 run 归档 `reports/provenance/g2lite_replay_format_split_runs_20261017.tar.gz`（含全部请求的名单位图与清理后的日志）；`reports/provenance/g2lite_replay/` 新增续二的 harness 快照与本关 diff，`check_provenance.py` 按链校验。
- 消耗：设备活跃时间 154 s（冒烟 6.4 s + 正式 147.7 s），现金 0，分配内存峰值 3.40 GB。累计约 79 分钟。

## 尚未完成

速度主线按 G1b 收口，融合核路线在 P1 后由用户决定停止（原型与记录在 PR #3，不接入模型路径）；以下均未启动、需用户决定与授权：CUDA 复核轨道（4080 主机）；> 32,768 的上下文；G2 到 G5 的任何质量结论（`docs/tasks/03_G2_TO_G5.md` 仍为 BLOCKED）。本机磁盘在 G1b 运行时只剩约 28 GiB 可用（97%），后续大 run 前需留意。

## 下一步

1. 用户评审 G2-lite 的 PR（分支 `task/g2-lite-mac`，不自行合并）。
2. 用户评审表头重放的 PR #6（分支 `task/g2-lite-replay-header`）、续篇的 PR #7（分支 `task/g2-lite-replay-filler-pin`，基于前者）、续二的 PR #8（分支 `task/g2-lite-replay-header-split`，基于 PR #7）与续三的 PR（分支 `task/g2-lite-replay-format-split`，基于 PR #8），都指向 main，不自行合并。
3. 用户决定方向（都需要明确授权与新任务书，不自动开始）：(a) 后段再拆半钉住（「Y 是编号 X」对「对应的四位数值。」，已接近词面重合与语义分不开的粒度）；为 BG2 配 k0 − 10 的只缩窗口对照以分离 T1-8K-11@1/8 上 BG2 与 BG 的差别；改字的变体（同义改写定义句或问题措辞）需新资料与 A 基线；每项约 2–3 分钟设备时间；(b) T2 换更大模型（Qwen3-1.7B）复测基线，需下载约 3.4 GB；(c) G3 训练索引器后在同一协议下复测 D——唯一能把 D 与 C 分开的路径，但速度主线已收口；(d) 融合核 / CUDA / 更长上下文的旧选项不变。
4. 不重跑 G0/G1a/G1b/G2-lite 的任何一遍，不训练，不租云。

```bash
source .venv/bin/activate
python -m pytest -q tests                                  # 全部契约（含 G1b 封装；真实 checkpoint 项只在本地快照存在时跑）
python bench/model_numerics.py --tag formalN               # G1b 三段数值对照，约 25 秒（需本地快照）
python bench/model_decode_bench.py --compile-blocks --tag formal   # G1b 整模型测速，约 6.5 分钟
python scripts/g1b_report.py                               # 由非 quick run 重新生成 G1b 报告
python bench/microbench.py --modes eager --tag pN          # G1a 单进程微基准，约 5 分钟
python scripts/g1a_report.py
HF_HUB_OFFLINE=1 python bench/g2lite_quality.py --methods A --tag pass1-A                      # G2-lite 第一遍：A 基线，约 90 秒
HF_HUB_OFFLINE=1 python bench/g2lite_quality.py --usable-from runs/<pass1>/summary.json --tag pass2   # 第二遍：可用格子四方法，约 2.5 分钟
HF_HUB_OFFLINE=1 python bench/g2lite_quality.py --usable-from runs/<pass1>/summary.json --variants value irrelevant --doc-indices 0 4 --tag pass3a
HF_HUB_OFFLINE=1 python bench/g2lite_quality.py --usable-from runs/<pass1>/summary.json --variants move --cases-from runs/<pass2>/quality.jsonl --tag pass3b
python scripts/g2lite_report.py                            # 由三遍 run 重新生成 G2-lite 报告（不加载模型）
HF_HUB_OFFLINE=1 caffeinate -i python bench/g2lite_quality.py --tasks T1 --lengths 2K 8K --methods B BH BM --tag replay-header   # 表头重放：24 份 × 6 请求，约 100 秒
python scripts/g2lite_replay_report.py                     # 与归档的第二遍 B 配对（需 runs/ 里有 pass2 run，见 reports/provenance/README.md）
HF_HUB_OFFLINE=1 caffeinate -i python bench/g2lite_quality.py --tasks T1 --lengths 2K 8K --methods B BH BF --tag replay-filler   # 续篇：钉住填充句，约 100 秒
python scripts/g2lite_replay_report.py --title "…" --card reports/cards/G2lite_replay_filler_mac_m4max.md --out reports/g2lite_replay_filler_mac_m4max.md --json reports/g2lite_replay_filler_summary_mac_m4max.json --manifest reports/g2lite_replay_filler_manifest_mac_m4max.json
HF_HUB_OFFLINE=1 caffeinate -i python bench/g2lite_quality.py --tasks T1 --lengths 2K 8K --methods B BH BG BA --tag replay-header-split   # 续二：只钉格式句 / 只钉前半句，约 120 秒
python scripts/g2lite_replay_report.py --title "…" --card reports/cards/G2lite_replay_header_split_mac_m4max.md --out reports/g2lite_replay_header_split_mac_m4max.md --json reports/g2lite_replay_header_split_summary_mac_m4max.json --manifest reports/g2lite_replay_header_split_manifest_mac_m4max.json
HF_HUB_OFFLINE=1 caffeinate -i python bench/g2lite_quality.py --tasks T1 --lengths 2K 8K --methods B BH BG BG1 BG2 --tag replay-format-split   # 续三：格式句拆半钉住，约 150 秒
python scripts/g2lite_replay_report.py --title "…" --card reports/cards/G2lite_replay_format_split_mac_m4max.md --out reports/g2lite_replay_format_split_mac_m4max.md --json reports/g2lite_replay_format_split_summary_mac_m4max.json --manifest reports/g2lite_replay_format_split_manifest_mac_m4max.json
```

## 接力格式

每次记录分支、代码 SHA、run_id、实际命令、证据路径、失败/未测项、剩余预算与下一条命令。未完成进程的 PID 与主机路径保存在本地忽略目录，不直接写进公开报告；公开接力只提供去标识化状态。
