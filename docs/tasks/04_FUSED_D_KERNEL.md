# 任务书：融合 D 链的自定义 Metal 核

状态：**P0 已完成并通过停止门（2026-10-17；两个独立进程 32K 处 fused 69.7 / 70.0 µs ≤ 80，对 PyTorch 链 0.69 / 0.68 ≤ 0.75，数值通过）。P1 已完成，停止门未通过（2026-10-17：S = 1076 处 38.5 / 39.8 µs ≤ 60 通过、数值通过，但对 PyTorch 取行路径的比值 0.82 / 0.83 > 0.80）；按规则停止。**用户于 2026-10-17 决定停止融合核路线（"停止"）：P2 到 P4 不做，P0 / P1 的核、契约、基准与报告留作记录，不接进模型路径；速度主线维持 G1b 的收口。** P1 结果：`reports/p1_attend_kernel_mac_m4max.md`、`reports/cards/P1_fused_attend_mac_m4max.md`。分支 `task/p0-fused-score-kernel`（基于 `task/g1b-qwen-mac`，PR #2 尚未合并）。结果：`reports/p0_score_kernel_mac_m4max.md`、`reports/cards/P0_fused_score_mac_m4max.md`。范围说明见 [`../SCOPING_FUSED_D_KERNEL.md`](../SCOPING_FUSED_D_KERNEL.md)，G1b 的结论与数据见 `reports/cards/G1b_mac_m4max.md`。

## 为什么有这一步

G1b 在 32K、ρ = 1/32 下整模型一步 D/A = 0.930（实测），没过 10% 门槛；剖析把抵消项定位到 D 链每层十来个相互依赖的小核的固定成本，而不是读的字节。要越过门槛只能把链压成少数几个核。P0 只回答一个问题：**链里最大的一段——打分（含新 key 的 z 写入与 u 投影）——用一个自定义 Metal 核做，能否接近带宽下界。**

## P0 的范围（只做这些）

- 一个 Metal 核（`torch.mps.compile_shader`，本机 torch 2.14.1 已核实可用）：输入 `zT [G, r, C]`（bf16）、`k_pre_new [G, d]`、`Wk [G, r, d]`、`q_cat [G, h·d]`、`Wq_t [G, h·d, r]`、`lo`、`hi`、`write_col`；输出 fp32 分数 `[G, hi−lo]`，并把新 key 的 z 写进 `zT` 的 `write_col` 列。另给一个两次启动的变体（序曲核 + 打分核）作对照。
- 对照：PyTorch 现有路径同样的工作（`project_keys` + 列写 + `project_queries` + `candidate_scores`，3 个 bmm 加 1 个 copy），以及只打分的 `candidate_scores`。
- 随机张量、bf16、B = 1、G = 8、h = 2、d = 128、r = 32，N0 ∈ {8192, 16384, 32256}（容量 N0 + 128，合法长度 N0 + 1，候选域 [4, L − 64)），28 层轮换工作集，预热 20、正式 50，每 pass 前后同步；至少两个独立进程。
- 正确性：fused 分数对 CPU float32 参照；z 列对 `project_keys`；候选域外的列毒化为 NaN 后分数必须有限；M 不是线程组大小整数倍时的边界。
- **不碰模型代码**，不改 `qwen_wrap`、不改 G1a 核、不下载、不训练、不用云、不花钱。

## P1 的范围（已授权，只做这些）

- 一个取行注意力 Metal 核（可带一个合并核）：输入主 K/V 缓存 `[G, C, d]`（bf16，batch 1）、`q_main [G, h, d]`、行表 `rows [G, S]`（int32 原位置，F ∪ 入选候选，不重编号）；输出 `[G, h, d]`。按行表直接从缓存读 K/V 行，组内两个查询头共用同一次读取，在线 softmax，fp32 累加；不物化 gather。
- 对照：G1a 的 PyTorch 路径（拉平 `index_select` 取 K、取 V、`subset_attention`，3 次核调用）；上下文参照：A 的完整 SDPA 读取（不参与门槛）。
- 随机张量、bf16、G = 8、h = 2、d = 128；S 由 ρ = 1/32 的 k 规则给出（8K 324、16K 580、32K 1076），28 层轮换；门槛在 S = 1076 处判定。行表的 int32 转换在计时之外（P2 的选择核应直接产出 int32）。
- 不碰模型代码、不改 G1a 核、不接进 `qwen_wrap`。

## 停止门（开工前写定）

P0 通过并建议进入 P1 的条件（三者同时满足）：
1. 32K 处 fused 打分核（含序曲）每层 ≤ 80 µs（28 层轮换中位数，每个独立进程都满足）；
2. fused fp32 分数对 CPU float32 参照：相对范数 ≤ 1e-5、max|Δ| ≤ 1e-5；z 列对 `project_keys` 的差 ≤ 1 个 bf16 ulp；毒化域外列后分数有限；
3. fused 对 PyTorch 四算子链的比值 ≤ 0.75。

任一不满足即停，保存负结果。即便通过，**P1 仍需用户另行授权**。预测写在 `reports/cards/P0_fused_score_mac_m4max.md`，运行前提交。

P1 的停止门（开工前写定，2026-10-17）：
1. S = 1076（N0 = 32256，ρ = 1/32）处融合取行注意力每层 ≤ 60 µs（28 层轮换中位数，每个独立进程都满足；若有合并核，两次启动合计）；
2. 与 `dense_mask_reference` 同义的契约通过：同一行表同一输出（fp32 输出对 CPU float32 参照 max|Δ| ≤ 5e-5、相对范数 ≤ 1e-5；bf16 输出与 fp32 输出的 RNE 舍入差 ≤ 1 ulp）；未入选行毒化为 NaN 后输出逐位不变；行表乱序与有序结果一致（容差内）；原位置取行正确（与 PyTorch gather 路径同行）；S 不是 4 或分区数整数倍时边界正确；
3. 融合核对 PyTorch 三算子路径（取 K、取 V、子集 SDPA）的比值 ≤ 0.80。

预测写在 `reports/cards/P1_fused_attend_mac_m4max.md`，运行前提交。即便通过，**P2 仍需用户另行授权**。

## 交付

`src/attention_lab/metal_score.py`（核源码与 Python 封装）、`tests/test_metal_score.py`、`bench/p0_score_kernel.py`、`reports/p0_score_kernel_mac_m4max.{md,json}`、卡片、`PROJECT_STATUS.md` 更新；原始 run 在 `runs/p0-score-*/`（本地）。

## 预算

工程时间半天；设备活跃时间 < 10 分钟；现金 0。
