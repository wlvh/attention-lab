# 任务书：融合 D 链的自定义 Metal 核

状态：**P0 已完成并通过停止门（2026-10-17；两个独立进程 32K 处 fused 69.7 / 70.0 µs ≤ 80，对 PyTorch 链 0.69 / 0.68 ≤ 0.75，数值通过）；P1 到 P4 未授权。** 分支 `task/p0-fused-score-kernel`（基于 `task/g1b-qwen-mac`，PR #2 尚未合并）。结果：`reports/p0_score_kernel_mac_m4max.md`、`reports/cards/P0_fused_score_mac_m4max.md`。范围说明见 [`../SCOPING_FUSED_D_KERNEL.md`](../SCOPING_FUSED_D_KERNEL.md)，G1b 的结论与数据见 `reports/cards/G1b_mac_m4max.md`。

## 为什么有这一步

G1b 在 32K、ρ = 1/32 下整模型一步 D/A = 0.930（实测），没过 10% 门槛；剖析把抵消项定位到 D 链每层十来个相互依赖的小核的固定成本，而不是读的字节。要越过门槛只能把链压成少数几个核。P0 只回答一个问题：**链里最大的一段——打分（含新 key 的 z 写入与 u 投影）——用一个自定义 Metal 核做，能否接近带宽下界。**

## P0 的范围（只做这些）

- 一个 Metal 核（`torch.mps.compile_shader`，本机 torch 2.14.1 已核实可用）：输入 `zT [G, r, C]`（bf16）、`k_pre_new [G, d]`、`Wk [G, r, d]`、`q_cat [G, h·d]`、`Wq_t [G, h·d, r]`、`lo`、`hi`、`write_col`；输出 fp32 分数 `[G, hi−lo]`，并把新 key 的 z 写进 `zT` 的 `write_col` 列。另给一个两次启动的变体（序曲核 + 打分核）作对照。
- 对照：PyTorch 现有路径同样的工作（`project_keys` + 列写 + `project_queries` + `candidate_scores`，3 个 bmm 加 1 个 copy），以及只打分的 `candidate_scores`。
- 随机张量、bf16、B = 1、G = 8、h = 2、d = 128、r = 32，N0 ∈ {8192, 16384, 32256}（容量 N0 + 128，合法长度 N0 + 1，候选域 [4, L − 64)），28 层轮换工作集，预热 20、正式 50，每 pass 前后同步；至少两个独立进程。
- 正确性：fused 分数对 CPU float32 参照；z 列对 `project_keys`；候选域外的列毒化为 NaN 后分数必须有限；M 不是线程组大小整数倍时的边界。
- **不碰模型代码**，不改 `qwen_wrap`、不改 G1a 核、不下载、不训练、不用云、不花钱。

## 停止门（开工前写定）

P0 通过并建议进入 P1 的条件（三者同时满足）：
1. 32K 处 fused 打分核（含序曲）每层 ≤ 80 µs（28 层轮换中位数，每个独立进程都满足）；
2. fused fp32 分数对 CPU float32 参照：相对范数 ≤ 1e-5、max|Δ| ≤ 1e-5；z 列对 `project_keys` 的差 ≤ 1 个 bf16 ulp；毒化域外列后分数有限；
3. fused 对 PyTorch 四算子链的比值 ≤ 0.75。

任一不满足即停，保存负结果。即便通过，**P1 仍需用户另行授权**。预测写在 `reports/cards/P0_fused_score_mac_m4max.md`，运行前提交。

## 交付

`src/attention_lab/metal_score.py`（核源码与 Python 封装）、`tests/test_metal_score.py`、`bench/p0_score_kernel.py`、`reports/p0_score_kernel_mac_m4max.{md,json}`、卡片、`PROJECT_STATUS.md` 更新；原始 run 在 `runs/p0-score-*/`（本地）。

## 预算

工程时间半天；设备活跃时间 < 10 分钟；现金 0。
