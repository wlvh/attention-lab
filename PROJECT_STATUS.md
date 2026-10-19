# 接力状态

## 当前持续目标：G3b-lite → G3c → G4-lite（2026-10-18）

用户已明确授权连续研发，三阶段共享新增14400秒真实设备运行预算、现金0；完成各里程碑提交PR后继续已授权下游，后续PR不自行合并。PR #12 在已审 head 45adb93、两项CI成功且无冲突时以 merge commit 合并为 `0dd6b73`，分支保留，Issue #11 已关闭。本地main已快进并核对父提交。

任务队列：

- [g3b](https://github.com/wlvh/attention-lab/issues/13)
- [g3c](https://github.com/wlvh/attention-lab/issues/14)
- [g4](https://github.com/wlvh/attention-lab/issues/15)

固定条件与预测见 `docs/tasks/07_G3B_LITE.md`、`08_G3C_REAL_SPARSE.md`、`09_G4_LITE.md` 和 `configs/mac_g3bg4lite.json`。当前分支 `task/g3b-lite` 基于main `0dd6b73`。G3a旧48份训练可复用，旧16份降格为开发诊断；新32份最终测试先登记再封存，首次评分到G4。主体冻结、P0/r32/rho=1/8，一个种子；第14层复用已审相同条件参数，其余27层训练。

磁盘仅余约23GiB：逐层内存处理，只保留每层固定2份完整审计特征，其余数组记录hash后释放；全部指标、参数及输入可追溯，不删除旧run/缓存。共享预算账将单独写入 `runs/g3bg4lite-20261018-budget.json` 并引用旧G3a账，不重置或覆盖旧账。

G3b已完成：`runs/g3b-20261018T142640Z-formal`，运行源码 `2b6db0bc95641403ce5d728bcdbc4a16c209cf4a`。28层参数齐全：27层各50轮新训练，第14层复用已审G3a。28层开发资料平均KL、coverage、局部相对L2均优于D0；这不代表整模型自由生成已恢复。主体前后hash相同。共享新增设备时间3537.07/14400秒（含预检），现金0；没有活跃作业。账本实际路径为 `runs/g3bg4lite-20261018-budget.json`，归档包含截至G3b结束的快照。

证据：[报告](reports/g3b_mac_m4max.md)、[各层参数身份](reports/g3b_parameters.json)、[精简归档](reports/provenance/g3b_evidence_20261018.tar.gz)及清单，全部逐查询指标重算一致。归档约38MiB；完整审计数组仅在本机保留，不宣称远端材料可直接重做全部特征级审计。26项CPU契约/归档检查通过。第五课6个代码单元从新内核顺序执行并导出[HTML](reports/05_training_all_layer_indexers.html)，三联图实际检查过标题、坐标、图例与第14层标记；整页入口返回queued，整页与窄屏布局仍未验。

下一步：提交G3b待审PR后，从固定提交建立G3c依赖分支；先CPU检查单层/全层接入与全选等价，再用4份旧开发资料和固定token序列检查真实状态差异。32份新最终测试还未评分。报告复核命令：`python scripts/g3b_report.py`，不会运行模型。页面预览、PR #10与历史诊断归档不阻挡研发；Metal/CUDA、更长上下文、T2、付费服务仍未授权。以下各节保留此前交付时点，旧停止点由本节限定授权取代。

## 2026-10-18 PR #12 复审补丁：预检绑定执行源码

本轮只修 P2 预检复用条件，未运行模型或训练，未合并 PR。`bench/g3a_train.py` 现在同时读取预检 summary/manifest，检查完成状态、记录配对、G3a 配置与基础数据配置的 SHA-256，以及 12 个本地执行源文件（运行器、特征/教师/索引训练、数据生成、完整注意力及依赖）。缺失或不一致拒绝；Git 提交号不作为门槛，文档、Notebook、报告和无关实验文件的提交不影响有效预检。

核验记录：[g3a_preflight_binding_check.json](reports/g3a_preflight_binding_check.json)。成功预检 `1ee30e2` 与正式运行 `b998ee7` 的两个配置哈希和全部 23 个记录源码哈希一致，两个提交间仅新增 `scripts/g3a_report.py`；本机的四份 summary/manifest 与归档逐字节一致。历史绑定通过；当前执行器源码已改变，因此旧预检不被当作修后版本的新运行许可。

CPU 回归 **18 项通过**（原有 12 项加 6 项预检绑定测试，涵盖全部 12 个执行文件的变化与缺失反例）。旧 run、manifest、归档、索引参数、成绩、Notebook 和 HTML 均未改写；第四课整页及窄屏布局仍未验。下一步只等待此补丁的最终合并确认；CPU 核验命令：`.venv/bin/python -m pytest tests/test_g3a.py tests/test_g3a_report.py -q`。

## 2026-10-18 当前增量：G3a 已完成，待 PR 复审

基于已合并 PR #5 的 main `d2fc4ad` 完成独立缩减任务 [Issue #11](https://github.com/wlvh/attention-lab/issues/11)，分支 `task/g3a-single-layer`。任务书 [06_G3A_SINGLE_LAYER.md](docs/tasks/06_G3A_SINGLE_LAYER.md)，运行前预测提交 `ee7e2f1`。本节是最新状态，下文保留 G0—G2-lite 的历史交付记录。

- 正式 run：`g3a-20261018T122927Z-formal`，运行代码 `b998ee7`；只训练第 14 层 P0/r32 索引器的 98,304 个参数，主体冻结。新 T1 48/16 划分，ρ=1/8，固定 50 轮。留出最终检查之前不选模型或调参。
- 留出结果：D0→D 的 KL 2.85786→0.13534、条件教师覆盖率 24.36%→87.83%、局部输出相对 L2 0.20997→0.02196；16/16 资料平均指标改善。只支持同分布固定特征上的单层学习，不证明答题、自由生成或速度改善。
- 原始数据、参数、指标、源码、失败预检和预算：[精简归档](reports/provenance/g3a_evidence_20261018.tar.gz) 及 [清单](reports/provenance/g3a_archive_manifest.json)；完整特征约 778MB 留在 `runs/g3a-20261018T122927Z-formal/features/`，每份 hash 在原 manifest。没有覆盖旧 runs、manifest 或冻结分数。
- 一次预检定位：整句填充导致长度调整振荡；训练前仅对 G3a 明确 2048±64 token，保留种子/样本和目标尾距。全部 64 份实际 1991–2082 token，与旧资料 hash 无重合。含失败、通过预检和正式运行累计 313.8 秒，新增现金 0；记录中 swap 为 0。
- 验证：12 项新增 CPU 契约与归档检查通过；[独立 NumPy 审计](reports/g3a_cpu_audit.json) 重算全部 16 份留出资料、2048 项查询/组/方法比较，最大差异 3.33e-6 < 预设 1e-4。Wq/Wk 实值变化范数分别为 8.1064/7.8479，主体前后 hash 相同。报告入口用禁用 site-packages 的 Python 成功重建，不需要 torch。
- 教学：[实验课04 HTML](reports/04_can_an_indexer_learn.html)、[Notebook](notebooks/04_can_an_indexer_learn.ipynb)、[报告](reports/g3a_mac_m4max.md)。10 个代码单元从新内核执行无错误；默认只读保存结果与 CPU 张量小例子。三张 PNG 检查过文字/坐标/图例/位置阴影，方法颜色已统一。整页 HTML 预览返回 queued，整页与窄屏布局未验；不把单元执行或 PNG 检查当整页验收。
- 运行后仅补强未来运行的“参数已更新”检查：比较张量实值，不只比较压缩文件 hash。原执行版本在归档，旧 manifest 未改，没有重跑模型。
- PR #6–#9 保持开放，不合并/删除/改写；PR #10 仍为 `e5490bc`、开放。本轮只读复核发现其中 18:44 状态所写“G3 未授权”已被后续 G3a 范围取代，合并该文档 PR 前须更新当前段，历史时点应保留；不允许其旧状态覆盖本节。

下一步：复审本分支的单层证据与代码；不启动全层训练。下一条只读命令：`python scripts/g3a_report.py`。完整 G3–G5、G2-lite 配置 v3 的新模型运行、T2 新变式、速度优化、Metal/CUDA/更长上下文、云与付费 API 都未解锁。本轮停在提交 PR 后，任何合并需用户另行明确授权。

更新日期：2026-10-18。状态：**G0_DONE / G1A_DONE（通过，continue）/ PR1_MERGED（`0798aa7`）/ NOTEBOOK_01_DONE / G1B_DONE（工程完成；性能 `correct_but_not_faster`，收口速度主线）/ NOTEBOOK_02_DONE / P0_FUSED_SCORE_DONE（停止门通过）/ P1_FUSED_ATTEND_DONE（停止门未通过：比值 0.82 / 0.83 > 0.80）/ FUSED_KERNEL_LINE_STOPPED（用户决定，2026-10-17）/ PR2_MERGED（`e682e9b`）/ PR3_MERGED（`158bad9`）/ PR4_MERGED（`5f55729`）/ G2LITE_DONE / PR5_REVIEW_FIX_READY（旧结果保留，配置 v3 NOT_RUN，待复审）**。`task/p0-fused-score-kernel`（PR #3）是已合并归档的停止原型；可合并的记录不等于内核完成生产验收。

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

## 当前收尾：PR #5 审查修复（2026-10-18）

PR #5 分支 `task/g2-lite-mac` 在旧 head `bb1b639` 后追加收尾修复。当前目标只到复审，不合并；main 和诊断分支 #6–#9 不改。旧 G2-lite 正式运行代码为 `af62df4`、配置 v2；四个 run 均保留，未重跑模型。本次配置 v3 只登记 C 的稳定父资料种子规则，状态 **NOT_RUN**。

- 主结果：T1 的 A 2K 12/12、8K 10/12；B 两档分别为 2/12、2/12 和 2/12、3/12；C/D0 四格均 0/12。T2 在当前提示、生成及评分协议下为 baseline_too_weak（8/12、1/12），未进入稀疏对照。所有逐题答案和评分保留。
- 修复：C 控制外部随机性（D0 内容响应不冻结）；回答变化、正误变化、编号回声分别从记录计数；旧 C 改写标混杂；案例目标及邻域不再被覆盖；实际预算范围和ρ/k/M/总读取比例分列；报告入口不导入 torch；运行计划/完成/跳过与原因分开，未完成不写 done。
- 解释：撤回窗口外必不变、必须同层组读齐四位值和结构性下界等强结论。读齐仍错只否定充分性；必要性未经证明。完整 Prefill 允许跨位置传播信息，具体路径未定位。
- 证据：`reports/g2lite_mac_m4max.md`、`reports/g2lite_summary_mac_m4max.json`、不变的 `reports/g2lite_answers_mac_m4max.json` / `reports/g2lite_manifest_mac_m4max.json`；旧归档 `reports/provenance/g2lite_runs_20261017.tar.gz`；新增运行时快照 `reports/provenance/g2lite_pr5_review/`、独立修复来源清单 `reports/provenance/g2lite_pr5_review_manifest.json` 和新报告生成清单 `reports/g2lite_report_build.json`。
- 教学：`notebooks/03_what_is_lost_when_reading_less.ipynb` 与 `reports/03_what_is_lost_when_reading_less.html`。原预测保留在卡片，审查修订另列；动态计数由 `.py` 汇总实现生成，Notebook 不另维护评分算法。
- 验收：25 项针对性 CPU 测试、5 个子测试通过；无 torch 的入口及旧归档重生成通过；89 项标准库环境测试通过（53 项依赖相关跳过）；原件、旧答案/manifest、主表/门槛/配对表与预测原文不变量通过。Notebook 新内核顺序执行 18 个代码单元、无错误，已导出 HTML；9 张图与旧版逐字节相同，5 张独立图已逐一检查。整页 HTML 布局未验：此前浏览器拒绝 file://，本轮受支持的 Codex 文件预览返回 queued，未确认实际展示。生成代码提交 `72b848d`；远端 CI 与本地检查分别确认，以 PR 当前 head 为准。
- 早期证据缺口：v1 calib-w2/w4/w5 的生成器源码未恢复；旧配置/输出仍可读。资料设计调整不是密度的单变量实验。诊断报告路径/default 输出问题留待处理 #6–#9，不扩大本补丁。

## 下一步与停止点

本轮代码、旧记录重生成与实验课执行已完成，交付修复提交后等待复审。合并必须由用户另行明确指令，采用 merge commit 并保留分支。G3、融合 Metal、CUDA、更长上下文、追加ρ及T2变式均未授权。没有新增云或付费 API 支出。

复核命令（使用现有隔离 Python 环境，不初始化环境、不启动 GPU）：

```bash
ATTENTION_LAB_TEST_CPU_ONLY=1 HF_HUB_OFFLINE=1 python -m pytest -q tests/test_g2lite.py tests/test_g2lite_review.py
python scripts/check_sources.py
python scripts/check_repository.py
python scripts/check_provenance.py
python scripts/g2lite_report.py --review-card reports/cards/G2lite_mac_m4max.md
```

## 接力格式

每次记录分支、代码 SHA、run_id、实际命令、证据路径、失败/未测项、剩余预算与下一条命令。未完成进程的 PID 与主机路径保存在本地忽略目录，不直接写进公开报告；公开接力只提供去标识化状态。

## 2026-10-19 复审收尾：共享预算跨进程独占

PR #16追加未来运行的预算保护：在读取/检查/登记之前取得同一账本的POSIX独占锁，持有到退出累计写回完成，竞争者立即拒绝。锁文件保留，避免换inode；正常与可捕获异常都记账并释放，强制终止或写盘失败后保留未完成登记，下一作业必须先核清，不自动猜测用时。

[CPU检查记录](reports/campaign_budget_lock_checks.json)：9项预算契约通过，含真实双进程同时启动、顺序启动、10+20=30的合成计时、失败/强制中断和锁释放；连同G3b回归共14项通过。新测试在旧源码上复现两项并发失败。这里只修未来保护；原run、manifest、参数和成绩保持不变，5次历史运行的完整对账随PR #18交付。后续按依赖同步到#17/#18并提交复审，不执行GitHub合并，不新增模型实验。
