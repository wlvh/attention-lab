# Gate card · G0 环境预检与带宽探针（Mac M4 Max）

Gate: G0 · run_id: `g0-mac-20261016T131252Z`（同日前两次运行 `130936Z`、`131139Z` 为同一探针的早期版本，结果一致，缺少后加的索引路径变式）· code commit: 待提交，分支 `task/g0-g1a-mac`

## 问题与执行前预测

**问题。** 这台 MacBook Pro（Apple M4 Max，16 核 CPU，40 核 GPU，128 GB）上，PyTorch MPS 能否正确执行本项目需要的运算？有效访存带宽多少？eager 模式每个小算子的提交开销多少？这些数字决定 G1a 预测表的量纲。

**预测写于 2026-10-16，运行之前。** 依据：Apple 公布的 M4 Max 40 核版标称带宽 546 GB/s；MPS 后端的一般经验；没有本机实测。

| 项 | 预测 | 依据 |
|---|---|---|
| torch 在 Python 3.12 venv 安装、MPS 可用 | 是 | macOS 27、Apple Silicon |
| `recommended_max_memory` | 90 到 105 GB | Metal 建议工作集约为统一内存的四分之三 |
| matmul fp32 / fp16 / bf16 | 支持 | 近年 torch 对 MPS 的覆盖 |
| matmul fp64 | 不支持，报错 | MPS 无 fp64 |
| top-k、index_select、bool mask softmax | 支持且与 CPU 一致 | 基本算子 |
| SDPA 普通多头（KV 已 repeat） | 支持 | 已有 MPS 内核 |
| SDPA `enable_gqa=True` | 不支持或报错 | GQA 融合路径多为 CUDA 专有 |
| SDPA bool attn_mask | 支持 | |
| 小索引矩阵 KL 反向 | 正常，梯度与 CPU 相对误差 ≤ 1e-4 | |
| `torch.compile` 逐元素小函数 | 五成把握能跑，首次编译 20 到 90 秒 | inductor 的 MPS 后端仍属实验性 |
| 大张量 copy（计读写）≥ 1 GiB | 300 到 450 GB/s | 标称的 55% 到 80% |
| 大张量 sum（只读）≥ 1 GiB | 250 到 450 GB/s | 归约内核效率略低 |
| 注意力形状扫描 q·Kᵀ，bf16，36 层轮换，每层 64 MiB | 150 到 350 GB/s | 单查询 matmul 不是纯流式内核 |
| 同一层热读 36 次 | 与轮换接近 | 64 MiB 已超过片上缓存 |
| eager 每个小算子 | 15 到 60 µs | MPS 提交开销高于 CUDA |
| `torch.mps.synchronize()` 单次 | 50 到 200 µs | |

**若预测成立，对 G1a 的含义。** 以 350 GB/s 只读带宽代入 roofline，4B 在 32K 的 S1 情景每步省约 11 到 12 ms，占理想步长约三成；每层 7 个额外算子、每个 30 µs，36 层新增 CPU 提交约 7.6 ms。8K 处的理想节省约 4 ms，与新增提交开销同量级，eager 下方向不确定。这些是预测，待 G0 实测后用真实带宽与提交开销重算。

## 实际执行

2026-10-16 13:00 到 13:13 UTC，本机，无云，无模型下载，无训练。

- 环境：Homebrew Python 3.12.10 建 `.venv`；`pip install -r requirements/bootstrap.in` 得到 torch 2.14.1、transformers 5.20.0、numpy 2.5.4，`pip check` 无冲突；`pip freeze` 写入 `requirements/lock-macos-arm64.txt`（126 行）。
- 清单：`scripts/environment_inventory.py --require-mps` 退出码 0；MPS built 且 available；CUDA 不可用。
- 探针：`scripts/g0_probe.py`，未设置任何 `PYTORCH_*` 环境变量，脚本在 fallback 开启时拒绝运行。
- 回填：`scripts/roofline.py --bw 390e9 --rho 0.03125 --r 32`，输出 `reports/g0_roofline_mac_bw390.{md,json}`；报告由 `scripts/g0_report.py` 从探针 JSON 生成。
- 原始日志：`runs/g0-setup.log`、`runs/g0-probe-stdout.log`、`runs/g0-mac-*/g0_probe.json`（本地，已复制精简版到 `reports/g0_probe_mac_m4max.json`）。

## 观察与检查

完整数字见 `reports/g0_mac_m4max.md`。对照预测：

| 项 | 预测 | 实测 | 判定 |
|---|---|---|---|
| MPS 可用、依赖可装 | 是 | 是 | 成立 |
| `recommended_max_memory` | 90 到 105 GB | 115.4 GB，物理内存的 84% | 偏低，方向对 |
| fp32 / fp16 / bf16 / fp64 | 前三支持，fp64 报错 | 同 | 成立 |
| top-k、index_select、mask softmax | 一致 | 下标全同、逐位相同、误差 2e-7 | 成立 |
| SDPA `enable_gqa=True` | 不支持 | **支持**，误差 4e-7 | **预测错误** |
| SDPA 普通、bool mask、bf16 | 支持 | 支持 | 成立 |
| 索引矩阵 KL 反向 | 误差 ≤ 1e-4 | 2e-7，损失与 CPU 相同 | 成立 |
| `torch.compile` 逐元素 | 五成，20 到 90 s | 可用，首次 1.0 到 3.7 s | 可用分支成立，时间远低于预测 |
| copy ≥ 1 GiB | 300 到 450 GB/s | 400 到 413 | 成立 |
| sum 只读 ≥ 1 GiB | 250 到 450 GB/s | 440 到 456 | 成立，上端 |
| 注意力扫描 36 层轮换 | 150 到 350 GB/s | 382 到 419（三次运行） | **高于预测** |
| 热读与轮换接近 | 是 | 405 到 423 对 382 到 419 | 成立 |
| 每小算子 | 15 到 60 µs | matmul 23 到 29，top-k 60 到 78，index_select 5 到 6 | 成立，top-k 在高端 |
| synchronize | 50 到 200 µs | 0.06 µs（空队列） | 测法不同，不可比 |

预测之外的发现，也是本关最重要的结果：**索引路径的成本由原语选择决定，而不是由字节数决定。**

| N | `torch.gather` 展开下标 | `index_select` 拉平 | 打分 zᵀ 视图 | 打分 z 预转置 | 最优链式每层 |
|---:|---:|---:|---:|---:|---:|
| 8192 | 187 µs | 16 µs | 59 µs | 45 µs | 97 µs |
| 32768 | 673 µs | 27 µs | 137 µs | 58 µs | 166 µs |

同一组 K/V、同一份名单，gather 的两种写法相差 25 倍；索引键字节只有 K 的四分之一，但 zᵀ 视图打分的时间与扫完整层 K 的 172 µs 相当。

## 支持的解释与替代解释

**支持的解释。** 单行查询乘整层 K 的 matmul 在 MPS 上接近流式读取，所以注意力扫描达到标称带宽的七成以上。`torch.gather` 配 stride 为 0 的展开 int64 下标走通用内核并可能物化下标张量，`index_select` 走专用行拷贝内核。细长矩阵乘（内维 32）受内核效率而非带宽限制，预转置让读取连续后改善一半。

**替代解释，未排除。** gather 的慢可能与下标 dtype（int64 对 int32）或 expand 的 stride 有关，而不是 gather 本身；换 torch 版本结果可能变化。三次运行的扫描带宽在 382 到 419 之间波动，单次数字应按 ±5% 读。`enable_gqa` 在 MPS 上"支持"只证明功能正确，内核是否内部 repeat KV 要靶向剖析。

**不能推出什么。** 这些都是 MPS 上的原语成本，不是 CUDA 结论；没有模型、没有任务、没有质量结论；密集注意力整层的实际时间还没有测。

## 决策与边界

**continue。** 能力关通过：所需运算在 MPS 上正确执行，反向可用，compile 可用，内存余量充足。

G1a 的实现约束由本关直接给出：索引键按 `[G, r, N]` 连续存放；按名单取 K/V 用拉平后的 `index_select`；top-k 只对候选域做一次。不满足这三条的实现不进入正式计时。

G1a 的预测（见报告第 6 节）：eager 下 8K 持平偏负（4B 净值约 −0.9 ms），16K 持平偏正（+1.1 ms），32K 正收益（+4.5 ms，约占理想步长的两成）；0.6B 的相对收益最大。朴素 gather 写法下所有组合都会显示 D 更慢，那是原语问题而不是架构问题。

**更正（2026-10-17，PR #1 评审）。** 上段的净值用了“S1 节省 − 最优链式”，重复扣除了索引键读与入选 KV 读的成本。按“A 完整 KV 理想读取 − 最优链式”重算，4B 为 8K −0.4、16K +2.1、32K +6.4 ms；方向不变，D 更有利。G1a 实测的 D_full/A 为 1.78 / 1.10 / 0.74，8K 与 16K 比更正前后的预测都更不利：A 在 8K 只用 110 µs 读完 32 MiB，有效读取超过 300 GB/s，密集内核在短上下文的实际时间低于带宽模型给它的理想值，而 D 的链式固定底不随 N 缩小。两版预测都没有捕捉到这一点。

边界：本机 MPS 结果不推 CUDA；回填表仍是预测，密集注意力内核的实际时间由 G1a 测。

## 下一步与成本

G1a：在 `src/attention_lab/` 实现 dense_reference、dense_mask_reference、gather_attention、grouped_indexer，先过正确性契约，再做单层 A 与 D 的微基准（8K / 16K / 32K，ρ = 1/32 与 1/8），A 必须含 V 读、softmax 与 PV。预计 1 到 2 个工作日，GPU 时间以分钟计，现金 0。CUDA 轨道是否启动另行决定。

本关实际消耗：约 13 分钟设备活跃时间（含依赖下载），现金 0。
