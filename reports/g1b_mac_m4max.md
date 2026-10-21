# G1b 实测报告 · Mac M4 Max：把索引注意力装进 Qwen3-0.6B，整模型一步的 A 与 D

本报告加载真实 Qwen3-0.6B 权重，输入 token id，经过全部 28 层，最后取得下一 token 的词表分数，项目编号为 G1b。每层仍执行原 Q/K/V 投影、归一化、RoPE、输出投影、残差和 MLP；A 读取全部合法历史，未训练的 D 用索引选位后读取原 K/V。D 保留首 4/近 64，测速的额外名额由请求第一步长度乘 1/32 确定。

先用相同 token 比较原库、封装 A 和全选 D 的词表分数及缓存，检查接入是否改变本应相同的计算；逐 token 与一次输入整段的差别另作参照。随后才做性能测量。

测速先生成一串固定随机 token，各路径用完整注意力处理前 N0 个、实际计算各层缓存，D 同时建立索引表。固定长度单步每次退回这份前缀，再输入同一个新 token；连续测试则依次输入预存的后续 128 个 token，缓存逐步增长。模型每步的预测不决定下一次输入。这些是真实模型前向，输入却没有问答目标，因此没有任务分数。

四条路径分别是原库＋DynamicCache（每步拼接历史）、原库＋StaticCache（预分配并写入）、本项目预分配缓存＋完整读取 A、同类缓存加索引表＋少读 D。它们分别维护缓存，使用相同主体权重。单步计时从 token id 到词表分数，前缀处理另计。具体过程见[第二课](02_qwen_integration_and_decode.html)。

<details><summary>运行版本与报告来源</summary>

由 `scripts/g1b_report.py` 从 1 个数值 run（g1b-numerics-20261017T085459Z-formal4） 与 1 个测速 run（g1b-mac-20261016T202847Z-formal）生成。**随机 token 只驱动执行；没有任务、没有质量结论；MPS 结果不推 CUDA。** 预测见 `reports/cards/G1b_mac_m4max.md`。

</details>

## 1. 环境、模型与设置

| 项 | 值 |
|---|---|
| 机器 | Apple M4 Max，40 核 GPU，128 GiB，macOS 27.0.1 |
| torch / transformers / Python | 2.14.1 / 5.20.0 / 3.12.10；MPS；fallback 与内存保护环境变量未设置 |
| 模型 | Qwen/Qwen3-0.6B @ `c1899de289a04d12100db370d81485cdf75e47ca`（权重与 tokenizer 快照同一 commit：True）；Qwen3ForCausalLM，attn `sdpa` |
| 形状（从 config 读取并与参数形状核对） | 28 层，16 查询头 / 8 KV 头，head_dim 128，hidden 1024，词表 151936，参数 596.0 M（独立存储 596.0 M） |
| 权重文件 | model.safetensors sha256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`，1503300328 bytes |
| 代码 | 正式测速 commit c95e097866c5cd8f04896d1e5ab5020ee83be29b，dirty diff sha256 None；源码哈希见 manifest；跑分后对 `qwen_wrap.py` 的改动（显式阶段状态）记录在 `reports/provenance/g1b/qwen_wrap_post_run_edit.diff` |
| 数值 run | g1b-numerics-20261017T085459Z-formal4，commit 44c6622569a1d9c42d205560e12f909ae2f34342，dirty diff sha256 None，HF_HUB_OFFLINE=1 |
| 测速 | dtype bfloat16，batch 1，N0 [8192, 16384, 32256]，128 步，ρ 0.03125，r 32，容量 N0+128，Prefill 块 2048，单步预热 10 / 循环 50（回文顺序），序列 3 次；每步前后 `torch.mps.synchronize()`；`logits_to_keep=1` |

## 2. 数值对照（三段 + 索引接入契约）

判定：**通过**（98 项门控检查）；只按运行前的绝对预设则为 **未通过**（17 项 bf16 检查未过预设）。float32 的判定全部按预设；bf16 按卡片登记的修订判据：封装偏差 ≤ 2 × 未改模型自身逐 token 路径对自身一次前向的偏差（噪声底）。容差来自 `configs/mac_gate1b.json`。

| dtype | device | L | 检查 | max abs | 相对范数 | argmax 一致 | 预设 atol | 预设 | 噪声底 max abs | 修订 |
|---|---|---:|---|---:|---:|---:|---:|---|---:|---|
| float32 | mps | 32 | wrapped_one_shot_vs_unmodified_logits | 0.000e+00 | 0.000e+00 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 32 | wrapped_one_shot_cache_vs_hf_cache | 0.000e+00 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 32 | wrapped_incremental_vs_unmodified_logits | 6.795e-05 | 3.716e-06 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 32 | wrapped_incremental_last_tokens_vs_unmodified_logits | 3.552e-05 | 3.921e-06 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 32 | wrapped_incremental_cache_vs_hf_cache | 6.409e-04 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 32 | wrapped_incremental_vs_one_shot_logits | 6.795e-05 | 3.716e-06 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 32 | wrapped_incremental_vs_unmodified_incremental_logits | 0.000e+00 | 0.000e+00 | 1.000 | 0.01 | ✓ 逐位相同 | — | — |
| float32 | mps | 32 | wrapped_incremental_cache_vs_unmodified_incremental_cache | 0.000e+00 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 32 | rollback_redecode_vs_first_pass | 0.000e+00 | 0.000e+00 | 1.000 | 0.01 | ✓ 逐位相同 | — | — |
| float32 | mps | 32 | D_full_selection_vs_wrapped_incremental_logits | 0.000e+00 | 0.000e+00 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 32 | D_full_selection_vs_unmodified_logits | 3.552e-05 | 3.921e-06 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 32 | D_full_selection_cache_vs_hf_cache | 6.409e-04 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 32 | chunked_build_plus_steps_vs_one_shot_build_kv_and_index | 6.409e-04 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 32 | chunked_prefill_vs_one_shot_build_kv_and_index | 0.000e+00 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 32 | D_rho_1_8_vs_unmodified_logits_information_only | 3.552e-05 | 3.921e-06 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 128 | wrapped_one_shot_vs_unmodified_logits | 0.000e+00 | 0.000e+00 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 128 | wrapped_one_shot_cache_vs_hf_cache | 0.000e+00 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 128 | wrapped_incremental_vs_unmodified_logits | 4.864e-05 | 1.645e-06 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 128 | wrapped_incremental_last_tokens_vs_unmodified_logits | 3.862e-05 | 3.808e-06 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 128 | wrapped_incremental_cache_vs_hf_cache | 4.883e-04 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 128 | wrapped_incremental_vs_one_shot_logits | 4.864e-05 | 1.645e-06 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 128 | wrapped_incremental_vs_unmodified_incremental_logits | 0.000e+00 | 0.000e+00 | 1.000 | 0.01 | ✓ 逐位相同 | — | — |
| float32 | mps | 128 | wrapped_incremental_cache_vs_unmodified_incremental_cache | 0.000e+00 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 128 | rollback_redecode_vs_first_pass | 0.000e+00 | 0.000e+00 | 1.000 | 0.01 | ✓ 逐位相同 | — | — |
| float32 | mps | 128 | D_full_selection_vs_wrapped_incremental_logits | 1.049e-05 | 8.778e-07 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 128 | D_full_selection_vs_unmodified_logits | 3.791e-05 | 3.845e-06 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 128 | D_full_selection_cache_vs_hf_cache | 4.883e-04 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 128 | chunked_build_plus_steps_vs_one_shot_build_kv_and_index | 4.883e-04 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 128 | chunked_prefill_vs_one_shot_build_kv_and_index | 2.289e-04 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 128 | D_rho_1_8_vs_unmodified_logits_information_only | 3.755e+00 | 2.069e-01 | 0.500 | 0.01 | 记录 | — | — |
| float32 | mps | 512 | wrapped_one_shot_vs_unmodified_logits | 0.000e+00 | 0.000e+00 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 512 | wrapped_one_shot_cache_vs_hf_cache | 0.000e+00 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 512 | wrapped_incremental_vs_unmodified_logits | 3.296e-05 | 4.722e-07 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 512 | wrapped_incremental_last_tokens_vs_unmodified_logits | 3.004e-05 | 3.076e-06 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 512 | wrapped_incremental_cache_vs_hf_cache | 4.578e-04 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 512 | wrapped_incremental_vs_one_shot_logits | 3.296e-05 | 4.722e-07 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 512 | wrapped_incremental_vs_unmodified_incremental_logits | 0.000e+00 | 0.000e+00 | 1.000 | 0.01 | ✓ 逐位相同 | — | — |
| float32 | mps | 512 | wrapped_incremental_cache_vs_unmodified_incremental_cache | 0.000e+00 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 512 | rollback_redecode_vs_first_pass | 0.000e+00 | 0.000e+00 | 1.000 | 0.01 | ✓ 逐位相同 | — | — |
| float32 | mps | 512 | D_full_selection_vs_wrapped_incremental_logits | 5.722e-06 | 4.949e-07 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 512 | D_full_selection_vs_unmodified_logits | 2.933e-05 | 3.108e-06 | 1.000 | 0.01 | ✓ | — | — |
| float32 | mps | 512 | D_full_selection_cache_vs_hf_cache | 4.578e-04 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 512 | chunked_build_plus_steps_vs_one_shot_build_kv_and_index | 4.578e-04 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 512 | chunked_prefill_vs_one_shot_build_kv_and_index | 4.578e-04 | — | — | 0.001 | ✓ | — | — |
| float32 | mps | 512 | D_rho_1_8_vs_unmodified_logits_information_only | 3.205e+00 | 2.458e-01 | 0.750 | 0.01 | 记录 | — | — |
| bfloat16 | mps | 32 | wrapped_one_shot_vs_unmodified_logits | 0.000e+00 | 0.000e+00 | 1.000 | 1.0 | ✓ | 3.05e-01 | ✓ |
| bfloat16 | mps | 32 | wrapped_one_shot_cache_vs_hf_cache | 0.000e+00 | — | — | 0.25 | ✓ | 2.00e+00 | ✓ |
| bfloat16 | mps | 32 | wrapped_incremental_vs_unmodified_logits | 3.047e-01 | 2.045e-02 | 0.969 | 1.0 | ✗ | 3.05e-01 | ✓ |
| bfloat16 | mps | 32 | wrapped_incremental_last_tokens_vs_unmodified_logits | 2.500e-01 | 2.096e-02 | 1.000 | 1.0 | ✗ | 2.50e-01 | ✓ |
| bfloat16 | mps | 32 | wrapped_incremental_cache_vs_hf_cache | 2.000e+00 | — | — | 0.25 | ✗ | 2.00e+00 | ✓ |
| bfloat16 | mps | 32 | wrapped_incremental_vs_one_shot_logits | 3.047e-01 | 2.045e-02 | 0.969 | 1.0 | ✗ | 3.05e-01 | ✓ |
| bfloat16 | mps | 32 | wrapped_incremental_vs_unmodified_incremental_logits | 0.000e+00 | 0.000e+00 | 1.000 | 1.0 | ✓ 逐位相同 | 3.05e-01 | ✓ |
| bfloat16 | mps | 32 | wrapped_incremental_cache_vs_unmodified_incremental_cache | 0.000e+00 | — | — | 0.25 | ✓ | 2.00e+00 | ✓ |
| bfloat16 | mps | 32 | rollback_redecode_vs_first_pass | 0.000e+00 | 0.000e+00 | 1.000 | 1.0 | ✓ 逐位相同 | 2.50e-01 | ✓ |
| bfloat16 | mps | 32 | D_full_selection_vs_wrapped_incremental_logits | 0.000e+00 | 0.000e+00 | 1.000 | 1.0 | ✓ | 2.50e-01 | ✓ |
| bfloat16 | mps | 32 | D_full_selection_vs_unmodified_logits | 2.500e-01 | 2.096e-02 | 1.000 | 1.0 | ✗ | 2.50e-01 | ✓ |
| bfloat16 | mps | 32 | D_full_selection_cache_vs_hf_cache | 2.000e+00 | — | — | 0.25 | ✗ | 2.00e+00 | ✓ |
| bfloat16 | mps | 32 | chunked_build_plus_steps_vs_one_shot_build_kv_and_index | 2.000e+00 | — | — | 0.25 | ✗ | 2.00e+00 | ✓ |
| bfloat16 | mps | 32 | chunked_prefill_vs_one_shot_build_kv_and_index | 0.000e+00 | — | — | 0.25 | ✓ | 2.00e+00 | ✓ |
| bfloat16 | mps | 32 | D_rho_1_8_vs_unmodified_logits_information_only | 2.500e-01 | 2.096e-02 | 1.000 | 1.0 | 记录 | — | — |
| bfloat16 | mps | 128 | wrapped_one_shot_vs_unmodified_logits | 0.000e+00 | 0.000e+00 | 1.000 | 1.0 | ✓ | 2.68e-01 | ✓ |
| bfloat16 | mps | 128 | wrapped_one_shot_cache_vs_hf_cache | 0.000e+00 | — | — | 0.25 | ✓ | 1.00e+00 | ✓ |
| bfloat16 | mps | 128 | wrapped_incremental_vs_unmodified_logits | 2.679e-01 | 1.027e-02 | 0.992 | 1.0 | ✓ | 2.68e-01 | ✓ |
| bfloat16 | mps | 128 | wrapped_incremental_last_tokens_vs_unmodified_logits | 2.578e-01 | 2.407e-02 | 1.000 | 1.0 | ✗ | 2.58e-01 | ✓ |
| bfloat16 | mps | 128 | wrapped_incremental_cache_vs_hf_cache | 1.000e+00 | — | — | 0.25 | ✗ | 1.00e+00 | ✓ |
| bfloat16 | mps | 128 | wrapped_incremental_vs_one_shot_logits | 2.679e-01 | 1.027e-02 | 0.992 | 1.0 | ✓ | 2.68e-01 | ✓ |
| bfloat16 | mps | 128 | wrapped_incremental_vs_unmodified_incremental_logits | 0.000e+00 | 0.000e+00 | 1.000 | 1.0 | ✓ 逐位相同 | 2.68e-01 | ✓ |
| bfloat16 | mps | 128 | wrapped_incremental_cache_vs_unmodified_incremental_cache | 0.000e+00 | — | — | 0.25 | ✓ | 1.00e+00 | ✓ |
| bfloat16 | mps | 128 | rollback_redecode_vs_first_pass | 0.000e+00 | 0.000e+00 | 1.000 | 1.0 | ✓ 逐位相同 | 2.58e-01 | ✓ |
| bfloat16 | mps | 128 | D_full_selection_vs_wrapped_incremental_logits | 2.188e-01 | 2.150e-02 | 1.000 | 1.0 | ✗ | 2.58e-01 | ✓ |
| bfloat16 | mps | 128 | D_full_selection_vs_unmodified_logits | 2.188e-01 | 1.825e-02 | 1.000 | 1.0 | ✓ | 2.58e-01 | ✓ |
| bfloat16 | mps | 128 | D_full_selection_cache_vs_hf_cache | 1.500e+00 | — | — | 0.25 | ✗ | 1.00e+00 | ✓ |
| bfloat16 | mps | 128 | chunked_build_plus_steps_vs_one_shot_build_kv_and_index | 1.500e+00 | — | — | 0.25 | ✗ | 1.00e+00 | ✓ |
| bfloat16 | mps | 128 | chunked_prefill_vs_one_shot_build_kv_and_index | 1.000e+00 | — | — | 0.25 | ✗ | 1.00e+00 | ✓ |
| bfloat16 | mps | 128 | D_rho_1_8_vs_unmodified_logits_information_only | 3.797e+00 | 2.107e-01 | 0.500 | 1.0 | 记录 | — | — |
| bfloat16 | mps | 512 | wrapped_one_shot_vs_unmodified_logits | 0.000e+00 | 0.000e+00 | 1.000 | 1.0 | ✓ | 1.56e-01 | ✓ |
| bfloat16 | mps | 512 | wrapped_one_shot_cache_vs_hf_cache | 0.000e+00 | — | — | 0.25 | ✓ | 1.50e+00 | ✓ |
| bfloat16 | mps | 512 | wrapped_incremental_vs_unmodified_logits | 1.562e-01 | 2.276e-03 | 1.000 | 1.0 | ✓ | 1.56e-01 | ✓ |
| bfloat16 | mps | 512 | wrapped_incremental_last_tokens_vs_unmodified_logits | 1.562e-01 | 1.519e-02 | 1.000 | 1.0 | ✓ | 1.56e-01 | ✓ |
| bfloat16 | mps | 512 | wrapped_incremental_cache_vs_hf_cache | 1.500e+00 | — | — | 0.25 | ✗ | 1.50e+00 | ✓ |
| bfloat16 | mps | 512 | wrapped_incremental_vs_one_shot_logits | 1.562e-01 | 2.276e-03 | 1.000 | 1.0 | ✓ | 1.56e-01 | ✓ |
| bfloat16 | mps | 512 | wrapped_incremental_vs_unmodified_incremental_logits | 0.000e+00 | 0.000e+00 | 1.000 | 1.0 | ✓ 逐位相同 | 1.56e-01 | ✓ |
| bfloat16 | mps | 512 | wrapped_incremental_cache_vs_unmodified_incremental_cache | 0.000e+00 | — | — | 0.25 | ✓ | 1.50e+00 | ✓ |
| bfloat16 | mps | 512 | rollback_redecode_vs_first_pass | 0.000e+00 | 0.000e+00 | 1.000 | 1.0 | ✓ 逐位相同 | 1.56e-01 | ✓ |
| bfloat16 | mps | 512 | D_full_selection_vs_wrapped_incremental_logits | 1.094e-01 | 1.289e-02 | 1.000 | 1.0 | ✓ | 1.56e-01 | ✓ |
| bfloat16 | mps | 512 | D_full_selection_vs_unmodified_logits | 1.328e-01 | 1.379e-02 | 1.000 | 1.0 | ✓ | 1.56e-01 | ✓ |
| bfloat16 | mps | 512 | D_full_selection_cache_vs_hf_cache | 1.500e+00 | — | — | 0.25 | ✗ | 1.50e+00 | ✓ |
| bfloat16 | mps | 512 | chunked_build_plus_steps_vs_one_shot_build_kv_and_index | 1.500e+00 | — | — | 0.25 | ✗ | 1.50e+00 | ✓ |
| bfloat16 | mps | 512 | chunked_prefill_vs_one_shot_build_kv_and_index | 1.000e+00 | — | — | 0.25 | ✗ | 1.50e+00 | ✓ |
| bfloat16 | mps | 512 | D_rho_1_8_vs_unmodified_logits_information_only | 3.250e+00 | 2.486e-01 | 0.750 | 1.0 | 记录 | — | — |
| float32 | cpu | 128 | wrapped_one_shot_vs_unmodified_logits | 0.000e+00 | 0.000e+00 | 1.000 | 0.01 | ✓ | — | — |
| float32 | cpu | 128 | wrapped_one_shot_cache_vs_hf_cache | 0.000e+00 | — | — | 0.001 | ✓ | — | — |
| float32 | cpu | 128 | wrapped_incremental_vs_unmodified_logits | 3.564e-05 | 8.252e-07 | 1.000 | 0.01 | ✓ | — | — |
| float32 | cpu | 128 | wrapped_incremental_last_tokens_vs_unmodified_logits | 3.564e-05 | 3.294e-06 | 1.000 | 0.01 | ✓ | — | — |
| float32 | cpu | 128 | wrapped_incremental_cache_vs_hf_cache | 5.188e-04 | — | — | 0.001 | ✓ | — | — |
| float32 | cpu | 128 | wrapped_incremental_vs_one_shot_logits | 3.564e-05 | 8.252e-07 | 1.000 | 0.01 | ✓ | — | — |
| float32 | cpu | 128 | wrapped_incremental_vs_unmodified_incremental_logits | 1.192e-05 | 2.392e-07 | 1.000 | 0.01 | ✓ | — | — |
| float32 | cpu | 128 | wrapped_incremental_cache_vs_unmodified_incremental_cache | 4.578e-05 | — | — | 0.001 | ✓ | — | — |
| float32 | cpu | 128 | rollback_redecode_vs_first_pass | 0.000e+00 | 0.000e+00 | 1.000 | 0.01 | ✓ 逐位相同 | — | — |
| float32 | cpu | 128 | D_full_selection_vs_wrapped_incremental_logits | 9.537e-06 | 7.162e-07 | 1.000 | 0.01 | ✓ | — | — |
| float32 | cpu | 128 | D_full_selection_vs_unmodified_logits | 3.624e-05 | 3.293e-06 | 1.000 | 0.01 | ✓ | — | — |
| float32 | cpu | 128 | D_full_selection_cache_vs_hf_cache | 5.188e-04 | — | — | 0.001 | ✓ | — | — |
| float32 | cpu | 128 | chunked_build_plus_steps_vs_one_shot_build_kv_and_index | 5.188e-04 | — | — | 0.001 | ✓ | — | — |
| float32 | cpu | 128 | chunked_prefill_vs_one_shot_build_kv_and_index | 0.000e+00 | — | — | 0.001 | ✓ | — | — |
| float32 | cpu | 128 | D_rho_1_8_vs_unmodified_logits_information_only | 3.755e+00 | 2.068e-01 | 0.500 | 0.01 | 记录 | — | — |

噪声底（未改模型自身：DynamicCache 分块后逐 token，对自身一次前向）：
- float32 mps L=32：词表分数 max abs 6.795e-05，相对范数 3.716e-06，argmax 一致 1.000；缓存 K 6.409e-04，V 2.022e-04
- float32 mps L=128：词表分数 max abs 4.864e-05，相对范数 1.645e-06，argmax 一致 1.000；缓存 K 4.883e-04，V 1.574e-04
- float32 mps L=512：词表分数 max abs 3.296e-05，相对范数 4.722e-07，argmax 一致 1.000；缓存 K 4.578e-04，V 1.564e-04
- bfloat16 mps L=32：词表分数 max abs 3.047e-01，相对范数 2.045e-02，argmax 一致 0.969；缓存 K 2.000e+00，V 1.062e+00
- bfloat16 mps L=128：词表分数 max abs 2.679e-01，相对范数 1.027e-02，argmax 一致 0.992；缓存 K 1.000e+00，V 1.000e+00
- bfloat16 mps L=512：词表分数 max abs 1.562e-01，相对范数 2.276e-03，argmax 一致 1.000；缓存 K 1.500e+00，V 7.500e-01
- float32 cpu L=128：词表分数 max abs 3.779e-05，相对范数 8.616e-07，argmax 一致 1.000；缓存 K 5.188e-04，V 1.717e-04

未改模型 MPS float32 对 CPU float32（平台噪声底，不门控）：max abs 1.407e-04，相对范数 3.668e-06，argmax 一致 1.000。

短提示参照（未改模型，top-1 与贪心续写；只是数值参照，不评价质量）：

- float32 mps：top-5 [' Paris', ' located', ' the', '...', ' ____'] logits [17.4989, 14.3372, 14.1693, 13.766, 13.6347]；贪心：' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'
- bfloat16 mps：top-5 [' Paris', ' located', ' the', '...', ' ____'] logits [17.5, 14.25, 14.125, 13.8125, 13.6875]；贪心：' Paris. The capital of France is also the capital of the Republic of France.'
- float32 cpu：top-5 [' Paris', ' located', ' the', '...', ' ____'] logits [17.4989, 14.3372, 14.1693, 13.766, 13.6347]；贪心：' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'

冒烟生成（bfloat16，提示 431 token，贪心 32 新 token，D 的 k0=54）：
- native_dynamic（与 native 首次分歧步 None）：'Attention Lab studies the trade-off between reading every past position and reading an indexed subse'
- wrapped_full（与 native 首次分歧步 None）：'Attention Lab studies the trade-off between reading every past position and reading an indexed subse'
- D_rho_1_8（与 native 首次分歧步 1）：'Attention is a model that summarizes the paragraph in one sentence.\nHuman: What is the main idea of '

## 3. Prefill 与首步（每条路径一次，ms）

| run | N0 | 路径 | Prefill ms | 首步（冷）ms | k0 | 缓存字节 |
|---|---:|---|---:|---:|---:|---:|
| g1b-mac-20261016T202847Z-formal | 8192 | 原生 A（DynamicCache） | 1670.8 | 17.88 | — | 939524096 |
| g1b-mac-20261016T202847Z-formal | 8192 | 原生 A（StaticCache） | 1832.1 | 17.57 | — | 954204160 |
| g1b-mac-20261016T202847Z-formal | 8192 | 封装 A（静态 LabCache） | 1779.2 | 13.44 | — | 954204160 |
| g1b-mac-20261016T202847Z-formal | 8192 | D（索引读取，含全部新增成本） | 1744.8 | 21.04 | 256 | 1073479680 |
| g1b-mac-20261016T202847Z-formal | 16384 | 原生 A（DynamicCache） | 4671.7 | 24.65 | — | 1879048192 |
| g1b-mac-20261016T202847Z-formal | 16384 | 原生 A（StaticCache） | 6274.4 | 19.72 | — | 1893728256 |
| g1b-mac-20261016T202847Z-formal | 16384 | 封装 A（静态 LabCache） | 5112.6 | 16.94 | — | 1893728256 |
| g1b-mac-20261016T202847Z-formal | 16384 | D（索引读取，含全部新增成本） | 5172.4 | 20.79 | 512 | 2130444288 |
| g1b-mac-20261016T202847Z-formal | 32256 | 原生 A（DynamicCache） | 19849.7 | 73.53 | — | 3699376128 |
| g1b-mac-20261016T202847Z-formal | 32256 | 原生 A（StaticCache） | 32719.2 | 48.96 | — | 3714056192 |
| g1b-mac-20261016T202847Z-formal | 32256 | 封装 A（静态 LabCache） | 21420.7 | 21.79 | — | 3714056192 |
| g1b-mac-20261016T202847Z-formal | 32256 | D（索引读取，含全部新增成本） | 21096.3 | 24.92 | 1008 | 4178313216 |

## 4. 固定 N 单步：整步耗时（ms）与配对比值

配对比值 = 同一循环内该路径中位数 ÷ 封装 A 中位数（回文顺序，每循环每路径两次）。D/A ≤ 0.90 为通过线，要求每个循环都 ≤ 0.90 且剖析无全长 matmul。

| run | N0 | 模式 | 路径 | 中位 ms | p95 ms | 最小 ms | n | 配对比值中位（范围） | ≤0.90 的循环数 |
|---|---:|---|---|---:|---:|---:|---:|---|---:|
| g1b-mac-20261016T202847Z-formal | 8192 | eager | 原生 A（DynamicCache） | 24.15 | 24.74 | 23.30 | 100 | 1.81 (1.65–1.95) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 8192 | eager | 原生 A（StaticCache） | 14.19 | 14.83 | 13.38 | 100 | 1.07 (1.00–1.14) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 8192 | eager | 封装 A（静态 LabCache） | 13.32 | 14.26 | 12.58 | 100 | 1.00 (1.00–1.00) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 8192 | eager | D（索引读取，含全部新增成本） | 16.11 | 17.19 | 15.30 | 100 | 1.21 (1.14–1.28) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 8192 | compile | 原生 A（DynamicCache） | 23.74 | 24.41 | 22.41 | 100 | 1.82 (1.73–1.91) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 8192 | compile | 原生 A（StaticCache） | 14.04 | 14.72 | 12.85 | 100 | 1.07 (1.01–1.15) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 8192 | compile | 封装 A（静态 LabCache） | 13.08 | 13.67 | 12.21 | 100 | 1.00 (1.00–1.00) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 8192 | compile | D（索引读取，含全部新增成本） | 15.91 | 16.56 | 14.98 | 100 | 1.22 (1.13–1.30) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 16384 | eager | 原生 A（DynamicCache） | 37.49 | 37.91 | 36.03 | 100 | 2.34 (2.28–2.43) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 16384 | eager | 原生 A（StaticCache） | 16.74 | 17.50 | 15.98 | 100 | 1.05 (1.00–1.12) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 16384 | eager | 封装 A（静态 LabCache） | 16.00 | 16.62 | 15.13 | 100 | 1.00 (1.00–1.00) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 16384 | eager | D（索引读取，含全部新增成本） | 16.84 | 17.61 | 15.95 | 100 | 1.05 (1.01–1.12) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 16384 | compile | 原生 A（DynamicCache） | 36.74 | 37.12 | 35.56 | 100 | 2.43 (2.23–2.56) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 16384 | compile | 原生 A（StaticCache） | 16.39 | 17.14 | 15.41 | 100 | 1.08 (1.00–1.14) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 16384 | compile | 封装 A（静态 LabCache） | 15.13 | 16.12 | 14.21 | 100 | 1.00 (1.00–1.00) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 16384 | compile | D（索引读取，含全部新增成本） | 16.68 | 17.53 | 15.40 | 100 | 1.10 (1.02–1.21) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 32256 | eager | 原生 A（DynamicCache） | 74.68 | 78.35 | 62.69 | 100 | 3.50 (3.02–3.75) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 32256 | eager | 原生 A（StaticCache） | 22.65 | 23.66 | 20.99 | 100 | 1.08 (1.01–1.14) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 32256 | eager | 封装 A（静态 LabCache） | 21.21 | 22.03 | 19.89 | 100 | 1.00 (1.00–1.00) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 32256 | eager | D（索引读取，含全部新增成本） | 19.68 | 20.62 | 17.95 | 100 | 0.93 (0.87–0.99) | 7/50 |
| g1b-mac-20261016T202847Z-formal | 32256 | compile | 原生 A（DynamicCache） | 73.27 | 75.58 | 61.40 | 100 | 3.59 (3.11–3.76) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 32256 | compile | 原生 A（StaticCache） | 22.07 | 23.07 | 20.65 | 100 | 1.08 (1.02–1.15) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 32256 | compile | 封装 A（静态 LabCache） | 20.40 | 21.15 | 19.19 | 100 | 1.00 (1.00–1.00) | 0/50 |
| g1b-mac-20261016T202847Z-formal | 32256 | compile | D（索引读取，含全部新增成本） | 19.06 | 20.29 | 17.33 | 100 | 0.93 (0.89–1.14) | 6/50 |

## 5. 连续 128 步：每步耗时、增长与 k 规则

| run | N0 | 模式 | 路径 | 每步中位 ms | 首 16 步中位 | 末 16 步中位 | 斜率 µs/步 | 128 步合计中位 ms | k（末步） | 读取数（末步） | 序列数 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| g1b-mac-20261016T202847Z-formal | 8192 | eager | 原生 A（DynamicCache） | 16.66 | 16.66 | 16.63 | -2.3 | 2147 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 8192 | eager | 原生 A（StaticCache） | 14.14 | 14.10 | 14.11 | -2.3 | 1813 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 8192 | eager | 封装 A（静态 LabCache） | 13.39 | 13.50 | 13.41 | -2.1 | 1716 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 8192 | eager | D（索引读取，含全部新增成本） | 16.86 | 16.62 | 16.88 | -0.1 | 2184 | 256 | 324 | 3 |
| g1b-mac-20261016T202847Z-formal | 8192 | compile | 原生 A（DynamicCache） | 16.44 | 16.41 | 16.54 | -2.8 | 2103 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 8192 | compile | 原生 A（StaticCache） | 13.86 | 13.88 | 13.87 | -1.1 | 1775 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 8192 | compile | 封装 A（静态 LabCache） | 13.00 | 13.04 | 12.93 | -3.2 | 1665 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 8192 | compile | D（索引读取，含全部新增成本） | 16.80 | 16.45 | 16.86 | 0.7 | 2192 | 256 | 324 | 3 |
| g1b-mac-20261016T202847Z-formal | 16384 | eager | 原生 A（DynamicCache） | 23.89 | 24.02 | 23.85 | -4.8 | 3082 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 16384 | eager | 原生 A（StaticCache） | 16.66 | 16.57 | 16.87 | 1.4 | 2144 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 16384 | eager | 封装 A（静态 LabCache） | 15.85 | 15.85 | 15.87 | -9.1 | 2048 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 16384 | eager | D（索引读取，含全部新增成本） | 17.89 | 17.37 | 17.62 | -4.7 | 2310 | 512 | 580 | 3 |
| g1b-mac-20261016T202847Z-formal | 16384 | compile | 原生 A（DynamicCache） | 23.60 | 23.38 | 23.49 | -8.7 | 3077 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 16384 | compile | 原生 A（StaticCache） | 16.39 | 16.51 | 16.37 | -3.8 | 2110 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 16384 | compile | 封装 A（静态 LabCache） | 15.18 | 15.34 | 15.24 | -11.6 | 1963 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 16384 | compile | D（索引读取，含全部新增成本） | 17.41 | 17.29 | 17.52 | -2.6 | 2271 | 512 | 580 | 3 |
| g1b-mac-20261016T202847Z-formal | 32256 | eager | 原生 A（DynamicCache） | 39.34 | 38.45 | 40.46 | -7.5 | 5106 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 32256 | eager | 原生 A（StaticCache） | 22.13 | 24.00 | 21.83 | -31.3 | 2921 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 32256 | eager | 封装 A（静态 LabCache） | 20.83 | 20.88 | 20.76 | -9.0 | 2709 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 32256 | eager | D（索引读取，含全部新增成本） | 19.05 | 19.84 | 19.30 | -30.3 | 2529 | 1008 | 1076 | 3 |
| g1b-mac-20261016T202847Z-formal | 32256 | compile | 原生 A（DynamicCache） | 38.26 | 37.23 | 39.55 | 1.7 | 4967 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 32256 | compile | 原生 A（StaticCache） | 21.91 | 22.59 | 21.80 | -24.4 | 2902 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 32256 | compile | 封装 A（静态 LabCache） | 20.25 | 20.33 | 20.27 | -11.8 | 2619 | — | — | 3 |
| g1b-mac-20261016T202847Z-formal | 32256 | compile | D（索引读取，含全部新增成本） | 18.77 | 18.96 | 18.98 | -27.3 | 2494 | 1008 | 1076 | 3 |

## 6. 内存与剖析

- g1b-mac-20261016T202847Z-formal N0=8192：加载后 1.20 GB，Prefill 后各路径 native_dynamic: 4.24 GB, native_static: 5.21 GB, wrapped_full: 5.21 GB, sparse: 5.21 GB；eager 测完 5.21 GB（driver 6.64 GB）；进程 ru_maxrss 1.13 GB
  - 剖析 native_dynamic：aten 算子 5260，触及长度的 matmul 类算子 0（无全长 matmul），top-k 0，index_select 1，cat 113，SDPA 调度 ['aten::_scaled_dot_product_attention_math_for_mps', 'aten::scaled_dot_product_attention']，触及长度的算子 ['aten::as_strided', 'aten::slice']
  - 剖析 native_static：aten 算子 5838，触及长度的 matmul 类算子 56（有全长 matmul），top-k 0，index_select 1，cat 57，SDPA 调度 ['aten::_scaled_dot_product_attention_math_for_mps', 'aten::scaled_dot_product_attention']，触及长度的算子 ['aten::_scaled_dot_product_attention_math_for_mps', 'aten::add', 'aten::as_strided', 'aten::expand', 'aten::index_copy_', 'aten::le', 'aten::scaled_dot_product_attention', 'aten::slice', 'aten::unsqueeze', 'aten::where']
  - 剖析 wrapped_full：aten 算子 5736，触及长度的 matmul 类算子 0（无全长 matmul），top-k 0，index_select 1，cat 57，SDPA 调度 ['aten::_scaled_dot_product_attention_math_for_mps', 'aten::scaled_dot_product_attention']，触及长度的算子 ['aten::as_strided', 'aten::slice']
  - 剖析 sparse：aten 算子 7614，触及长度的 matmul 类算子 0（无全长 matmul），top-k 28，index_select 57，cat 86，SDPA 调度 ['aten::_scaled_dot_product_attention_math_for_mps', 'aten::scaled_dot_product_attention']，触及长度的算子 ['aten::as_strided', 'aten::slice', 'aten::view']
  - 块编译（MLP + RMSNorm，四路径共用）：ok，首调用 0.3 s，与 eager 词表分数 max abs 差 0.111328125
    dynamo 计数变化：{"frames": {"total": 0, "ok": 0}, "stats": {"calls_captured": 0, "unique_graphs": 0}, "aot_autograd": {"total": 0, "autograd_cache_hit": 0, "ok": 0}, "inductor": {"fxgraph_cache_hit": 0, "extern_calls": 0}}
- g1b-mac-20261016T202847Z-formal N0=16384：加载后 1.20 GB，Prefill 后各路径 native_dynamic: 8.37 GB, native_static: 10.31 GB, wrapped_full: 9.21 GB, sparse: 9.21 GB；eager 测完 9.21 GB（driver 16.31 GB）；进程 ru_maxrss 1.13 GB
  - 剖析 native_dynamic：aten 算子 5260，触及长度的 matmul 类算子 0（无全长 matmul），top-k 0，index_select 1，cat 113，SDPA 调度 ['aten::_scaled_dot_product_attention_math_for_mps', 'aten::scaled_dot_product_attention']，触及长度的算子 ['aten::as_strided', 'aten::slice']
  - 剖析 native_static：aten 算子 5838，触及长度的 matmul 类算子 56（有全长 matmul），top-k 0，index_select 1，cat 57，SDPA 调度 ['aten::_scaled_dot_product_attention_math_for_mps', 'aten::scaled_dot_product_attention']，触及长度的算子 ['aten::_scaled_dot_product_attention_math_for_mps', 'aten::add', 'aten::as_strided', 'aten::expand', 'aten::index_copy_', 'aten::le', 'aten::scaled_dot_product_attention', 'aten::slice', 'aten::unsqueeze', 'aten::where']
  - 剖析 wrapped_full：aten 算子 5736，触及长度的 matmul 类算子 0（无全长 matmul），top-k 0，index_select 1，cat 57，SDPA 调度 ['aten::_scaled_dot_product_attention_math_for_mps', 'aten::scaled_dot_product_attention']，触及长度的算子 ['aten::as_strided', 'aten::slice']
  - 剖析 sparse：aten 算子 7614，触及长度的 matmul 类算子 0（无全长 matmul），top-k 28，index_select 57，cat 86，SDPA 调度 ['aten::_scaled_dot_product_attention_math_for_mps', 'aten::scaled_dot_product_attention']，触及长度的算子 ['aten::as_strided', 'aten::slice', 'aten::view']
  - 块编译（MLP + RMSNorm，四路径共用）：ok，首调用 0.1 s，与 eager 词表分数 max abs 差 0.107666015625
    dynamo 计数变化：{"frames": {"total": 0, "ok": 0}, "stats": {"calls_captured": 0, "unique_graphs": 0}, "aot_autograd": {"total": 0, "autograd_cache_hit": 0, "ok": 0}, "inductor": {"fxgraph_cache_hit": 0, "extern_calls": 0}}
- g1b-mac-20261016T202847Z-formal N0=32256：加载后 1.20 GB，Prefill 后各路径 native_dynamic: 15.15 GB, native_static: 18.91 GB, wrapped_full: 16.72 GB, sparse: 16.72 GB；eager 测完 16.72 GB（driver 24.91 GB）；进程 ru_maxrss 1.13 GB
  - 剖析 native_dynamic：aten 算子 5260，触及长度的 matmul 类算子 0（无全长 matmul），top-k 0，index_select 1，cat 113，SDPA 调度 ['aten::_scaled_dot_product_attention_math_for_mps', 'aten::scaled_dot_product_attention']，触及长度的算子 ['aten::as_strided', 'aten::slice']
  - 剖析 native_static：aten 算子 5838，触及长度的 matmul 类算子 56（有全长 matmul），top-k 0，index_select 1，cat 57，SDPA 调度 ['aten::_scaled_dot_product_attention_math_for_mps', 'aten::scaled_dot_product_attention']，触及长度的算子 ['aten::_scaled_dot_product_attention_math_for_mps', 'aten::add', 'aten::as_strided', 'aten::expand', 'aten::index_copy_', 'aten::le', 'aten::scaled_dot_product_attention', 'aten::slice', 'aten::unsqueeze', 'aten::where']
  - 剖析 wrapped_full：aten 算子 5736，触及长度的 matmul 类算子 0（无全长 matmul），top-k 0，index_select 1，cat 57，SDPA 调度 ['aten::_scaled_dot_product_attention_math_for_mps', 'aten::scaled_dot_product_attention']，触及长度的算子 ['aten::as_strided', 'aten::slice']
  - 剖析 sparse：aten 算子 7670，触及长度的 matmul 类算子 0（无全长 matmul），top-k 28，index_select 57，cat 86，SDPA 调度 ['aten::_scaled_dot_product_attention_math_for_mps', 'aten::scaled_dot_product_attention']，触及长度的算子 ['aten::as_strided', 'aten::slice', 'aten::view']
  - 块编译（MLP + RMSNorm，四路径共用）：ok，首调用 0.1 s，与 eager 词表分数 max abs 差 0.1875
    dynamo 计数变化：{"frames": {"total": 0, "ok": 0}, "stats": {"calls_captured": 0, "unique_graphs": 0}, "aot_autograd": {"total": 0, "autograd_cache_hit": 0, "ok": 0}, "inductor": {"fxgraph_cache_hit": 0, "extern_calls": 0}}

## 7. 判定

| run | N0 | 模式 | D/A 配对中位 | 范围 | ≤0.90 循环 | D 无全长 matmul | 性能通过 |
|---|---:|---|---:|---|---:|---|---|
| g1b-mac-20261016T202847Z-formal | 8192 | eager | 1.21 | 1.14–1.28 | 0/50 | True | 不通过 |
| g1b-mac-20261016T202847Z-formal | 8192 | compile | 1.22 | 1.13–1.30 | 0/50 | True | 不通过 |
| g1b-mac-20261016T202847Z-formal | 16384 | eager | 1.05 | 1.01–1.12 | 0/50 | True | 不通过 |
| g1b-mac-20261016T202847Z-formal | 16384 | compile | 1.10 | 1.02–1.21 | 0/50 | True | 不通过 |
| g1b-mac-20261016T202847Z-formal | 32256 | eager | 0.93 | 0.87–0.99 | 7/50 | True | 不通过 |
| g1b-mac-20261016T202847Z-formal | 32256 | compile | 0.93 | 0.89–1.14 | 6/50 | True | 不通过 |

性能判定：没有设置在同模型、同设备、同工作量下呈现每个循环都 ≤ 0.90 的整步收益。工程完成判定见数值一节与卡片。


## 附录：显式阶段状态修复后的最小对照（不替换正式表）

PR #2 评审指出封装用 `T == 1` 推断解码阶段；修复为显式的 `phase` 后，用减少循环数的一次 eager 运行确认 32K、ρ = 1/32 下封装 A 与 D 的整步耗时与正式运行一致。表中「正式」列来自上面的正式 run（eager、单步中位、n = 100）；附录 run 的单步 n 见表。

| 附录 run | commit | 路径 | 单步中位 ms（附录） | 单步中位 ms（正式） | 偏差 | 128 步每步中位（附录 / 正式） | D/A 配对中位（附录 / 正式） |
|---|---|---|---:|---:|---:|---|---|
| g1b-mac-20261017T085522Z-phasefix | 44c66225 | 封装 A（静态 LabCache） | 21.76（n=20） | 21.21 | +2.6% | 21.63 / 20.83 | 1.000 / 1.000 |
| g1b-mac-20261017T085522Z-phasefix | 44c66225 | D（索引读取，含全部新增成本） | 19.25（n=20） | 19.68 | -2.2% | 19.82 / 19.05 | 0.884 / 0.930 |

