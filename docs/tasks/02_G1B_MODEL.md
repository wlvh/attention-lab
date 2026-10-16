# 后续任务：G1b模型集成

状态：**UNLOCKED（2026-10-17，由 G0/G1a 证据解锁；PR #1 合并提交 `0798aa7`），进行中**，分支 `task/g1b-qwen-mac`。`03_G2_TO_G5.md` 仍为 BLOCKED。

从0.6B单模型开始。核对官方revision和实际q/k/v/o、归一化、RoPE与cache接口，冻结主体，只替换self_attn读取部分。先dense封装与未改原模型数值对应，再加D。

静态KV和z分配分开；z在RoPE前K处建；一次/分块索引表一致。固定请求k，valid_length增长；测固定N单步和连续128步，记录重编译、CPU同步、分配与回退。

最少测试详见IMPLEMENTATION。保留A-step/A-best的输入处理接口。全选等价不等于小子集质量保持；profile确认子集实际少算。随机token仅用于执行轨道，不评价任务质量。

输出：代码/模型revision、数值测试、每个形状的能力/速度/内存可行表、缓存教学课、下一关能用的模型集合。更大模型只有能力或性能问题需要时才增加。
