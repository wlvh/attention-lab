# 教材按稳定锚点定位

原HTML原样保留。题号可能随版本变化，因此用标题和锚点；本次检查下表所有id实际存在。

| 锚点 | 对应内容 | 本地HTML |
|---|---|---|
| `model-layer-basics` | 先分清：模型的“层”、层里的“头”和 token 的位置 | [跳转](reference/handbook_with_exercises_updated%284%29.html#model-layer-basics) |
| `qkv-history-followup` | 小王 所以，生成 Q/K/V 不依赖历史信息，只有注意力计算才依赖历史 K/V，是吗？ 大王： 要区分 “这一步直接读取什么” 和 “它的输 | [跳转](reference/handbook_with_exercises_updated%284%29.html#qkv-history-followup) |
| `attention-index-comparison` | 和普通 Transformer 相比，索引筛选多做了哪一步？ | [跳转](reference/handbook_with_exercises_updated%284%29.html#attention-index-comparison) |
| `q-index-training` | 这些系数是什么？没见过我的资料，怎么提前学好？ | [跳转](reference/handbook_with_exercises_updated%284%29.html#q-index-training) |
| `q-index-relevance` | 模型怎样学会判断相关性？DeepSeek V4.1 Flash 为此增加了新的训练任务吗？ | [跳转](reference/handbook_with_exercises_updated%284%29.html#q-index-relevance) |
| `q-sparse-noncontiguous` | 没选中的第 1、3～8 条要重算吗？第 2、9 条又怎样参与生成第 10 个 token？ | [跳转](reference/handbook_with_exercises_updated%284%29.html#q-sparse-noncontiguous) |
| `layer-selection-explained` | 在生成同一个 token 的过程中，不同层就可能读取不同历史。 第 9 个位置的内部数值经过前面的层后发生了变化，后面的层据此产生自己的查询 | [跳转](reference/handbook_with_exercises_updated%284%29.html#layer-selection-explained) |
| `q-kv-quantization` | KV 量化怎样真正省空间？为什么存得更小仍可能跑得更慢？ | [跳转](reference/handbook_with_exercises_updated%284%29.html#q-kv-quantization) |
| `q-flash-execution` | 这些临时结果为什么也会占用大量显存？FlashAttention 怎样解决这个困难？ | [跳转](reference/handbook_with_exercises_updated%284%29.html#q-flash-execution) |
| `appendix-repro-lab` | 把缓存亲手做出来：完整代码与实际结果 | [跳转](reference/handbook_with_exercises_updated%284%29.html#appendix-repro-lab) |
| `exercise-prefix-review` | 评审一个最小前缀缓存 | [跳转](reference/handbook_with_exercises_updated%284%29.html#exercise-prefix-review) |
| `exercise-state-recovery` | 存得下、读得动，能从哪里恢复？ | [跳转](reference/handbook_with_exercises_updated%284%29.html#exercise-state-recovery) |

这些是教材位置，不是本项目已完成实现的声明。完整文件与hash见source_manifest.json。
