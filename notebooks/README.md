# 教学 Notebook

| Notebook | 内容 | 默认模式 |
|---|---|---|
| `00_orientation.ipynb` | 仓库导读、源材料校验、载荷预测工具 | 只读，标准库 |
| `01_environment_and_attention.ipynb` | G0 与 G1a 的实验课：取子集两种写法为何差 25 倍；少读 KV 为何仍可能更慢；分段相加为何不等于链式；预测与观察对照 | 只读：加载 `reports/` 的实测 JSON，展示同版本源码，CPU 小例子；不加载模型，不跑基准 |
| `02_qwen_integration_and_decode.ipynb` | G1b 的实验课：把索引注意力装进真实 Qwen——为什么单层快不等于整模型快；真实接入位置、缓存增长、完整步骤耗时、预测成立或失效的原因 | 只读：加载 `reports/g1b_*.json`，展示同版本 `qwen_wrap.py` 源码，tiny 随机 Qwen3 的 CPU 小例子；不下载权重，不跑整模型基准 |
| `03_what_is_lost_when_reading_less.ipynb` | G2-lite 的实验课：少看历史会丢掉什么——A / B / C / D0 四种读取方式在同一前缀、同一问题、同一名额下的质量对照；基线可用性、覆盖率、代表案例的真实选择位置、反事实改写 | 只读：加载 `reports/g2lite_*.json`，展示同版本选择器源码，tiny 随机 Qwen3 的 CPU 契约重放；不加载 0.6B 模型，不跑 GPU |
| `04_can_an_indexer_learn.ipynb` | G3a：固定主体，仅训练第14层索引器；训练/留出 KL、教师覆盖、局部误差、实际名单与残余误差 | 读取 G3a 归档，并运行12参数CPU教学算例；不加载0.6B、不启动研究训练或GPU |
| `05_training_all_layer_indexers.ipynb` | G3b：28层参数、逐层指标与完整路径训练的局限 | 只读归档、重算与画图，不加载模型或训练 |
| `06_from_teacher_features_to_sparse_states.ipynb` | G3c：单层/全层接入、全选检查、同token状态差异 | 只读已有记录和源码，不运行模型 |
| `07_trained_sparse_quality_and_cost.ipynb` | G4-lite：新T1自由生成、配对错误、实际选位和同token成本 | 只读冻结记录，CPU重判与作图，不运行模型 |

已执行的 HTML 在 `../reports/`。从全新内核重跑并导出：

```bash
source .venv/bin/activate
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_environment_and_attention.ipynb
python -m jupyter nbconvert --to html notebooks/01_environment_and_attention.ipynb --output-dir reports
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_qwen_integration_and_decode.ipynb
python -m jupyter nbconvert --to html notebooks/02_qwen_integration_and_decode.ipynb --output-dir reports
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_what_is_lost_when_reading_less.ipynb
python -m jupyter nbconvert --to html notebooks/03_what_is_lost_when_reading_less.ipynb --output-dir reports
```

规范见 `../docs/TEACHING_NOTEBOOKS.md`。后续课程按关卡增加，不预先创建空 Notebook。

第四课：`.venv/bin/python -m jupyter nbconvert --execute --to notebook --inplace notebooks/04_can_an_indexer_learn.ipynb`，随后 `--to html --output-dir reports`。只加载保存结果；整页视觉验收范围见 `PROJECT_STATUS.md`。

第五课：`.venv/bin/python -m jupyter nbconvert --execute --to notebook --inplace notebooks/05_training_all_layer_indexers.ipynb`，随后 `--to html --output-dir reports`。

第四课3a节展开输入、初始参数、索引分数、教师概率、真实梯度、Adam改变量与名单，完整小型记录在 `../reports/lesson04_cpu_trace.json`；它没有固定集合且只选一个位置，不能展示多位置主注意力重新加权。
