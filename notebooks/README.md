# 教学 Notebook

| Notebook | 内容 | 默认模式 |
|---|---|---|
| `00_orientation.ipynb` | 仓库导读、源材料校验、载荷预测工具 | 只读，标准库 |
| `01_environment_and_attention.ipynb` | 人工 Q/K/V 数组：先分别测打分、选位、取行，再把它们与主注意力连起来，比较一次读取的总耗时 | 只读：加载 `reports/` 的实测 JSON，展示同版本源码，CPU 小例子；不加载模型，不跑基准 |
| `02_qwen_integration_and_decode.ipynb` | 真实 Qwen 权重、固定随机 token 输入：检查原库与封装数值，再测完整前向和连续 128 步缓存增长 | 只读：加载 `reports/g1b_*.json`，展示同版本 `qwen_wrap.py` 源码，tiny 随机 Qwen3 的 CPU 小例子；不下载权重，不跑整模型基准 |
| `03_what_is_lost_when_reading_less.ipynb` | 读编号—数值资料并自由回答：完整、近期窗口、随机、未训练索引怎样选位、答错什么，改资料后怎样变化 | 只读：加载 `reports/g2lite_*.json`，展示同版本选择器源码，tiny 随机 Qwen3 的 CPU 契约重放；不加载 0.6B 模型，不跑 GPU |
| `04_can_an_indexer_learn.ipynb` | 保存第 14 层四个查询与历史 K/V，只更新索引参数：从候选分数、KL 到新名单和局部读出；附 12 参数 CPU 算例 | 读取 G3a 归档，并运行12参数CPU教学算例；不加载0.6B、不启动研究训练或GPU |
| `05_training_all_layer_indexers.ipynb` | 各层分别提取完整模型特征、训练和局部评测；前层训练结果不接回模型改变后层输入 | 只读归档、重算与画图，不加载模型或训练 |
| `06_from_teacher_features_to_sparse_states.ipynb` | 把各层索引器接回模型，强制输入同一串 token：比较各方法自己计算的 Q/K、读出和词表分数 | 只读已有记录和源码，不运行模型 |
| `07_trained_sparse_quality_and_cost.ipynb` | 32 份新资料上各自生成答案；另输入同一串 32 个控制 token 比成本，区分质量、读取量与缓存占用 | 只读冻结记录，CPU重判与作图，不运行模型 |

已执行的 HTML 在 `../reports/`。从全新内核重跑并导出：

```bash
source .venv/bin/activate
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_environment_and_attention.ipynb
python scripts/export_lessons.py --lessons 1
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_qwen_integration_and_decode.ipynb
python scripts/export_lessons.py --lessons 2
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_what_is_lost_when_reading_less.ipynb
python scripts/export_lessons.py --lessons 3
```

规范见 `../docs/TEACHING_NOTEBOOKS.md`。00—07 已交付，现有任务是教学修订，不因打开课程自动新增实验。

第四课：`.venv/bin/python -m jupyter nbconvert --execute --to notebook --inplace notebooks/04_can_an_indexer_learn.ipynb`，随后运行 `.venv/bin/python scripts/export_lessons.py --lessons 4`。读取保存结果并执行 12 参数、20 步 CPU 教学更新；不加载真实模型。整页视觉验收范围见 `PROJECT_STATUS.md`。

第五课：`.venv/bin/python -m jupyter nbconvert --execute --to notebook --inplace notebooks/05_training_all_layer_indexers.ipynb`，随后运行 `.venv/bin/python scripts/export_lessons.py --lessons 5`。

第四课3a节展开输入、初始参数、索引分数、教师概率、真实梯度、Adam改变量与名单，完整小型记录在 `../reports/lesson04_cpu_trace.json`；它没有固定集合且只选一个位置，不能展示多位置主注意力重新加权。


局部测速、CPU 检查、小算例和复现记录默认折叠，正文保留简述。Notebook 通过折叠元数据保留全部代码与输出；HTML 可以点击展开。使用统一导出入口以保留折叠状态：

```bash
.venv/bin/python scripts/export_lessons.py
.venv/bin/python scripts/export_lessons.py --lessons 7
```

默认导出第 1—7 课，`--lessons` 可指定课程。导出不执行代码；原有 `presentation-only` 输入现在折叠保留，支持性实验的代码和记录一并折叠，核心结果正常展开。文字或折叠元数据修改无需重跑模型；计算代码改变时，从新内核顺序执行相应 Notebook。
