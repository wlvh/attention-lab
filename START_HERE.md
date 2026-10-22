# 从这里开始

本轮研究已经完成，当前交付是 1 份导读、7 份实验课及对应源码和记录。继续使用已有仓库和环境，不重新初始化或运行早期实验。

## 先选择想理解的计算

| 想看清什么 | 阅读入口 |
|---|---|
| 整项实验最后得到什么 | [第七课：回答质量与实际成本](reports/07_trained_sparse_quality_and_cost.html) |
| 参数怎样通过训练改变选位 | [第四课：一个索引器的训练与 12 参数 CPU 算例](reports/04_can_an_indexer_learn.html) |
| 从位置名单怎样取 K/V，为什么少读仍可能慢 | [第一课：取行与单层读取](reports/01_environment_and_attention.html) |
| 读取替换到真实模型的哪里 | [第二课：Qwen 接入与整步计算](reports/02_qwen_integration_and_decode.html) |
| 未训练的方法怎样回答，改资料怎样改变输出 | [第三课：资料、回答与改写](reports/03_what_is_lost_when_reading_less.html) |
| 怎样为各层分别训练 | [第五课：逐层训练](reports/05_training_all_layer_indexers.html) |
| 本层少读怎样影响后层输入 | [第六课：真实推理状态](reports/06_from_teacher_features_to_sparse_states.html) |

HTML 可直接阅读，不必先运行任何代码。Notebook 入口及导出方式在 [课程目录](notebooks/README.md)。默认执行只读既有结果及小规模 CPU 检查；第四课另做 12 参数、20 步 CPU 教学更新，不加载真实模型或重训正式索引器。

## 哪些文件描述历史

`docs/archive/` 保留原始方案；`docs/tasks/` 和 `reports/cards/` 保留各阶段当时的预测、范围和决定。里面的“下一步”“待审”“未解锁”需按记录日期理解，不代表项目当前仍停在那里。部分原型用到的 P0/P1 阶段名，也不是训练课的索引位置设置。

正式研究结果已经合并；教学修订的状态见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。新实验、合并、云支出都不由阅读这些历史文件自动授权。

## 给接手本项目的 Codex

先读 `AGENTS.md`、`PROJECT_STATUS.md` 和当前用户任务，再查相关课的正式 `.py`、运行版本和记录。不把早期启动说明当成重跑指令；保留 `.venv`、模型缓存、runs、历史分支和 worktree。公开远端仍是 `wlvh/attention-lab`，不要新建仓库或运行旧发布脚本。
