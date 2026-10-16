# 实验记录最小规格

每个run单独目录，原始大文件不默认推Git。精简可重现材料进入reports或经授权的artifact存储。实际实现可以增加字段，不可把未测值写成0。

## run_manifest.json

参考 [templates/run_manifest.json](templates/run_manifest.json)。必须包含：run_id、UTC、kind（prediction / correctness / capability / benchmark / quality / training）、status、git_commit、dirty_diff_sha、代码/配置hash、model/tokenizer revision、数据hash、seed、设备/系统/软件、backend、dtype、有效与分配长度、命令与计时边界、预算、结果路径与哈希。

null代表未知/未测，不代表零。预测和实测放不同kind；bootstrap验证放correctness且scope=repository_tools，不标GPU能力。

## microbench.jsonl

每行包含组合(model_shape,N0,query_length,ρ,k0,r,F_count,actual_selected,scanned_length,capacity,backend,dtype)、策略、工作集模式/大小、重复编号、elapsed_ms、同步方式、是否编译/重编译、内存字段、异常。保存原始重复值，汇总可重算。

## quality.jsonl

doc_id、task、原始生成顺序、输入hash、目标/生成解析、策略、模型/索引版本、上下文长度、实际输出token、exact_match、逐项评分、format_error、truncated。配对比较按doc_id连接，不能按输出行号碰巧对齐。

## index_diagnostics.jsonl

layer、kv_group、query_position、doc_id、m_C、coverage、omitted_mass、teacher_entropy、KL、local_abs/relative_error、R对照、是否来自D真实轨迹。只对已授权少量样本保存具体文本/位置。

## gate_card.md

参考 [templates/gate_card.md](templates/gate_card.md)。先写预测，测完再填观察与决策。未执行部分留下未执行原因，不补示例结果。
