# PR #5 收尾修复 · 2026-10-18

范围：在 `task/g2-lite-mac` 的 `bb1b639` 后追加修复提交，保留四个正式 G2-lite run；不重跑模型，不启动 G3 或速度优化，不合并，不修改 main 或诊断分支 #6–#9。

## 修复与验收对象

| 问题 | 改动 | 证据／检查 |
|---|---|---|
| C 改写时随机名单也变 | 新运行使用 `parent_doc_id or doc_id`，流仍由请求计数与步号区分 | `test_c_rewrites_replay_actual_original_positions_on_cpu`：真实 tokenizer 资料与 value/irrelevant 改写，同候选域、k、请求计数、步号，逐层逐组实际原位置名单一致 |
| 旧 C 反事实被当成受控结果 | 新汇总显式保存随机条件是否相同；旧 24 个 C 改写均标混杂 | 从旧 JSONL 配对检查，不重跑、不倒填种子或 manifest；D0 仍对变化输入重新打分 |
| 回答不变与仍然错误混淆 | 分列回答变化、正误变化、两次都错、编号回声及转为编号 | `rewrite_observations()` 生成报告、卡片区块与 Notebook 同一张表；反例 `T1-2K-00/B/ρ=1/8` 保留 |
| “必须同层组读齐四位值” | 撤回必要条件和结构性下界，区分记录覆盖、值 token 覆盖与答案 | 原预测保留；说明读齐仍错只否定充分性，不据此否定必要性 |
| 案例邻域键被覆盖 | 使用实际记录序号 `j` | 对 24 份对照资料检查目标及全部邻近记录，不只检查一个例子 |
| 预算只显示首个值 | 展示 k0、总读取数实际范围；按首生成步逐请求计算 k/M 与总读取比例 | 对照旧记录范围；与ρ分列，分母为首生成步 N=P+Q、M=N−|F| |
| 报告隐式导入 torch | 按路径局部加载纯 Python 汇总模块；包初始化不改 | `python -S` 下 `--help`；仅开放现有 NumPy 的隔离导入路径重建归档报告，确认找不到 torch/transformers |
| 内存提前停止仍写 done | 记录计划/完成/跳过/待处理数量、原因；未完成状态与非零退出码 | CPU 模拟容量跳过、警戒线停止、部分请求；报告拒绝 incomplete 及错标 done 的不完整计数 |

正式选择器、Qwen 封装、评分定义没有改变；新代码中的 C 种子修复影响未来运行，不追溯性地改变历史结果。配置 v2 的原件由快照保留；当前配置 v3 只登记软件修订，**NOT_RUN**。

## 来源与不变量

- 旧模型运行：`af62df48d2a993dc6b4e0369ee4137bd6cc47615`，Qwen3-0.6B 与 tokenizer revision 均为 `c1899de289a04d12100db370d81485cdf75e47ca`，配置 v2。
- 四个 run：`g2lite-20261017T130007Z-pass1-A`、`g2lite-20261017T130319Z-pass2-ABCD0`、`g2lite-20261017T130551Z-pass3a-rewrites`、`g2lite-20261017T130715Z-pass3b-move`。
- 旧归档、逐题答案、run manifest、旧归档清单的 SHA-256，以及主表/门槛/配对表的固定摘要，记录在 `reports/provenance/g2lite_pr5_review_manifest.json`；`check_provenance.py` 校验。
- 修改过的运行器、汇总文件和配置，各自的 `af62df4` 原件在 `reports/provenance/g2lite_pr5_review/`；修复后的差异另列。旧 manifest 不修改。
- 新报告的生成代码提交、逐文件源码 SHA-256、旧输入文件与名单位图哈希另记于 `reports/g2lite_report_build.json`，它不是新模型 run manifest。
- 卡片在 `## 实际执行` 之前的预测和正式运行前修订保持逐字节相同；观察与判定可以纠正，不能改写预测来贴合结果。

主结果保持：T1 A 为 12/12、10/12；B 为 2/12、2/12、2/12、3/12；C/D0 四格均 0/12。T2 当前提示、生成及评分协议下为 baseline_too_weak（8/12、1/12）。新报告不能被解释为模型能力的一般结论。

## 早期证据缺口与留后事项

v1 `calib-w2`、`calib-w4`、`calib-w5` 的生成器哈希为 `f6ea72476fe667c3ca8b31bb29228f6c54f8593cde02e9cceb1823ab047f23e4`。现有分支历史中的三个生成器版本均不匹配；配置与回答保留，但没有恢复出确切源码。不编造快照，不把这一缺口升级成正式 v2 主结果的无限前置条件。后续 v2 校准源码哈希可对应 `af62df4`。

v1→v2 同时调整记录数量、填充布局与措辞；不能把改善单独归因于密度。PR #6–#9 属同一批案例的后续诊断，其默认输出路径、档案搬迁入口等维护问题不在本补丁处理；不把诊断结果加入 #5 冻结表。

## 可重放的收尾检查

使用已有项目环境，离线运行；不安装或下载模型，不调用 MPS：

```bash
ATTENTION_LAB_TEST_CPU_ONLY=1 HF_HUB_OFFLINE=1 python -m pytest -q tests/test_g2lite.py tests/test_g2lite_review.py
python scripts/check_sources.py
python scripts/check_repository.py
python scripts/check_provenance.py
python scripts/g2lite_report.py --review-card reports/cards/G2lite_mac_m4max.md
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_what_is_lost_when_reading_less.ipynb
python -m jupyter nbconvert --to html notebooks/03_what_is_lost_when_reading_less.ipynb --output-dir reports
```

报告只需 Python 标准库与 NumPy（读取 NPZ 名单位图）；Notebook 的已有 CPU 演示另需 torch、transformers、Matplotlib、Jupyter/nbconvert。报告入口的无 torch 检查使用 `-S` 排除 site-packages，只给 NumPy 建立临时只读包入口，没有创建新 Python 环境或改变现有环境。

## 本次验证状态

- 生成代码提交：`72b848d76683ac0bdc26d4c7cdf8e0d67b24fff8`；报告生成清单中的源码文件无未提交变化。后续证据提交仅更新导出产物、验收记录和来源 diff 的无上下文表示。
- 针对性 CPU：25 项测试、5 个子测试通过；包含原资料与 value/irrelevant 的实际名单重放、旧 300 条请求重新判分、邻域完整性、实际预算、提前停止记账和报告拒绝不完整记录。
- 无 torch：`-S` 的帮助入口通过；只开放现有 NumPy 后从归档生成报告通过，环境中找不到 torch/transformers；没有安装软件或初始化环境。
- 标准库环境：89 项测试通过，其中 53 项因第三方依赖不可用按设计跳过；这不替代上面的 CPU 契约检查。
- 不变量：逐题答案、旧 manifest、旧 run 归档及旧归档清单逐字节不变；主表准确率、门槛和配对表摘要不变；预测原文摘要不变。新字段只修汇总、展示和解释。
- 实验课：现有 Python 3.12.10 的新内核顺序执行 18 个代码单元，无错误输出；HTML 已导出。9 张图像与旧版逐字节相同，实际有 5 张独立图，已逐一查看标题、坐标、图例与数值趋势，无截断。
- 展示边界：**整页 HTML 布局未验**。此前浏览器安全策略拒绝 file://；本轮通过受支持的 Codex 文件预览请求打开，返回 queued，未确认实际展示。只把上述保存图像列为已视觉检查，不以执行成功代替整页验收。
- 远端 CI 独立于本地结果，以 [PR #5 当前 head 的 checks](https://github.com/wlvh/attention-lab/pull/5/checks) 为准；交付时读取实际状态，不沿用旧绿色徽章。

停止点：修复提交、旧记录重生成的报告及第三课交付后，等待用户复审。合并需要用户明确指令，采用 merge commit 并保留分支。
