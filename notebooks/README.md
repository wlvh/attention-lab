# 三份实验课的阅读入口

默认直接打开HTML，不执行Python、不下载模型、不训练、不启动GPU基准。打开Notebook也可先阅读已保存输出；其中的可执行小例子仅用CPU，不会隐式启动正式模型实验。

| 课程 | 已保存HTML | Notebook | 内容与版本 |
|---|---|---|---|
| 01 环境与单层注意力 | [直接阅读](../reports/01_environment_and_attention.html) | [源码与保存输出](01_environment_and_attention.ipynb) | G0/G1a；`ff31d3a` 含PR #4措辞更正，数据来源见G0/G1a报告和manifest |
| 02 Qwen接入与整步解码 | [直接阅读](../reports/02_qwen_integration_and_decode.html) | [源码与保存输出](02_qwen_integration_and_decode.ipynb) | G1b；`ff31d3a` 含PR #4措辞更正，历史运行源码在provenance/g1b；tiny CPU示例不是0.6B复测 |
| 03 少读历史会丢什么 | [直接阅读](../reports/03_what_is_lost_when_reading_less.html) | [源码与保存输出](03_what_is_lost_when_reading_less.ipynb) | 旧v2运行 `af62df4` → 报告生成代码 `72b848d` → 保存导出 `a3832a8`；当前v3 NOT_RUN |

[00导读](00_orientation.ipynb)用于认识原方案和预测核算，不是实测课程。三课都已进入main；当前项目状态与固定版本诊断索引见[交接当前段](../docs/HANDOFF_2026-10-18.md#current-state-20261018)。

第三课修订保留旧回答与成绩，标注旧C改写的随机条件混杂，修正邻域与预算展示，撤回未经证明的机制结论。18个代码单元已从新内核执行，9处图像中5张独立图已检查；**整页HTML布局仍未验**。本轮预览状态查询超时，没有确认开头、主要表格、案例、代码区和结尾的实际页面，不把执行成功当视觉验收。

以下为明确需要重放时的命令，不是默认阅读动作；本轮不执行。01/02 的历史源码与当前工作树可能不同，重放前须核对对应快照，不能把当前源码冒充旧run版本。03 已显式区分旧运行快照与当前汇总代码。使用已有隔离环境，不安装或下载模型：

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
