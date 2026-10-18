# Notebook 与 HTML：可重放的实验课

## 目的

Codex负责执行，用户事后按计算问题逐段理解。讲解单位是一条机制，不是随意十行代码。Notebook并非正式训练入口，也不维护另一套简化attention。

## 两个模式

默认阅读：展示当前报告对应源码、加载已存在的精简run数据、运行小规模CPU反例。不安装包、不下载权重、不训练、不自动租云。若结果不存在，显示未执行并指向脚本入口，不能用示例数字填洞。

主动重放：用户明确选择run与样本后，调用同一正式实现；长基准和训练用独立py脚本，不由Run All隐式触发。

## 每段固定结构

1. 本段回答什么问题，结果前先显示已登记预测。
2. 展示对应commit中的正式函数（建议每格10–30行以内；长函数按逻辑拆讲，引用不能偷换版本）。
3. 写清输入输出形状与原逻辑位置；必要时显示少量数值。
4. 调用正式函数的小例子或加载正式run记录。
5. 做等价/反例检查；一次只改变一个条件。
6. 解释真实观察、不能推出什么、下一步怎样验证。

## 版本一致

源码与结果要共同锁定。Notebook从当前工作树import但展示旧commit源码是不合格的。推荐在run对应干净工作树导出Notebook，并嵌入展示用源码及其hash；阅读模式只读已保存结果。需要重放旧版本则使用该commit的独立worktree和依赖记录。

如果因dirty工作树运行，必须保留diff及相关源码hash。不能自动修改旧结果以“跟上代码”。模型和tokenizer revision也一起保存。

## 首批课程顺序

当前只有`00_orientation.ipynb`：解释仓库、源材料、载荷预测和未运行状态。不声称MPS基准完成。

第一课`01_environment_and_attention.ipynb`已交付：本机能力、密集mask与gather等价、真少算、索引成本、MPS测量。

第二课记录Qwen接入与整步成本，第三课记录G2-lite质量与反事实边界。后续课程或实验需明确授权；当前只沿第三课已有案例逐段讲解，不创建第四课。

## 输出检查

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_environment_and_attention.ipynb
python -m jupyter nbconvert --to html notebooks/01_environment_and_attention.ipynb --output-dir reports
```

以上是已有Notebook的显式重放命令，不是默认阅读动作。历史课程重放前需核对对应源码/结果版本；本轮直接读已保存HTML，不重新执行三课。

在全新内核从上到下跑通，保留实际输出；HTML检查中文、代码、表格与图片无截断。图仅从测量记录生成；无实测性能前不画预测为实际曲线。小数学例子的数值要标“教学算例”。

GPU长实验日志不直接整段塞进输出，引用run路径及sha256。重模型不驻留读者的默认阅读内核。读取本地HTML不需要启动GPU或云服务。

官方执行机制参考 [SOURCES.md](SOURCES.md) 中Jupyter/nbconvert来源；具体工程约定是本项目设计。
