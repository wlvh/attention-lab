# 从这里开始

## 给项目所有者

不需要先学会逐格运行 Notebook。把本目录放到 Mac 的独立项目目录，在**本地** Codex 中打开，让它读 `AGENTS.md` 和 `docs/CODEX_HANDOFF.md`。

第一次只完成 **G0 环境预检 + G1a 最小执行可行性**。不要直接下载三个模型、跑全部训练或租云。

日常阅读 `reports/` 的 HTML；需要改一个变量、重放小例子时打开 `notebooks/`。当前的 `00_orientation.ipynb` 是导读与预测核算，不是 Mac 测速报告。

## 已有公开远端

所有者已经创建并明确选择公开仓库 `https://github.com/wlvh/attention-lab`。本次导入保留全部九个原始材料；当前入口与发布约定调整为公开仓库，不改写原始方案。

```bash
git clone https://github.com/wlvh/attention-lab.git
cd attention-lab
```

无需执行旧的 `scripts/publish_repo.py --execute`；该脚本是历史私有建仓工具，对现有公开仓库会拒绝操作。后续用正常分支、提交和 PR 推进，不改可见性、不 force push。详细说明见 `docs/tasks/00_PUBLISH.md`。

## 给刚接手的 Codex

最短阅读路线：

`AGENTS.md → PROJECT_STATUS.md → docs/PLAN_CURRENT.md → docs/tasks/01_G0_G1A.md`

操作细节按需查 `docs/MAC_RUNBOOK.md`；接口查 `docs/IMPLEMENTATION.md`；验收查 `docs/MEASUREMENT.md`；每关解释怎么交查 `docs/TEACHING_NOTEBOOKS.md`。

原始教材不需要每轮全部放入上下文。按 `docs/REFERENCE_MAP.md` 定位相关锚点。历史 v1/v2/v3 只有溯源权，不覆盖当前补充约定。

## 一页构建顺序

| 阶段 | 创建什么 | 用什么事实解锁下一阶段 |
|---|---|---|
| G0 | 隔离环境、能力与内存清单 | 实际所需 MPS 运算正确；CPU/后端异常明确 |
| G1a | 单层 A 与 D 的真实子集执行、测试、微基准 | 至少有值得复核的设备/形状/执行方式组合；否则一次定位后收口 |
| G1b | 最小模型封装、静态 KV/z、连续生成 | 全选与分段对照通过，连续运行没有失控重编译或状态污染 |
| G2 | 任务生成与完整基线校准 | 最小可用模型与 G1 性能可行组合有交集 |
| G3 | 单层 P0/P1 → 全层小索引器 | 留出资料与实际稀疏轨迹支持其选择能力；解释均匀分布例外 |
| G4 | 冻结测试，A-step/A-best 与 D 联合比较 | 质量、三口径耗时、内存与预算可一起判断 |
| G5 | 可选分层预算/名单复用 | 前面有证据、预算有余量，改结构后重新测速 |

“支持运算”“通过微基准”“模型具有质量”“正式质量—速度取舍”是四件不同的事。不要提前宣布后一件已经成立。
