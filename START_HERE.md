# 从这里开始

## 给项目所有者

这是既有研究项目的阅读入口。G0/G1a/G1b 已完成，速度主线停止；G2-lite 主交付已随 PR #5 合并（`d2fc4ad`），现在从真实案例开始学习。不要因接管新会话重建环境、重跑关卡或下载模型。

直接打开已保存的 HTML，无需执行代码：

| 课程 | 入口 | 内容版本 |
|---|---|---|
| 01 环境与单层注意力 | [阅读第一课](reports/01_environment_and_attention.html) | G0/G1a；`ff31d3a` 的措辞校正版 |
| 02 Qwen接入与整步解码 | [阅读第二课](reports/02_qwen_integration_and_decode.html) | G1b；`ff31d3a` 的措辞校正版 |
| 03 少读历史会丢什么 | [阅读第三课](reports/03_what_is_lost_when_reading_less.html) | 旧v2运行 `af62df4`、报告代码 `72b848d`、导出 `a3832a8`；整页布局未验 |

第三课第一段可直接看 `T1-2K-00` 的 B / ρ=1/8 改值案例：目标记录在窗口外，回答却从8340变成1563。分清资料Prefill和后续读取，不把“仍错”当成“未变”。先讲这一段，随后由用户决定下一段。

完整版本边界与 Notebook 入口见 [notebooks/README.md](notebooks/README.md)。`00_orientation.ipynb` 仍是历史导读与预测核算，不是额外一份实测课。

## 给刚接手的 Codex

最短路线：

`AGENTS.md → PROJECT_STATUS.md → docs/HANDOFF_2026-10-18.md 当前段 → notebooks/README.md`

[交接当前段与诊断索引](docs/HANDOFF_2026-10-18.md#current-state-20261018)明确区分已合并主交付、同批案例诊断和未授权实验。前半段历史交接原文不作为当前执行指令。PR #10 的文档更新待审，不自动合并。

- 原始v2结果、旧manifest和预测原文保留；当前配置v3是C的控制规则修复，NOT_RUN。
- #6–#9不合并、不关闭、不删除、不改写历史；不能从#9复制旧代码/报告覆盖修正后的main。
- G3、追加ρ、T2新变式、Metal/CUDA/更长上下文均未授权；没有新增云或付费API授权。
- 默认读HTML。Notebook仅在明确需要时使用已有环境做CPU小例子，不自动下载、训练或运行GPU基准；不要无意触发整套测试或旧实验入口。

具体接管提示见 [docs/CODEX_HANDOFF.md](docs/CODEX_HANDOFF.md)。操作、接口和教学规范按需查看，不重新执行历史启动任务。

## 已有公开远端

仓库是用户已创建的公开仓库 [wlvh/attention-lab](https://github.com/wlvh/attention-lab)。只有没有本地源码的新读者才需要克隆；本机既有项目复用原目录。无需执行历史 `scripts/publish_repo.py --execute`，不重新建仓、不改可见性、不force push、不自动开启Pages。

研究目标仍是通过可运行、可干预、可解释的结果理解取舍。负结果、失败边界和明确的未知都是交付；不以新增PR或新实验数量替代教学。
