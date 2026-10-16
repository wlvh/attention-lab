# 来源、证据与归属

## 本轮来源层次

**原始上传件**：`docs/source_manifest.json`记录文件名、SHA-256、字节数。v3与roofline由用户转交的Fable 5.1成果，教材由用户提供、通过其提问形成；不把第三方评审归为用户本人产出。原件未改。

**当前对话整理**：BACKGROUND、PLAN_CURRENT、DECISIONS、操作与教学规范根据用户最新要求和后续评审整理。后续修订与原v3冲突时明确标注，原件仍可追溯。

**当前工具验证**：`reports/bootstrap_validation.md`只覆盖仓库启动工具、文档链接/源hash及导读Notebook；不是MPS或Qwen测试。

**历史记录**：早期训练方案、v2、CPU机制核验zip仅作存档。未在本轮重新执行其中全部实验，不把其中报告的通过数或数值作为当前GPU结论。

## 官方操作依据（2026-10-16查阅）

| 来源 | 用途 | 状态 |
|---|---|---|
| [Codex CLI](https://developers.openai.com/codex/cli) | 本地目录内读写与运行命令 | 入口已查阅；实际CLI版本由本机检查 |
| [AGENTS.md](https://developers.openai.com/codex/guides/agents-md) | 项目级指令与继承 | 已查阅，仓库文件不取代用户/工具权限 |
| [GitHub CLI repo create](https://cli.github.com/manual/gh_repo_create) | private建仓、source/push能力 | 历史私有建仓工具依据；当前已使用用户创建的公开仓库，无需再建仓 |
| [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html) | MPS设备与能力检查 | 已查阅；不同发布版本系统要求可能不同，匹配本机版本 |
| [MPS环境变量](https://docs.pytorch.org/docs/stable/mps_environment_variables.html) | fallback与内存保护 | 已查阅；禁把关闭保护当优化 |
| [torch.mps](https://docs.pytorch.org/docs/stable/mps.html) | 同步及可用内存计量API | 对应版本实际存在性由G0确认 |
| [nbconvert执行](https://nbconvert.readthedocs.io/en/latest/execute.html) | 顺序执行Notebook | 官方参考入口；本轮网页抓取失败，实际本地导读执行验证另列 |
| [Jupyter运行](https://docs.jupyter.org/en/latest/running.html) | 本地浏览器与内核 | 操作参考，执行时匹配本机安装 |

Codex官方文档当前可能重定向到OpenAI旗下ChatGPT Learn。使用官方入口，不复制非官方镜像的旧参数。

## 模型/论文依据的处理

模型配置表和索引训练思路继承v3及教材已有引用。本次是建仓整理，不重新进行论文性能审计。正式集成前需从官方模型仓库和所用Transformers源码读取并固定revision，核对q_norm/k_norm、RoPE、cache更新顺序与模型上下文上限。

参考入口：
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- [Qwen3旧版本参考源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py)（理解结构用，不强制安装该版本）
- [DeepSeek V3.2报告](https://arxiv.org/html/2512.02556v1)（冻结教师训练索引的背景，不代表本项目复现其完整架构）

本项目的分组投影、P0/P1、固定读取预算、任务阈值是实验设计，不是模型官方承诺。云价格以真正租用当天报价核对，本包不提供实时租用报价。

## 原件使用与许可

2026-10-16用户明确选择公开仓库。未选新的开源许可证；第三方材料和模型分别保留原许可。此次公开仅限本项目材料，Pages或额外平台发布另行决定。参见根目录 `RIGHTS.md`。
