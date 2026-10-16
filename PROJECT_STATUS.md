# 接力状态

更新日期：2026-10-16。状态：**MATERIALS_PUBLISHED / G0_NOT_RUN_ON_MAC**。

## 远端与发布

用户创建的公开仓库：`wlvh/attention-lab`，默认分支 `main`。GitHub读取已经核对 `private=false` 与 `push=true`；首次入口提交为 `95885aa2c24d692c595c3ea88024a6c836ffdea6`。完整导入应以远端当前提交、文件清单和 SHA-256 核验为准，不以创建空仓库代替文件上传。

## 已准备

当前有效方案、背景与决策记录、原始 v3/roofline/教材、Mac 操作手册、Codex 接力指令、Notebook 交付规范、预算、测量协议、分关任务书，以及预测核算、来源和链接校验工具。导读只是启动材料，不是性能结果。

## 尚未完成

用户 Mac 的环境检查、GPU 运算和微基准；Qwen 适配器、静态 KV/z、任务评测与索引训练；G1–G5 的任何速度或质量结论。没有启动云资源或付费模型 API。

## 下一步

克隆仓库，在 Mac 本地 Codex 打开，读 `AGENTS.md`、`docs/CODEX_HANDOFF.md`，在 `task/g0-g1a-mac` 分支执行 `docs/tasks/01_G0_G1A.md`。不要重新建仓或运行历史私有发布脚本。

## 接力格式

每次记录分支、代码 SHA、run_id、实际命令、证据路径、失败/未测项、剩余预算与下一条命令。未完成进程的 PID 与主机路径保存在本地忽略目录，不直接写进公开报告；公开接力只提供去标识化状态。

## 本轮发布检查

20项启动工具测试通过，九个来源原件哈希一致，导读Notebook已在Linux CPU环境顺序重跑。完整文档与脚本已纳入本次提交；Mac环境预检、真实注意力实现与训练仍未执行。详见reports/public_publication.md。
