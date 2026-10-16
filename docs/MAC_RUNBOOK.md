# Mac 本地操作手册

本手册让本地Codex执行；用户只处理必须的本机授权。当前未在用户Mac实测，不保证某个PyTorch版本、BF16或compile组合可用。官方入口见 [SOURCES.md](SOURCES.md)。

## 1. 项目放在哪里，谁在计算

建议目录`~/Projects/attention-lab`，避免把虚拟环境和大缓存放入自动同步目录。Codex本地模式或CLI打开这个目录。编码模型的在线服务与本机Qwen实验算力是两件事：选择云端Codex执行环境不会自动使用Mac GPU。

```bash
cd ~/Projects/attention-lab
codex
```

已安装Codex无需重复安装。尚未安装时按官方CLI文档走现行安装与登录流程，不使用过期模型名/危险全权限参数。项目不要求另外购买API额度。

## 2. 首轮允许哪些动作

在目录内建立隔离Python环境、安装必要依赖（首次联网由本地权限策略批准）、只读检查硬件、保存日志、运行有界小运算与G1a。不要使用sudo改系统、关闭安全限制、上传凭证、自动租云。

确认已在项目根目录，优先选择现有可用的原生Python 3.12（不是要求重装系统Python）：

```bash
uname -m
python3 -c 'import platform,sys; print(platform.machine(),sys.version,sys.executable)'
python3 scripts/environment_inventory.py --output runs/inventory-before.json
```

若Python为x86_64而机器为Apple Silicon，先解决原生环境，不在Rosetta环境上给出GPU结论。输出只收集所需元数据，不打印完整环境变量或用户凭证。

## 3. 建隔离环境与可用版本

以下是命令模板，若目录已存在先核对而不是覆盖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/bootstrap.in
python -m pip check
```

bootstrap.in是首次解析依赖的输入清单，**不是锁文件**。G0运算通过后生成精确记录：

```bash
python -m pip freeze > requirements/lock-macos-arm64.txt
python scripts/environment_inventory.py --require-mps --output runs/inventory-after.json
python -m ipykernel install --sys-prefix --name attention-lab --display-name 'Attention Lab (.venv)'
```

不要在CPU开发容器生成的freeze冒充Mac锁文件。首次正式比较后不随意升级；升级时先保存旧锁并重测相关部分。

## 4. MPS能力与内存

`environment_inventory.py`只采集信息；MPS available并不代表关键操作/反向/编译已经通过。G0实现必须实际测试matmul、top-k、gather、SDPA或明确参考路径、索引小矩阵反向和同步。

先CPU小FP64参考，再MPS FP32，然后本机支持的低精度；MPS不以FP64作为执行要求。标清torch版本、OS与实际dtype。compile可选预检，不承诺自动支持或更快。

正式纯MPS基准不设置`PYTORCH_ENABLE_MPS_FALLBACK=1`；非支持操作应明确报错或分出CPU混合路径，不偷偷纳入MPS速度。不要设`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0`关闭保护。实际可用内存、安全限额可能低于用户总内存，取更小值。

80GB预警/100GB工作集硬边界不是一次申请100GB的指令。每档运行前估计张量、索引与临时量；使用系统memory pressure和swap观察，压力出现立即减规模或停止。进程RSS、MPS active/driver allocated有重叠，不可简单相加。

## 5. 正确计时

代码计时前后调用目标设备同步，例如MPS的`torch.mps.synchronize()`。数据分配/模型加载、预热/编译、正式循环、剖析分别记录。禁止每个算子都sync后把人为串行耗时写成真实端到端。

固定N单步之后必须测连续生成：k固定但valid_length增长。记录graph break/recompile；控制分配容量和实际扫描域。Mac轮换工作集不能直接用4080 L2大小作证据。

## 6. 长运行、防休眠与停止

长实验先保存小检查点，再启用项目范围的日志/PID记录。运行时保持电源和散热条件相近。可在明确长命令前使用macOS `caffeinate -i`保持运行期间不空闲休眠；合盖/断电等行为不保证被它避免。

G1a基准脚本尚未存在，Codex在创建时必须提供前台命令、带日志命令、受限退出与恢复命令；不要现在用一个虚构脚本名声称能够执行。

停止优先Ctrl-C；后台仅对登记并核对仍属于本项目的PID发送TERM，先让程序保存。不要`killall python`。恢复前检查旧进程是否仍活着、run元数据与checkpoint是否匹配，避免启动第二份GPU作业。

## 7. 结果放哪、怎么读

run原始记录放`runs/<run_id>/`，默认gitignored；提交精简结果与可追溯manifest到reports对应目录。不要提交模型下载、KV数组、整段敏感机器日志。

```bash
source .venv/bin/activate
python -m jupyter lab --ip=127.0.0.1
```

保持默认登录保护，仅监听本机，不用`0.0.0.0`或关闭token。Notebook选项目内核；默认Run All只读结果和小例子。HTML可双击打开，`00_orientation.html`是导读，不是GPU实验结果。

## 8. 故障定位

| 现象 | 先查什么 |
|---|---|
| MPS unavailable | 原生Python、torch构建、macOS实际要求；查匹配版本官方文档 |
| dtype不支持 | 降到FP32验证，不把其他dtype结果混进同表 |
| 子集版慢 | 索引、top-k是否逐组循环、gather写读、同步、分配；先剖析再训练 |
| 内存剧增 | 全量attention/logits、重复KV、保留反向图/Notebook引用、compile工作区 |
| 固定N快、连续生成慢 | k/shape变化、Python整数guard、重编译、每步同步与整段复制 |
| 只在Notebook成功 | 隐藏状态、错kernel、工作路径；全新内核和py入口复验 |
| Codex中断 | 读PROJECT_STATUS与run日志，不靠聊天推断完成 |

## 9. Windows / 云备用

以后需要CUDA时再检查Windows主机的RAM、GPU型号和VRAM、驱动、实际Python/WSL环境。不在当前任务预先安装WSL或远控。切换设备保留同一问题和实现，但单独报告结果。

云租用先看 [BUDGET.md](BUDGET.md)，即使预算余量足够也不等于已经批准启动服务。
