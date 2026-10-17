# G1a 证据链归档

PR #1 评审指出：六份正式 run 的 manifest 记录的 `bench/microbench.py` SHA-256（`4c1e5b66…`）与提交文件（`69216b81…`）不一致。本目录把证据链接上。

## 内容

| 文件 | 作用 |
|---|---|
| `microbench_at_formal_runs.py` | 跑分时的基准源码，由实施会话的编辑记录精确还原，SHA-256 与全部六份 manifest 一致 |
| `microbench_post_run_edit.diff` | 跑分后到提交之间对该文件的全部改动：一行，把 manifest 里的路径参数做去家目录处理，不涉及计时、执行路径或数据生成 |
| `g1a_runs_20261016.tar.gz` | 七个 run 目录（1 个 smoke、5 个 eager、1 个 eager+compile）的逐轮 `microbench.jsonl`、`summary.json`、`manifest.json`、`profiles.txt`，加序列日志与脚本 |
| `archive_manifest.json` | 以上文件的字节数与 SHA-256，以及无 GPU 重新生成报告的命令 |

## 核验

```bash
python scripts/check_provenance.py
```

CI 在每次推送时运行它。它检查三件事：快照哈希等于每份 manifest 记录的哈希；归档文件的字节数与哈希未变；提交的 `bench/microbench.py` 与快照只差 diff 文件记录的那些行。以后任何对基准脚本的修改都必须同时更新该 diff 文件，否则检查失败。

## 无 GPU 重新生成报告

```bash
mkdir -p /tmp/g1a_check && tar xzf reports/provenance/g1a_runs_20261016.tar.gz -C /tmp/g1a_check
python scripts/g1a_report.py --runs /tmp/g1a_check/runs/g1a-mac-*/summary.json \
  --out /tmp/g1a_check/g1a.md --json /tmp/g1a_check/summary.json --manifest /tmp/g1a_check/manifest.json
diff /tmp/g1a_check/g1a.md reports/g1a_mac_m4max.md
```

旧 manifest 中的哈希没有被改写；这是恢复可追溯性，不是修改历史记录。

## 跑分后对 `src/attention_lab/` 的改动

2026-10-17 复审后，`src/attention_lab/runinfo.py` 的 `refuse_unsafe_env` 由字符串比较改为数值解析（拒绝 `"0.0"`、`"0e0"` 等零值写法）。该函数只在进程启动时检查环境变量，不参与任何计时或注意力计算；因此 G1a 与补测 manifest 中记录的 `runinfo.py` 哈希与当前文件不同，旧 manifest 不改。已有运行都没有设置这些变量，这一遗漏没有影响任何已报告的耗时。

# G1b 证据链归档（PR #2）

PR #2 评审指出：`scripts/g1b_report.py` 默认读取被 gitignore 的 `runs/` 目录，干净检出无法重生报告。本节把 G1b 的报告输入与跑分时的源码快照归档。

## 内容

| 文件 | 作用 |
|---|---|
| `g1b_runs_20261017.tar.gz` | 数值 run 四个（首轮按预设未通过的 `…T202211Z-formal`、登记修订判据后的 `…T202822Z-formal2`、加入同进度表比较的 `…T204139Z-formal3`、NaN 检查修复后的 `…-formal4`）、正式测速 run `g1b-mac-20261016T202847Z-formal`（含 steps.jsonl、profiles.txt）、显式阶段修复后的最小对照 run（tag `phasefix`）、日志 |
| `p0p1_runs_20261017.tar.gz` | PR #3 的 P0 / P1 正式 run（各两个独立进程）与冒烟 run 的 summary / jsonl / manifest；只做完整性校验，其报告在 PR #3 |
| `g1b/qwen_wrap_at_formal_decode_run.py` | 正式测速时的 `src/attention_lab/qwen_wrap.py`（commit `c95e097`），SHA-256 与测速 manifest 一致 |
| `g1b/qwen_wrap_post_run_edit.diff` | 测速后到提交之间对该文件的全部改动：显式阶段状态（评审第 1 条）；A 的解码读取与 D 的核调用不变 |
| `g1b/model_decode_bench_at_formal_decode_run.py`、`g1b/model_decode_bench_post_run_edit.diff` | 测速脚本快照与之后的唯一改动（manifest 记录 `HF_HUB_OFFLINE`） |
| `g1b/model_numerics_at_formal_run.py`、`…_formal2_run.py`、`…_formal3_run.py` | 三个早期数值 run 各自运行时的脚本版本（`2ec049f`、`c95e097`、`bed0aaa`）；第四个 run 用提交中的当前版本 |
| `archive_manifest.json` | 以上文件的字节数与 SHA-256 |

## 核验

`python scripts/check_provenance.py` 同时覆盖 G1a 与 G1b：测速 manifest 记录的 `qwen_wrap.py` / `model_decode_bench.py` 哈希等于快照；提交文件与快照只差 diff 文件记录的行；数值 run 4 与附录 run 的 manifest 哈希等于提交文件；归档内三个早期数值 run 的 manifest 哈希等于各自快照；归档文件字节数与哈希未变。

## 无 GPU、不加载模型重新生成报告

```bash
mkdir -p /tmp/g1b_check && tar xzf reports/provenance/g1b_runs_20261017.tar.gz -C /tmp/g1b_check
python scripts/g1b_report.py \
  --numerics /tmp/g1b_check/runs/g1b-numerics-*-formal4/results.json \
  --runs /tmp/g1b_check/runs/g1b-mac-*/summary.json \
  --out /tmp/g1b_check/g1b.md --json /tmp/g1b_check/summary.json \
  --numerics-json /tmp/g1b_check/numerics.json --manifest /tmp/g1b_check/manifest.json
diff /tmp/g1b_check/g1b.md reports/g1b_mac_m4max.md
```

`g1b_report.py` 只读 JSON，不导入 torch、不加载模型、不启动 GPU。旧 run 的 manifest 没有被改写。

# G2-lite 证据链归档（PR #5）

## 内容

| 文件 | 作用 |
|---|---|
| `g2lite_runs_20261017.tar.gz` | 正式三遍四个 run（`g2lite-20261017T130007Z-pass1-A`、`…T130319Z-pass2-ABCD0`、`…T130551Z-pass3a-rewrites`、`…T130715Z-pass3b-move`）的 data.jsonl（含送入模型的 token id）、quality.jsonl（逐题回答与解析）、summary.json、manifest.json、运行时的 config.json，以及代表案例的 `positions/*.npz`（每层每组第一个生成 token 那一步的名单位图与逐步覆盖率）；冒烟、预检与校准 run 的 quality / summary / manifest / config（不含 data.jsonl）；日志 |
| `g1b/decode_at_g1a_g1b_runs.py`、`g1b/decode_post_g1b_edit.diff` | G1a 到 G1b 所有 run 使用的 `src/attention_lab/decode.py` 快照，与 G2-lite 对它的全部改动（B、C 的名单生成函数、共用取行核、`return_rows`）；D0 的核序列不变 |
| `g1b/qwen_wrap_at_g1b_numerics4_and_phasefix_runs.py`、`g1b/qwen_wrap_post_g1b_edit.diff` | G1b 数值 run 4 与附录 run 使用的 `qwen_wrap.py`（PR #2 合并版本）快照，与 G2-lite 对它的全部改动（选择器、每步均匀抽样流、名单记录） |

## 核验

`python scripts/check_provenance.py` 新增：G1b 的数值 run 4 与附录 run 的 manifest 哈希等于 PR #2 版本的 `qwen_wrap.py` 快照；G1a 与 G1b 全部 manifest 的 `decode.py` 哈希等于其快照；提交文件与两份快照只差记录的 diff；`reports/g2lite_manifest_mac_m4max.json` 中每个正式 run 的 manifest 对 `decode.py`、`qwen_wrap.py`、`g2lite_data.py`、`g2lite_eval.py`、`bench/g2lite_quality.py` 的哈希等于提交文件（正式 run 全部在提交 `af62df4` 的干净工作树上运行）。

## 无 GPU、不加载模型重新生成报告

```bash
mkdir -p /tmp/g2lite_check && tar xzf reports/provenance/g2lite_runs_20261017.tar.gz -C /tmp/g2lite_check
python scripts/g2lite_report.py --runs-dir /tmp/g2lite_check/runs \
  --out /tmp/g2lite_check/g2lite.md --json /tmp/g2lite_check/summary.json \
  --answers /tmp/g2lite_check/answers.json --manifest /tmp/g2lite_check/manifest.json
diff /tmp/g2lite_check/g2lite.md reports/g2lite_mac_m4max.md
```

`g2lite_report.py` 只读 JSON / JSONL 与 npz，不导入 torch、不加载模型。校准 run（`-calib-`）与冒烟 run（`-quick-`）被默认排除；它们的数字只出现在卡片的校准附录里。

校准与冒烟 run 是用会话临时目录里的候选配置启动的，归档副本中 summary.json / manifest.json 的 `args.config` 与 `command` 字段把该临时目录替换成了 `<scratchpad>`（它含本机用户名）；实际使用的配置原样归档为同目录的 config.json，正式 run 未做任何改动。

# G2-lite 表头重放证据链归档（分支 `task/g2-lite-replay-header`）

## 内容

| 文件 | 作用 |
|---|---|
| `g2lite_replay_runs_20261017.tar.gz` | 重放正式 run（`g2lite-*-replay-header`：B 重跑、BH、BM 在 24 份 T1 资料 × 2 档 ρ）与冒烟 run（`-quick-smoke-replay`）的 data.jsonl、quality.jsonl、summary.json、manifest.json、运行时的 config.json、全部请求的 `positions/*.npz`（每层每组第一个生成 token 那一步的名单位图，表头覆盖率由此算出），以及日志 |
| `g2lite/decode_at_g2lite_runs.py`、`g2lite/qwen_wrap_at_g2lite_runs.py`、`g2lite/g2lite_data_at_g2lite_runs.py`、`g2lite/g2lite_quality_at_g2lite_runs.py` | G2-lite 四个正式 run（commit `af62df4`）使用的四个源文件快照，SHA-256 与 `reports/g2lite_manifest_mac_m4max.json` 里每份 manifest 一致 |
| `g2lite/*_post_g2lite_edit.diff` | 重放对这四个文件的全部改动：`select_rows_recent_pinned`（B+H 名单）、`recent_pinned` 选择器与 `pinned_positions`、资料的 `header_span` 字段、诊断方法 BH / BM 与 `diagnostic_replay` 标记。A、B、C、D0 的名单生成与核序列不变 |

## 核验

`python scripts/check_provenance.py` 新增：G2-lite 四份正式 manifest 的 `decode.py` / `qwen_wrap.py` / `g2lite_data.py` / `bench/g2lite_quality.py` 哈希等于 `g2lite/` 快照（`g2lite_eval.py` 未改，仍对提交文件）；G1b 快照到 G2-lite 快照、G2-lite 快照到提交文件各自只差记录的 diff；`reports/g2lite_replay_header_manifest_mac_m4max.json` 中每个重放 run 的五个源文件哈希等于提交文件。

## 无 GPU、不加载模型重新生成重放报告

```bash
mkdir -p /tmp/g2lite_replay_check && tar xzf reports/provenance/g2lite_runs_20261017.tar.gz -C /tmp/g2lite_replay_check runs/g2lite-20261017T130319Z-pass2-ABCD0
tar xzf reports/provenance/g2lite_replay_runs_20261017.tar.gz -C /tmp/g2lite_replay_check
python scripts/g2lite_replay_report.py --runs-dir /tmp/g2lite_replay_check/runs \
  --reference /tmp/g2lite_replay_check/runs/g2lite-20261017T130319Z-pass2-ABCD0 \
  --out /tmp/g2lite_replay_check/replay.md --json /tmp/g2lite_replay_check/summary.json --manifest /tmp/g2lite_replay_check/manifest.json
diff /tmp/g2lite_replay_check/replay.md reports/g2lite_replay_header_mac_m4max.md
```

参照 run（第二遍的 A 与 B）来自上一节的归档；重放 run 的 `summary.json` 标 `diagnostic_replay`，`scripts/g2lite_report.py` 默认排除它们，旧报告不受影响。
