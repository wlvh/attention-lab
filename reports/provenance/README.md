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
