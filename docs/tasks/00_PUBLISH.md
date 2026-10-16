# 任务00：核对已有公开远端

## 当前事实

所有者已经建立 `https://github.com/wlvh/attention-lab` 并明确选择公开。当前操作是导入并核验既有项目材料，不再是创建私有仓库。

## 本地接手

```bash
git clone https://github.com/wlvh/attention-lab.git
cd attention-lab
python3 scripts/check_sources.py
python3 scripts/check_repository.py
python3 -m unittest discover -s tests -v
git remote -v
git rev-parse HEAD
```

不需要运行 `scripts/publish_repo.py --execute`。该脚本保留历史用途，仍拒绝公开仓库；这不是当前项目的障碍。认证只在本机正常授权，不把 token 放入聊天或文件。

## 发布验收

远端名称正确、`private=false`、分支和提交可读；文件数量与内容清单一致；九个原始材料通过 SHA-256 与大小核验。空仓库或只有 README 不算完整上传。

后续只在任务分支实施，不改仓库可见性、不 force push、不替换无关文件、不自动启动 Pages。公开结果之前过滤真实用户名、主机名、绝对路径、访问令牌及无关私有信息。首轮执行入口为 `01_G0_G1A.md`。
