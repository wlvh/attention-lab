# 教学 Notebook

| Notebook | 内容 | 默认模式 |
|---|---|---|
| `00_orientation.ipynb` | 仓库导读、源材料校验、载荷预测工具 | 只读，标准库 |
| `01_environment_and_attention.ipynb` | G0 与 G1a 的实验课：取子集两种写法为何差 25 倍；少读 KV 为何仍可能更慢；分段相加为何不等于链式；预测与观察对照 | 只读：加载 `reports/` 的实测 JSON，展示同版本源码，CPU 小例子；不加载模型，不跑基准 |

已执行的 HTML 在 `../reports/`。从全新内核重跑并导出：

```bash
source .venv/bin/activate
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_environment_and_attention.ipynb
python -m jupyter nbconvert --to html notebooks/01_environment_and_attention.ipynb --output-dir reports
```

规范见 `../docs/TEACHING_NOTEBOOKS.md`。后续课程按关卡增加，不预先创建空 Notebook。
