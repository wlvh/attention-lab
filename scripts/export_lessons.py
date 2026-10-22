"""Export saved lessons with optional code/details folded; never execute cells."""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = r"""{% extends 'lab/index.html.j2' %}
{% block codecell %}
{% set reading = cell.metadata.get('attention_lab_reading', {}) %}
{% if reading.get('role') == 'support' %}
<details class="lesson-support" data-cell-id="{{ cell.id | escape_html }}" style="margin:12px 0;padding:10px;border:1px solid #dce3ec;border-radius:8px;">
<summary style="cursor:pointer;color:#475569;">{{ reading.get('title', '展开辅助检查：代码与记录') | escape_html }}</summary>
{{ super() }}
</details>
{% else %}
{{ super() }}
{% endif %}
{% endblock codecell %}
{% block input_group %}
{% set reading = cell.metadata.get('attention_lab_reading', {}) %}
{% if reading.get('role') != 'support' and cell.metadata.get('jupyter', {}).get('source_hidden', False) %}
<details class="lesson-call-code" style="margin:8px 0;">
<summary style="cursor:pointer;color:#64748b;">展开调用与展示代码</summary>
{{ super() }}
</details>
{% else %}
{{ super() }}
{% endif %}
{% endblock input_group %}
"""


def export_lesson(path: Path, output_dir: Path) -> Path:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    exporter = HTMLExporter(template_name="lab", raw_template=TEMPLATE)
    exporter.exclude_input = False
    html, _ = exporter.from_notebook_node(notebook, resources={"metadata": {"name": path.stem}})
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{path.stem}.html"
    target.write_text(html, encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lessons", nargs="+", type=int, choices=range(1, 8), default=list(range(1, 8)))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    for number in dict.fromkeys(args.lessons):
        matches = list((ROOT / "notebooks").glob(f"{number:02d}_*.ipynb"))
        if len(matches) != 1:
            parser.error(f"expected one notebook for lesson {number}; found {len(matches)}")
        print(export_lesson(matches[0], args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
