"""Notebook presentation only: keep long evidence available in closed details.

No data loading, model execution, scoring or training lives here.
"""
from contextlib import contextmanager
from html import escape


def show_figure(fig, number, title):
    """Display an unchanged PNG and its caption at half the notebook content width."""
    from IPython.core.pylabtools import print_figure
    from IPython.display import HTML, display
    import matplotlib.pyplot as plt

    png = print_figure(fig, fmt="png", base64=True)
    if png is None:
        raise ValueError("Cannot display an empty teaching figure")
    label = escape(f"图 {number}｜{title}", quote=True)
    display(HTML(
        '<figure style="width:50%;max-width:50%;min-width:0;margin:24px auto;">'
        '<img style="display:block;width:100%;max-width:100%;min-width:0;height:auto;" '
        'alt="' + label + '" src="data:image/png;base64,' + png + '">'
        '<figcaption style="margin-top:10px;line-height:1.6;text-align:center;">'
        '<strong>' + label + '</strong></figcaption></figure>'
    ))
    plt.close(fig)


@contextmanager
def details(title):
    """Capture text, tables and figures, preserving them in an HTML details block."""
    from IPython.display import HTML, display
    from IPython.utils.capture import capture_output
    from nbconvert.filters.markdown import markdown2html

    def pre(text):
        return '<pre style="white-space:pre-wrap;overflow-wrap:anywhere">' + escape(text) + '</pre>'

    with capture_output() as captured:
        yield
    parts = [pre(captured.stdout)] if captured.stdout else []
    if captured.stderr:
        parts.append(pre(captured.stderr))
    for output in captured.outputs:
        data = output.data
        if 'text/html' in data:
            parts.append(data['text/html'])
        elif 'image/png' in data:
            parts.append('<img style="display:block;width:50%;max-width:50%;height:auto;margin:0 auto;" alt="' + escape(title, quote=True)
                         + '" src="data:image/png;base64,' + data['image/png'] + '">')
        elif 'text/markdown' in data:
            parts.append(markdown2html(data['text/markdown']))
        elif 'text/plain' in data:
            parts.append(pre(data['text/plain']))
    display(HTML('<details><summary>' + escape(title) + '</summary>\n'
                 + '\n'.join(parts) + '\n</details>'))
