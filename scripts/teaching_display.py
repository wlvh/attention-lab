"""Notebook presentation only: keep long evidence available in closed details.

No data loading, model execution, scoring or training lives here.
"""
from contextlib import contextmanager
from html import escape


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
            parts.append('<img style="max-width:100%;height:auto" alt="' + escape(title, quote=True)
                         + '" src="data:image/png;base64,' + data['image/png'] + '">')
        elif 'text/markdown' in data:
            parts.append(markdown2html(data['text/markdown']))
        elif 'text/plain' in data:
            parts.append(pre(data['text/plain']))
    display(HTML('<details><summary>' + escape(title) + '</summary>\n'
                 + '\n'.join(parts) + '\n</details>'))
