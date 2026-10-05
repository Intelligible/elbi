"""The notebook HTML export renderer."""

from __future__ import annotations

from elbi.notebook_html import render


def _page(*cells: dict) -> str:
    return render({"cells": list(cells)}, title="T <1>", exported_at="2026-10-05")


def test_no_script_can_run_in_the_page() -> None:
    # Kernel HTML is embedded as-is, so the CSP is what keeps an output inert.
    page = _page(
        {
            "cell_type": "code",
            "source": "x",
            "outputs": [
                {
                    "output_type": "display_data",
                    "data": {
                        "text/html": "<img src=x onerror=alert(1)><script>1</script>"
                    },
                }
            ],
        }
    )
    assert "default-src 'none'" in page
    assert "script-src" not in page, "no script source may be allowed"


def test_source_and_text_are_escaped() -> None:
    page = _page(
        {
            "cell_type": "code",
            "source": "a < b",
            "outputs": [
                {"output_type": "stream", "name": "stdout", "text": ["</pre><b>"]},
                {
                    "output_type": "error",
                    "ename": "E",
                    "evalue": "<x>",
                    "traceback": [],
                },
            ],
        }
    )
    assert "a &lt; b" in page
    assert "&lt;/pre&gt;&lt;b&gt;" in page
    assert "E: &lt;x&gt;" in page
    assert "<title>T &lt;1&gt;</title>" in page


def test_markdown_cells_render() -> None:
    page = _page({"cell_type": "markdown", "source": ["# Heading\n", "*em*"]})
    assert "<h1>Heading</h1>" in page
    assert "<em>em</em>" in page


def test_richest_output_wins() -> None:
    page = _page(
        {
            "cell_type": "code",
            "source": "df",
            "execution_count": 3,
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": "PLAIN",
                        "text/html": "<table><tr><td>1</td></tr></table>",
                    },
                },
                {"output_type": "display_data", "data": {"image/png": "iVBORw0K"}},
            ],
        }
    )
    assert "<td>1</td>" in page and "PLAIN" not in page
    assert 'src="data:image/png;base64,iVBORw0K"' in page
    assert "[3]" in page
    assert "Code only" not in page


def test_traceback_colours_are_dropped() -> None:
    page = _page(
        {
            "cell_type": "code",
            "source": "1/0",
            "outputs": [
                {
                    "output_type": "error",
                    "ename": "ZeroDivisionError",
                    "evalue": "division by zero",
                    "traceback": [
                        "\x1b[0;31mZeroDivisionError\x1b[0m: division by zero"
                    ],
                }
            ],
        }
    )
    assert "\x1b" not in page
    assert "ZeroDivisionError: division by zero" in page
