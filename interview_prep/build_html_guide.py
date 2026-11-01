"""Generate a unified, print-ready HTML interview preparation guide from markdown files."""

from __future__ import annotations

import html
import re
from pathlib import Path

PREP_DIR = Path(__file__).resolve().parent


def markdown_to_html(md_text: str) -> str:
    """Simple parser for common markdown elements without external dependencies."""

    # Convert code blocks
    def replace_code_block(match):
        lang = match.group(1) or ""
        code = html.escape(match.group(2).strip())
        return f'<pre><code class="language-{lang}">{code}</code></pre>'

    md_text = re.sub(r"```(\w+)?\n([\s\S]*?)```", replace_code_block, md_text)

    # Tables
    lines = md_text.splitlines()
    in_table = False
    html_lines = []
    table_rows = []

    for line in lines:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            row = [c.strip() for c in line.strip()[1:-1].split("|")]
            if all(set(c).issubset({"-", ":", " "}) for c in row if c):
                continue  # Header separator
            table_rows.append(row)
            in_table = True
        else:
            if in_table:
                # Flush table
                html_lines.append('<div class="table-container"><table>')
                if table_rows:
                    html_lines.append(
                        "<thead><tr>"
                        + "".join(f"<th>{r}</th>" for r in table_rows[0])
                        + "</tr></thead>"
                    )
                    html_lines.append("<tbody>")
                    for r in table_rows[1:]:
                        html_lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
                    html_lines.append("</tbody>")
                html_lines.append("</table></div>")
                table_rows = []
                in_table = False
            html_lines.append(line)

    if in_table:
        html_lines.append('<div class="table-container"><table><tbody>')
        for r in table_rows:
            html_lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
        html_lines.append("</tbody></table></div>")

    content = "\n".join(html_lines)

    # Headers
    content = re.sub(r"^# (.*?)$", r"<h1>\1</h1>", content, flags=re.M)
    content = re.sub(r"^## (.*?)$", r"<h2>\1</h2>", content, flags=re.M)
    content = re.sub(r"^### (.*?)$", r"<h3>\1</h3>", content, flags=re.M)
    content = re.sub(r"^#### (.*?)$", r"<h4>\1</h4>", content, flags=re.M)

    # Blockquotes
    content = re.sub(r"^> (.*?)$", r"<blockquote>\1</blockquote>", content, flags=re.M)

    # Horizontal rules
    content = re.sub(r"^---$", r"<hr/>", content, flags=re.M)

    # Formatting
    content = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", content)
    content = re.sub(r"\*(.*?)\*", r"<em>\1</em>", content)
    content = re.sub(r"`(.*?)`", r"<code>\1</code>", content)

    # Lists
    content = re.sub(r"^\s*-\s+(.*?)$", r"<li>\1</li>", content, flags=re.M)

    return content


def build_unified_html():
    qa = (PREP_DIR / "ML_Ops_Live_Interview_QA.md").read_text(encoding="utf-8")
    cheat = (PREP_DIR / "ML_Ops_Live_Interview_Cheat_Sheet.md").read_text(encoding="utf-8")
    unknowns = (PREP_DIR / "ML_Ops_Live_Interview_Unknowns.md").read_text(encoding="utf-8")

    body_html = f"""
    <div class="page-break">
        <div class="doc-header">
            <h1>🚕 ML-Ops-Live — Master Technical Interview Guide</h1>
            <p class="subtitle">Complete Question Bank, Architecture Cheat Sheet & Risk Analysis</p>
            <p class="meta">Target Roles: AI Engineer | ML Engineer | MLOps Engineer | Backend</p>
        </div>
    </div>

    <div class="section-container">
        {markdown_to_html(cheat)}
    </div>

    <div class="page-break"></div>
    <div class="section-container">
        {markdown_to_html(unknowns)}
    </div>

    <div class="page-break"></div>
    <div class="section-container">
        {markdown_to_html(qa)}
    </div>
    """

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ML-Ops-Live — Technical Interview Preparation Guide</title>
    <style>
        @page {{
            size: A4;
            margin: 20mm;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #1f2328;
            background-color: #ffffff;
            line-height: 1.6;
            margin: 0 auto;
            max-width: 900px;
            padding: 30px;
        }}
        .doc-header {{
            text-align: center;
            border-bottom: 2px solid #0969da;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .doc-header h1 {{
            color: #0969da;
            font-size: 2.2rem;
            margin-bottom: 8px;
        }}
        .subtitle {{
            font-size: 1.15rem;
            color: #57606a;
            font-weight: 500;
            margin-bottom: 5px;
        }}
        .meta {{
            font-size: 0.9rem;
            color: #656d76;
        }}
        h1, h2, h3, h4 {{
            color: #1f2328;
            font-weight: 600;
        }}
        h1 {{
            font-size: 1.8rem;
            border-bottom: 1px solid #d0d7de;
            padding-bottom: 8px;
            margin-top: 35px;
        }}
        h2 {{
            font-size: 1.4rem;
            border-bottom: 1px solid #eaeef2;
            padding-bottom: 6px;
            margin-top: 25px;
            color: #0969da;
        }}
        h3 {{ font-size: 1.15rem; margin-top: 20px; }}
        h4 {{ font-size: 1.0rem; margin-top: 15px; color: #57606a; }}
        blockquote {{
            border-left: 4px solid #0969da;
            padding: 8px 16px;
            margin: 15px 0;
            background-color: #f6f8fa;
            color: #424a53;
        }}
        code {{
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
            font-size: 0.88em;
            background-color: #f6f8fa;
            border: 1px solid #d0d7de;
            border-radius: 4px;
            padding: 2px 5px;
            color: #0969da;
        }}
        pre {{
            background-color: #f6f8fa;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            padding: 14px;
            overflow-x: auto;
            font-size: 0.88em;
            line-height: 1.45;
        }}
        pre code {{
            border: none;
            padding: 0;
            background: none;
            color: #1f2328;
        }}
        .table-container {{
            overflow-x: auto;
            margin: 15px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 0.9em;
        }}
        th, td {{
            border: 1px solid #d0d7de;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background-color: #f6f8fa;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #fafbfc;
        }}
        hr {{
            border: 0;
            height: 1px;
            background: #d0d7de;
            margin: 25px 0;
        }}
        @media print {{
            body {{
                padding: 0;
                max-width: 100%;
            }}
            .page-break {{
                page-break-before: always;
            }}
            pre, blockquote, table {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    {body_html}
</body>
</html>
"""
    output_path = PREP_DIR / "ML_Ops_Live_Interview_Guide.html"
    output_path.write_text(full_html, encoding="utf-8")
    print(f"Generated printable HTML guide: {output_path} ({len(full_html):,} bytes)")


if __name__ == "__main__":
    build_unified_html()
