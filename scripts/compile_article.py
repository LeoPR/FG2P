#!/usr/bin/env python3
"""
Compila ARTICLE.md + satélites em DOCX (via pandoc) e HTML.

Uso:
    python scripts/compile_article.py          # gera DOCX
    python scripts/compile_article.py --html   # gera HTML
    python scripts/compile_article.py --both   # gera os dois

Saídas:
    docs/FG2P_Report.docx
    docs/FG2P_Consolidated_Report.html

Requer: pandoc (instalado), python-docx, markdown2
"""

import argparse
import re
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
ARTICLE_PATH = ROOT_DIR / "docs" / "article" / "ARTICLE.md"
OUTPUT_DOCX = ROOT_DIR / "docs" / "FG2P_Report.docx"
OUTPUT_HTML = ROOT_DIR / "docs" / "FG2P_Consolidated_Report.html"

SATELLITES = [
    ("DA_LOSS_ANALYSIS.md", "Apêndice A: Análise Aprofundada da Phonetic DA Loss"),
    ("EXPERIMENTS.md", "Apêndice B: Log Completo de Experimentos"),
    ("PIPELINE.md", "Apêndice C: Pipeline Técnico"),
    ("ORIGINALITY_ANALYSIS.md", "Apêndice D: Análise de Originalidade"),
    ("GLOSSARY.md", "Apêndice E: Glossário Didático"),
    ("../linguistics/PHONOLOGICAL_ANALYSIS.md", "Apêndice F: Análise Fonológica"),
]

STRIP_PATTERNS = [
    r'^\*\*Status dos experimentos\*\*',
]


# ── Consolidação de markdown ─────────────────────────────────────────────────

def read_file(path):
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  ! Erro lendo {path.name}: {e}")
        return ""


def adjust_headings(content, offset):
    lines = content.split("\n")
    result = []
    for line in lines:
        m = re.match(r'^(#+)', line)
        if m:
            n = len(m.group(1))
            line = "#" * (n + offset) + line[n:]
        result.append(line)
    return "\n".join(result)


def strip_metadata(content):
    lines = content.split("\n")
    return "\n".join(
        l for l in lines
        if not any(re.match(p, l.strip()) for p in STRIP_PATTERNS)
    )


def consolidate():
    print("Consolidando documentos...")
    content = read_file(ARTICLE_PATH)
    content = strip_metadata(content)
    content += "\n\n---\n\n# Documentação Complementar\n\n"

    for doc_name, title in SATELLITES:
        doc_path = (ARTICLE_PATH.parent / doc_name).resolve()
        if doc_path.exists():
            print(f"  + {doc_name}")
            sat = read_file(doc_path)
            sat = "\n".join(sat.split("\n")[1:]).lstrip()
            sat = adjust_headings(sat, 1)
            content += f"\n## {title}\n\n{sat}\n\n---\n\n"
        else:
            print(f"  - Nao encontrado: {doc_path.name}")

    return content


# ── Geração DOCX via pandoc ──────────────────────────────────────────────────

def generate_docx(md_content):
    print("\nGerando DOCX com pandoc...")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", encoding="utf-8", delete=False
    ) as f:
        # Adiciona metadados YAML para pandoc (título, data, TOC)
        metadata = (
            "---\n"
            "title: 'FG2P — Phonetic Distance-Aware Loss'\n"
            f"date: '{date.today().isoformat()}'\n"
            "toc: true\n"
            "toc-depth: 3\n"
            "lang: pt-BR\n"
            "---\n\n"
        )
        f.write(metadata + md_content)
        tmp_path = Path(f.name)

    pandoc_path = Path.home() / "AppData/Local/Pandoc/pandoc.exe"
    pandoc_cmd = str(pandoc_path) if pandoc_path.exists() else "pandoc"

    cmd = [
        pandoc_cmd,
        str(tmp_path),
        "-o", str(OUTPUT_DOCX),
        "--from", "markdown+tex_math_dollars",
        "--to", "docx",
        "--toc",
        "--toc-depth=2",
        "--number-sections",
        "--highlight-style=tango",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    tmp_path.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"Erro pandoc: {result.stderr}")
        return False

    print(f"DOCX gerado: {OUTPUT_DOCX}")
    return True


# ── Geração HTML ─────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }

body { font-family: Georgia, serif; font-size: 15px; line-height: 1.7;
       color: #1a1a1a; display: flex; min-height: 100vh; }

/* ── Sidebar ── */
#sidebar {
    width: 280px; min-width: 280px;
    background: #1e2433; color: #cdd6f4;
    position: fixed; top: 0; left: 0; bottom: 0;
    overflow-y: auto; overflow-x: hidden;
    padding: 1.2em 0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
    z-index: 100;
}
#sidebar .sidebar-title {
    color: #89b4fa; font-size: 12px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .08em;
    padding: .4em 1.2em .8em; border-bottom: 1px solid #313654;
    margin-bottom: .5em;
}
#sidebar ul { list-style: none; }
#sidebar ul li a {
    display: block; padding: .28em 1.2em;
    color: #cdd6f4; text-decoration: none;
    border-left: 3px solid transparent;
    transition: background .15s, border-color .15s;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
#sidebar ul li a:hover  { background: #2a3150; color: #fff; }
#sidebar ul li a.active { background: #2a3150; border-left-color: #89b4fa; color: #cdd6f4; }

/* h1 entries */
#sidebar ul li.lvl1 > a { font-weight: 600; color: #b4befe; padding-top: .5em; }
/* h2 entries */
#sidebar ul li.lvl2 > a { padding-left: 2em; }
/* h3 entries */
#sidebar ul li.lvl3 > a { padding-left: 3em; font-size: 12px; color: #a6adc8; }

/* ── Content ── */
#content {
    margin-left: 280px;
    padding: 2.5em 3.5em 4em;
    max-width: 900px;
    width: 100%;
}

h1 { font-size: 1.9em; color: #1e3a6e; margin: 1.6em 0 .5em;
     border-bottom: 2px solid #0066cc; padding-bottom: .3em; }
h1:first-child { margin-top: 0; }
h2 { font-size: 1.4em; color: #0050a0; margin: 2em 0 .4em; }
h3 { font-size: 1.15em; color: #0066cc; margin: 1.4em 0 .3em; }
h4 { font-size: 1em; color: #444; margin: 1em 0 .2em; }

p  { margin: .6em 0; text-align: justify; }

code { background: #f0f0f0; padding: 2px 5px; border-radius: 3px;
       font-family: 'Cascadia Code','Consolas',monospace; font-size: .88em; }
pre  { background: #f5f5f5; padding: 1em 1.2em; border-left: 3px solid #0066cc;
       border-radius: 0 4px 4px 0; overflow-x: auto; margin: 1em 0; }
pre code { background: none; padding: 0; }

table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: .93em; }
th { background: #0060bb; color: #fff; padding: .45em .7em; text-align: left; }
td { border: 1px solid #d0d8e8; padding: .4em .7em; }
tr:nth-child(even) td { background: #f7f9fd; }

blockquote { border-left: 4px solid #89b4fa; margin: 1em 0 1em 1em;
             padding: .5em 1em; background: #f5f7ff; color: #444; }
hr  { border: none; border-top: 1px solid #dde; margin: 2em 0; }

footer { margin-top: 4em; padding-top: 1em; border-top: 1px solid #dde;
         color: #888; font-size: .82em; text-align: center; }

@media print {
    #sidebar { display: none; }
    #content { margin-left: 0; max-width: 100%; padding: 0; }
    h1 { page-break-before: always; }
    h1:first-child { page-break-before: avoid; }
}
"""

SIDEBAR_JS = """
(function () {
  const links = document.querySelectorAll('#sidebar a');
  const headings = document.querySelectorAll('#content h1,#content h2,#content h3');

  function activate(id) {
    links.forEach(l => {
      l.classList.toggle('active', l.getAttribute('href') === '#' + id);
    });
  }

  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) activate(e.target.id); });
  }, { rootMargin: '-10% 0px -80% 0px' });

  headings.forEach(h => { if (h.id) obs.observe(h); });
})();
"""


def protect_math(text):
    placeholders = {}

    def store(m):
        key = f"MATHPLACEHOLDER{len(placeholders)}ENDMATH"
        placeholders[key] = m.group(0)
        return key

    text = re.sub(r'\$\$.+?\$\$', store, text, flags=re.DOTALL)
    text = re.sub(r'\$[^$\n]+?\$', store, text)
    return text, placeholders


def restore_math(html, placeholders):
    for key, value in placeholders.items():
        html = html.replace(key, value)
    return html


def slugify(text):
    import unicodedata
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    return text.strip('-')


def build_toc(markdown_text):
    entries = []
    seen_slugs = {}
    for line in markdown_text.split("\n"):
        m = re.match(r'^(#{1,2})\s+(.+)', line)
        if not m:
            continue
        level = len(m.group(1))
        title_clean = re.sub(r'\*+([^*]+)\*+', r'\1', m.group(2).strip())
        title_clean = re.sub(r'`([^`]+)`', r'\1', title_clean)
        slug = slugify(title_clean)
        count = seen_slugs.get(slug, 0)
        seen_slugs[slug] = count + 1
        if count:
            slug = f"{slug}-{count}"
        entries.append((level, title_clean, slug))

    if not entries:
        return "", {}

    slug_map = {title: slug for _, title, slug in entries}

    items = []
    for level, title, slug in entries:
        items.append(f'<li class="lvl{level}"><a href="#{slug}">{title}</a></li>')

    sidebar_html = (
        '<nav id="sidebar">\n'
        '  <div class="sidebar-title">FG2P — Navegacao</div>\n'
        '  <ul>\n'
        + "\n".join(f"    {i}" for i in items)
        + "\n  </ul>\n</nav>"
    )
    return sidebar_html, slug_map


def inject_anchors(html_body, slug_map):
    def add_id(m):
        tag, content = m.group(1), m.group(2)
        title_clean = re.sub(r'<[^>]+>', '', content).strip()
        slug = slug_map.get(title_clean)
        if slug:
            return f'<{tag} id="{slug}">{content}</{tag}>'
        return m.group(0)
    return re.sub(r'<(h[1-3])>(.*?)</h[1-3]>', add_id, html_body, flags=re.DOTALL)


def generate_html(md_content):
    import markdown2
    print("\nGerando HTML...")
    sidebar_html, slug_map = build_toc(md_content)
    md_safe, placeholders = protect_math(md_content)
    html_body = markdown2.markdown(md_safe, extras=["tables", "fenced-code-blocks"])
    html_body = restore_math(html_body, placeholders)
    html_body = inject_anchors(html_body, slug_map)

    gen_date = date.today().isoformat()
    footer = f'<footer>Documento gerado em {gen_date} — Projeto FG2P</footer>'

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>FG2P — Relatorio Consolidado</title>
  <style>{CSS}</style>
  <script>
    MathJax = {{ tex: {{ inlineMath: [['$','$']], displayMath: [['$$','$$']] }}, svg: {{ fontCache: 'global' }} }};
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
{sidebar_html}
<div id="content">
{html_body}
{footer}
</div>
<script>{SIDEBAR_JS}</script>
</body>
</html>"""

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"HTML gerado: {OUTPUT_HTML}")
    return True


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not ARTICLE_PATH.exists():
        print(f"Artigo nao encontrado: {ARTICLE_PATH}")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--html", action="store_true", help="Gera HTML")
    parser.add_argument("--both", action="store_true", help="Gera DOCX + HTML")
    args = parser.parse_args()

    md = consolidate()

    if args.html:
        generate_html(md)
    elif args.both:
        generate_docx(md)
        generate_html(md)
    else:
        generate_docx(md)
