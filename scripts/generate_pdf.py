#!/usr/bin/env python3
"""
Consolida ARTICLE.md + satélites em um único HTML pronto para impressão.
Uso: python scripts/generate_pdf.py
Saída: docs/FG2P_Consolidated_Report.html

Para gerar PDF: abra o HTML no navegador → Ctrl+P → Salvar como PDF

Requer: pip install markdown2
"""

import re
import unicodedata
from datetime import date
from pathlib import Path
import markdown2

ROOT_DIR = Path(__file__).parent.parent
ARTICLE_PATH = ROOT_DIR / "docs" / "article" / "ARTICLE.md"
OUTPUT_HTML = ROOT_DIR / "docs" / "FG2P_Consolidated_Report.html"

SATELLITES = [
    ("DA_LOSS_ANALYSIS.md", "Apêndice A: Análise Aprofundada da Phonetic DA Loss"),
    ("EXPERIMENTS.md", "Apêndice B: Log Completo de Experimentos"),
    ("PIPELINE.md", "Apêndice C: Pipeline Técnico"),
    ("ORIGINALITY_ANALYSIS.md", "Apêndice D: Análise de Originalidade"),
    ("GLOSSARY.md", "Apêndice E: Glossário Didático"),
    ("../linguistics/PHONOLOGICAL_ANALYSIS.md", "Apêndice F: Análise Fonológica"),
]

CSS = """
body { font-family: Georgia, serif; line-height: 1.6; color: #222; max-width: 900px; margin: 0 auto; padding: 2em; }
nav#toc { background: #f0f4ff; border: 1px solid #c0d0ee; border-radius: 4px; padding: 1em 1.5em; margin-bottom: 2em; }
nav#toc h2 { color: #003399; font-size: 1.1em; margin: 0 0 0.5em 0; }
nav#toc ol { margin: 0; padding-left: 1.4em; }
nav#toc li { margin: 0.2em 0; font-size: 0.92em; }
nav#toc a { color: #0055cc; text-decoration: none; }
nav#toc a:hover { text-decoration: underline; }
h1 { border-bottom: 2px solid #0066cc; padding-bottom: 0.3em; color: #003399; }
h2 { color: #0066cc; margin-top: 2em; }
h3 { color: #0077cc; }
code { background: #f4f4f4; padding: 2px 5px; font-family: monospace; font-size: 0.9em; }
pre { background: #f4f4f4; padding: 1em; border-left: 3px solid #0066cc; overflow-x: auto; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th { background: #0066cc; color: white; padding: 0.5em; }
td { border: 1px solid #ccc; padding: 0.5em; }
tr:nth-child(even) { background: #f9f9f9; }
blockquote { border-left: 4px solid #0066cc; margin: 1em 0; padding-left: 1em; color: #555; }
hr { border: none; border-top: 2px solid #ddd; margin: 2em 0; }
footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid #ddd; color: #888; font-size: 0.85em; }
@media print {
    nav#toc { display: none; }
    h1 { page-break-before: always; }
    h1:first-of-type { page-break-before: avoid; }
    pre, table { page-break-inside: avoid; }
}
"""


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


def strip_metadata_lines(content):
    """Remove linhas de status/experimentos do cabeçalho do artigo."""
    lines = content.split("\n")
    filtered = []
    for line in lines:
        if re.match(r'^\*\*Status dos experimentos\*\*', line.strip()):
            continue
        filtered.append(line)
    return "\n".join(filtered)


def consolidate():
    print("Consolidando documentos...")
    content = read_file(ARTICLE_PATH)
    content = strip_metadata_lines(content)
    # inclui as referências (não remove mais)
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
            print(f"  - Não encontrado: {doc_path.name}")

    return content


def protect_math(text):
    """Extrai blocos LaTeX antes do markdown2 processar (evita _t → <em>t</em>)."""
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
    """Gera um ID de anchor a partir do texto de heading."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    return text.strip('-')


def build_toc(markdown_text):
    """Extrai headings ## e ### e gera HTML de navegação."""
    entries = []
    seen_slugs = {}
    for line in markdown_text.split("\n"):
        m = re.match(r'^(#{2,3})\s+(.+)', line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        # Remove markdown inline (bold, etc.)
        title_clean = re.sub(r'\*+([^*]+)\*+', r'\1', title)
        slug = slugify(title_clean)
        # Garante slugs únicos
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
        indent = "  " if level == 3 else ""
        items.append(f'{indent}<li><a href="#{slug}">{title}</a></li>')

    toc_html = (
        '<nav id="toc">\n'
        '  <h2>Índice</h2>\n'
        '  <ol>\n'
        + "\n".join(f"    {i}" for i in items)
        + "\n  </ol>\n</nav>"
    )
    return toc_html, slug_map


def inject_anchors(html_body, slug_map):
    """Adiciona id= nos headings do HTML para os links do ToC funcionarem."""
    def add_id(m):
        tag = m.group(1)
        content = m.group(2)
        title_clean = re.sub(r'<[^>]+>', '', content).strip()
        slug = slug_map.get(title_clean)
        if slug:
            return f'<{tag} id="{slug}">{content}</{tag}>'
        return m.group(0)

    return re.sub(r'<(h[23])>(.*?)</h[23]>', add_id, html_body, flags=re.DOTALL)


if __name__ == "__main__":
    if not ARTICLE_PATH.exists():
        print(f"Artigo não encontrado: {ARTICLE_PATH}")
        exit(1)

    md = consolidate()
    toc_html, slug_map = build_toc(md)
    md_safe, placeholders = protect_math(md)
    html_body = markdown2.markdown(md_safe, extras=["tables", "fenced-code-blocks"])
    html_body = restore_math(html_body, placeholders)
    html_body = inject_anchors(html_body, slug_map)

    gen_date = date.today().isoformat()
    footer = f'<footer>Documento gerado em {gen_date} — Projeto FG2P</footer>'

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>FG2P — Relatório Consolidado</title>
  <style>{CSS}</style>
  <script>
    MathJax = {{ tex: {{ inlineMath: [['$','$']], displayMath: [['$$','$$']] }}, svg: {{ fontCache: 'global' }} }};
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
{toc_html}
{html_body}
{footer}
</body>
</html>"""

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nHTML gerado: {OUTPUT_HTML}")
    print("Para PDF: abra no navegador e use Ctrl+P -> Salvar como PDF")
