"""
Parser para docs/presentation/PRESENTATION.md.

Extrai metadados YAML do front matter e conteúdo estruturado de cada slide
(títulos, subtítulos, tabelas, blocos de código, bullets, texto).

Uso:
    from src.reporting.presentation_parser import parse_presentation
    from pathlib import Path

    data = parse_presentation(Path("docs/18_APRESENTACAO.md"))
    meta  = data["meta"]    # author, professor, institution, course, year, sota_per, ...
    slides = data["slides"] # lista de dicts, um por slide
"""

import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Front matter YAML
# ---------------------------------------------------------------------------

def _parse_yaml_frontmatter(text: str) -> dict:
    """
    Extrai campos do bloco YAML entre os dois primeiros '---'.
    Usa regex simples (sem depender de pyyaml) para pares key: "value" ou key: value.
    Campos com '#' no início da linha são comentários e são ignorados.
    """
    # Encontrar o bloco entre o primeiro e o segundo '---'
    match = re.match(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
    if not match:
        return {}

    frontmatter = match.group(1)
    # Campos de configuração do Marp — não são dados da apresentação
    _MARP_SYSTEM_KEYS = {"marp", "theme", "paginate", "backgroundColor", "style", "class", "size"}

    meta = {}

    for line in frontmatter.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Interromper no bloco 'style: |' (CSS multi-linha)
        if line.startswith('style:'):
            break
        kv_match = re.match(r'^(\w+)\s*:\s*"?([^"#\n]+)"?\s*$', line)
        if kv_match:
            key = kv_match.group(1).strip()
            if key in _MARP_SYSTEM_KEYS:
                continue
            val = kv_match.group(2).strip().strip('"')
            meta[key] = val

    return meta


# ---------------------------------------------------------------------------
# Divisão em slides
# ---------------------------------------------------------------------------

def _split_slides(text: str) -> list[str]:
    """
    Divide o documento em slides, ignorando o bloco YAML inicial.
    O front matter ocupa os primeiros dois '---'; o terceiro em diante separa slides.
    Retorna lista de strings (uma por slide), sem os separadores '---'.
    """
    # Encontrar fim do front matter
    fm_end = re.search(r'^---\s*\n.*?\n---\s*\n', text, re.DOTALL)
    if not fm_end:
        body = text
    else:
        body = text[fm_end.end():]

    # Dividir pelo separador de slide '---' em linha própria
    raw_slides = re.split(r'\n---\s*\n', body)
    # Remover slides vazios
    return [s.strip() for s in raw_slides if s.strip()]


# ---------------------------------------------------------------------------
# Extração de elementos por slide
# ---------------------------------------------------------------------------

def _extract_title(text: str) -> str:
    """Primeiro cabeçalho # de nível 1 ou 2."""
    match = re.search(r'^#{1,2}\s+(.+)$', text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _extract_subtitle(text: str) -> str:
    """Segundo cabeçalho (nível 2 ou 3) após o título, se existir."""
    headers = re.findall(r'^#{2,3}\s+(.+)$', text, re.MULTILINE)
    return headers[1].strip() if len(headers) > 1 else (headers[0].strip() if headers else "")


def _extract_tables(text: str) -> list[dict]:
    """
    Extrai todas as tabelas Markdown do slide.
    Cada tabela é um dict {"headers": [...], "rows": [[...], ...]}.
    Remove negrito (**) e backticks inline dos valores para texto limpo.
    """
    tables = []
    # Uma tabela Markdown é uma sequência de linhas começando com |
    table_blocks = re.findall(
        r'(?:^\|.+\|\s*\n)+',
        text,
        re.MULTILINE,
    )
    for block in table_blocks:
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        # Linha 0: cabeçalhos; Linha 1: separador (---); Linha 2+: dados
        header_line = lines[0]
        separator_line = lines[1] if len(lines) > 1 else ""
        # Verificar que linha 1 é separador (contém apenas -, | e :)
        if not re.match(r'^[\|\-:\s]+$', separator_line):
            continue
        headers = _parse_table_row(header_line)
        rows = [_parse_table_row(line) for line in lines[2:] if line.startswith('|')]
        if headers:
            tables.append({"headers": headers, "rows": rows})
    return tables


def _parse_table_row(line: str) -> list[str]:
    """Divide uma linha de tabela Markdown em células, removendo formatação."""
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    cleaned = []
    for cell in cells:
        # Remover **bold**, *italic*, `backticks` e ← →
        cell = re.sub(r'\*\*(.+?)\*\*', r'\1', cell)
        cell = re.sub(r'\*(.+?)\*', r'\1', cell)
        cell = re.sub(r'`(.+?)`', r'\1', cell)
        cleaned.append(cell.strip())
    return cleaned


def _extract_code_blocks(text: str) -> list[str]:
    """
    Extrai todos os blocos ``` ... ``` do slide.
    Retorna lista de strings (conteúdo bruto, sem as marcações de backtick).
    """
    return re.findall(r'```[^\n]*\n(.*?)```', text, re.DOTALL)


def _extract_bullets(text: str) -> list[str]:
    """
    Extrai linhas de bullet list (começando com - ou *) fora de blocos de código.
    """
    # Remover blocos de código antes de extrair bullets
    clean = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remover tabelas
    clean = re.sub(r'(?:^\|.+\|\s*\n)+', '', clean, flags=re.MULTILINE)
    bullets = re.findall(r'^[\-\*]\s+(.+)$', clean, re.MULTILINE)
    return [b.strip() for b in bullets]


def _extract_prose(text: str) -> str:
    """
    Extrai texto "livre" (não título, não tabela, não código, não bullet).
    Útil para parágrafos explicativos.
    """
    clean = text
    # Remover blocos de código
    clean = re.sub(r'```.*?```', '', clean, flags=re.DOTALL)
    # Remover linhas de tabela
    clean = re.sub(r'^\|.+\|\s*$', '', clean, flags=re.MULTILINE)
    # Remover cabeçalhos
    clean = re.sub(r'^#{1,3}\s+.+$', '', clean, flags=re.MULTILINE)
    # Remover bullets
    clean = re.sub(r'^[\-\*]\s+.+$', '', clean, flags=re.MULTILINE)
    # Compactar espaços
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Interface pública
# ---------------------------------------------------------------------------

def parse_presentation(md_path: Path) -> dict[str, Any]:
    """
    Lê um arquivo Markdown de apresentação (Marp) e retorna:
    {
        "meta": {
            "author": str,
            "professor": str,
            "institution": str,
            "course": str,
            "year": str,
            "sota_per": str,
            "sota_wer": str,
            "test_words": str,
            "oov_accuracy": str,
        },
        "slides": [
            {
                "index": int,         # 0-based
                "title": str,
                "subtitle": str,
                "tables": [{"headers": [...], "rows": [[...], ...]}, ...],
                "code_blocks": [str, ...],
                "bullets": [str, ...],
                "prose": str,
                "raw": str,           # texto original do slide (para debug)
            },
            ...
        ]
    }
    """
    text = md_path.read_text(encoding="utf-8")

    meta = _parse_yaml_frontmatter(text)
    raw_slides = _split_slides(text)

    slides = []
    for i, raw in enumerate(raw_slides):
        slides.append({
            "index": i,
            "title": _extract_title(raw),
            "subtitle": _extract_subtitle(raw),
            "tables": _extract_tables(raw),
            "code_blocks": _extract_code_blocks(raw),
            "bullets": _extract_bullets(raw),
            "prose": _extract_prose(raw),
            "raw": raw,
        })

    return {"meta": meta, "slides": slides}


# ---------------------------------------------------------------------------
# CLI — diagnóstico / teste rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    md_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/18_APRESENTACAO.md")

    if not md_file.exists():
        print(f"Arquivo não encontrado: {md_file}")
        sys.exit(1)

    data = parse_presentation(md_file)

    print("\n=== METADADOS ===")
    for k, v in data["meta"].items():
        print(f"  {k}: {v}")

    print(f"\n=== SLIDES ({len(data['slides'])} total) ===")
    for s in data["slides"]:
        tables_info = f"{len(s['tables'])} tab." if s["tables"] else ""
        code_info   = f"{len(s['code_blocks'])} code" if s["code_blocks"] else ""
        bullet_info = f"{len(s['bullets'])} bullets" if s["bullets"] else ""
        parts = [x for x in [tables_info, code_info, bullet_info] if x]
        extras = f"  [{', '.join(parts)}]" if parts else ""
        title_display = s["title"] or s["subtitle"] or "(sem título)"
        print(f"  [{s['index']:2d}] {title_display}{extras}")
