# Refatoração: Unified Mode System para Apresentações

**Data**: 2026-02-27
**Status**: ✅ COMPLETO E TESTADO

---

## Problema Original

Antes da refatoração, o sistema de apresentação tinha:
- ❌ **2 arquivos Markdown duplicados** (17_APRESENTACAO.md + 17_APRESENTACAO_COMPACTA.md)
- ❌ Sincronização manual → bugs inevitáveis (como descobrimos na seção Glossários)
- ❌ Lógica hardcoded duplicada em presentation_generator.py
- ❌ Impossível editar facilmente e decidir o que aparece aonde
- ❌ Flags CLI confusos (--compact, --duration, etc.)

---

## Solução Implementada: Unified Mode System

### Arquitetura

```
17_APRESENTACAO.md                ← UM arquivo com tags [modes: ...]
    ├─ [modes: full, compact]            ← Slides em ambas versões
    ├─ [modes: full]                     ← Slides apenas em full (detalhes)
    └─ [modes: compact]                  ← Slides apenas em compact (resumidos)
           ↓
 filter_markdown_by_mode(md, mode)       ← Parser inteligente
           ↓
 build_presentation_from_markdown()      ← Gerador unificado
           ↓
 results/fg2p_presentation.pptx          ← Output: 31 slides (full) ou 20 (compact)
```

### Sintaxe das Tags

```markdown
[modes: full, compact]
## Slide Compartilhado
Conteúdo que aparece em ambas versões...

---

[modes: full]
## Slide Apenas Full
Este slide aparece apenas em mode=full, ideal para detalhes e explanações...

---

[modes: compact]
## Slide Apenas Compact
Este slide aparece apenas em mode=compact (raramente usado, para variações)...
```

---

## Benefícios

✅ **Um arquivo único de verdade**
- Você edita em um lugar, as duas versões funcionam
- Sem duplicação, sem sincronização manual

✅ **Marcação semântica simples**
- Tags `[modes: ...]` deixam claro o propósito de cada slide
- Fácil de adicionar novos modos (ex: "mini" para 5 min)

✅ **Usuário tem controle total**
- Edite o MD e decida quais slides vão aonde
- Adicione `[modes: compact]` a um slide para removê-lo da versão full

✅ **Gerador inteligente e agnóstico**
- Não precisa saber nada sobre G2P ou apresentações
- Apenas segue as tags e gera PPTX

✅ **CLI clara e intuitiva**
- `--mode full` → 31 slides completos (29 min)
- `--mode compact` → 20 slides (10 min)
- Sem flags confusos

---

## Novo Workflow

### Editar a Apresentação

1. Abra `docs/presentation/PRESENTATION.md`
2. Marque slides com `[modes: ...]`:
   ```markdown
   [modes: full, compact]
   ## Meu Novo Slide
   Conteúdo...
   ```
3. Salve o arquivo

### Gerar a Apresentação

```bash
# Gerar versão completa (default)
python src/reporting/presentation_generator.py --mode full

# Gerar versão compacta (10 min)
python src/reporting/presentation_generator.py --mode compact

# Customizar saída
python src/reporting/presentation_generator.py --mode full -o minha_apresentacao.pptx
```

---

## Mudanças no Código

### Novo: Função de Filtering

```python
def filter_markdown_by_mode(markdown_text: str, mode: str) -> str:
    """
    Remove slides não-relevantes baseado em [modes: ...] tags.
    Preserva apenas slides com o modo solicitado.
    """
```

### Refatorado: Main Function

```python
parser.add_argument(
    "--mode",
    choices=["full", "compact"],
    default="full",
    help="Modo: 'full' (31 slides) ou 'compact' (20 slides, 10 min)"
)
```

### Refatorado: build_presentation_from_markdown

```python
def build_presentation_from_markdown(
    md_path: Path,
    output_path: Path,
    filtered_markdown: str = None  ← Novo parâmetro
) -> Path:
```

---

## Compatibilidade Retroativa

Os flags antigos ainda funcionam (com warnings):
```bash
# Funcionam, mas deprecados:
python src/reporting/presentation_generator.py --compact
python src/reporting/presentation_generator.py --duration 10
```

---

## Resultados de Teste

✅ **Modo Compact**: 20 slides · 10 minutos
✅ **Modo Full**: 31 slides · 29 slides (completo)
✅ **Arquivo PPTX válido**: Microsoft PowerPoint 2007+
✅ **Parser de tags**: Funcionando corretamente

---

## Próximos Passos (Opcionais)

1. **Deletar arquivos antigos** (quando confortável):
   ```bash
   rm docs/presentation/PRESENTATION.md
   rm docs/17_APRESENTACAO_COMPACTA.md
   ```

2. **Renomear arquivo merged** (após revisar):
   ```bash
   mv docs/presentation/PRESENTATION.md docs/presentation/PRESENTATION.md
   ```

3. **Adicionar novos modos** (ex: "mini" para 5 min):
   ```python
   choices=["full", "compact", "mini"]
   ```

4. **Documentar as tags** em um comment no início do MD

---

## Conclusão

O sistema é agora **mais simples, mais inteligente e mais fácil de manter**.
Você edita um arquivo, o gerador faz o resto. 🚀
