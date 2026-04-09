# STIL 2026 — Submission Guide

**Venue**: 17th Symposium on Information Technology and Human Language
**Local**: Cuiaba, MT, Brazil (co-located com BRACIS 2026)
**Datas**: 19-22 outubro 2026
**Deadline**: **20 abril 2026** (extendida) — URGENTE
**Site**: https://bracis.sbc.org.br/2026/stil/
**Submissao**: https://jems3.sbc.org.br/stil2026/

---

## Especificacoes

| Item | Valor |
|------|-------|
| Template | SBC (sbc-template.sty + sbc.bst) |
| Formato | A4, coluna unica, Times 12pt |
| Margens | Sup 3,5cm, Inf 2,5cm, Lat 3,0cm |
| Long paper | 10 paginas conteudo + referencias ilimitadas |
| Short paper | 6 paginas conteudo + referencias ilimitadas |
| Linguas | Portugues, Ingles, Espanhol |
| Review | Double-blind |
| Publicacao | Historicamente no ACL Anthology |
| Co-located | BRACIS 2026 |

## Requisitos Obrigatorios (STIL 2026)

- [ ] Secao de Limitacoes
- [ ] Declaracao de Etica
- [ ] Declaracao de uso de IA generativa (pagina extra permitida)
- [ ] Todos os autores com PhD ou 3+ publicacoes PLN devem servir como revisores

## Processo de Submissao

1. **Preparar PDF**: Compilar main.tex com `latexmk -pdf main`. PDF em build/main.pdf.
2. **Anonimizar**: Verificar que nao ha nomes, afiliacoes ou links para repositorio.
3. **Secoes obrigatorias**: Limitacoes + Etica + IA generativa (pagina extra permitida).
4. **Submeter via JEMS**: Acessar https://jems3.sbc.org.br/stil2026/, criar conta, upload do PDF, preencher metadados.
5. **Aguardar revisao**: Double-blind. Notificacao prevista ~julho 2026.
6. **Camera-ready**: Se aceito, submeter versao final com nomes e afiliacoes.
7. **Apresentacao**: Long papers = oral; short papers = poster.

## Anonimizacao (double-blind)

- Remover nomes de autores e afiliacoes
- Nao referenciar repositorio GitHub
- Usar linguagem neutra ("este trabalho", "nosso sistema" — ok em PT-BR)

## Politica de IA (SBC/BRACIS)

> "Authors who use an LLM in any part of the article writing process take full responsibility for all content, including checking for plagiarism and correcting all text. We suggest that this use be properly mentioned in the Acknowledgements section, without prejudice to the evaluation process."

STIL 2026 exige adicionalmente uma **secao dedicada** de disclosure de IA generativa (pagina extra permitida).

**Proibicao**: IA nao pode ser listada como autor (criterios de autoria BRACIS).

Fonte: [BRACIS 2026 CFP](https://bracis.sbc.org.br/2026/bracis/)

## Status do Draft

- [x] STIL_DRAFT.md — Draft completo em portugues (~7 paginas)
- [x] main.tex compilando sem erros (0 erros, 0 overfull)
- [x] sbc-template.sty + sbc.bst presentes
- [x] stil_refs.bib — 11 referencias
- [x] latexmkrc configurado (build/ para artefatos)
- [x] Anonimizado
- [x] Secao de Limitacoes inclusa
- [x] Declaracao de Etica inclusa
- [x] Declaracao de uso de IA generativa inclusa
- [ ] Revisao final do conteudo
- [ ] PDF gerado e revisado visualmente
- [ ] Submetido antes do deadline

## Compilacao

```bash
cd docs/article/publications/stil/
latexmk -pdf main
# PDF em build/main.pdf
```

## Arquivos

```
stil/
├── README.md          ← ESTE ARQUIVO
├── STIL_DRAFT.md      ← Draft fonte em markdown
├── main.tex           ← LaTeX final (SBC format)
├── stil_refs.bib      ← Referencias (subset de REFERENCES.bib)
├── sbc-template.sty   ← Template SBC (Hubner & Bordini 2001/2005)
├── sbc.bst            ← Estilo bibliografico SBC (apalike-based)
├── latexmkrc          ← Config de compilacao
└── build/             ← Artefatos intermediarios (gitignored)
    └── main.pdf       ← PDF gerado
```
