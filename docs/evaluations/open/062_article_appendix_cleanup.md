ID: 062
Title: §10+ Limpeza — separar guia de uso, refs internas, apendice
Type: documentation
Priority: Medium
Status: Open

## Problema

O final do ARTICLE.md (§10 em diante) mistura conteudo de artigo cientifico
com documentacao tecnica do projeto:

1. §10 "Guia de Uso: inference_light.py" — ja marcado como "nao publicar"
   mas ainda presente no corpo do documento
2. "Documentacao Complementar" (linha 1333) — tabela com refs internas
   (DA_LOSS_ANALYSIS.md, EXPERIMENTS.md, PIPELINE.md, etc.)
3. "Referencias" (linha 1348) — indice manual que aponta para REFERENCES.bib

## Acao

### A) §10 — Guia de uso
Ja marcado como apendice tecnico. Manter como esta no ARTICLE.md (meta-artigo).
Garantir que derivados (ICASSP, TASLP) NAO incluem esta secao.

### B) Documentacao Complementar
Remover tabela de documentos internos do corpo do artigo.
Mover para nota de rodape ou comentario HTML no ARTICLE.md.
Razao: DA_LOSS_ANALYSIS.md, PIPELINE.md etc. sao docs internos do repositorio,
nao referencias que um leitor externo pode acessar.

### C) Referencias
A secao atual e um indice manual apontando para REFERENCES.bib.
Para o meta-artigo esta ok. Para derivados, substituir por bibliografia LaTeX real.
Verificar: a referencia @yao2015sequence foi removida do texto (C0-D, ticket 049)
mas pode ainda estar no indice manual.

## Verificacao

- [ ] §10 continua marcado como "nao publicar"
- [ ] Tabela de documentacao complementar removida do corpo ou marcada
- [ ] Secao de referencias nao contem entradas removidas (ex: @yao2015sequence)

Dependencias: ticket 049 (C0/C1 concluido)
