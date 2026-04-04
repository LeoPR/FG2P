ID: 064
Title: Hierarquia de fontes — rastreabilidade ARTICLE.md vs README vs EXPERIMENTS.md
Type: documentation
Priority: High
Status: Open

## Problema

Informacoes sobre experimentos, metricas e resultados aparecem em multiplos locais:

| Local | Conteudo | Risco |
|-------|----------|-------|
| ARTICLE.md | Manuscrito cientifico completo | Pode estar desatualizado se README mudar |
| README.md | Resumo publico do projeto | Pode ter numeros diferentes do ARTICLE |
| EXPERIMENTS.md | Log cronologico de todos os experimentos | Fonte mais detalhada |
| models/*_metadata.json | Dados brutos de cada experimento | Fonte mais confiavel numericamente |

Exemplo concreto: se um PER muda no EXPERIMENTS.md mas ninguem atualiza
ARTICLE.md e README.md, os tres documentos ficam inconsistentes.

## Hierarquia proposta (fonte de verdade)

```
NIVEL 1 — models/*_metadata.json + resultados brutos
  (numeros exatos, gerados pelo codigo, nunca editados manualmente)
  ↓ alimenta
NIVEL 2 — EXPERIMENTS.md
  (log completo de todos os experimentos, com configs e datas)
  ↓ alimenta
NIVEL 3 — ARTICLE.md
  (manuscrito cientifico, seleciona e interpreta os dados)
  ↓ alimenta
NIVEL 4 — README.md
  (resumo publico, cita numeros do ARTICLE.md)
  ↓ alimenta
NIVEL 5 — Derivados (ICASSP_DRAFT.md, TASLP, etc.)
  (adaptam do ARTICLE.md para formato/venue especifico)
```

Regra: um numero so muda de cima pra baixo. Se EXPERIMENTS.md corrige um PER,
ARTICLE.md e README.md devem ser atualizados. Nunca o contrario.

## Acao

1. Documentar esta hierarquia em docs/INDEX.md ou docs/article/ARTICLE.md §0
2. Verificar se ha inconsistencias atuais entre README e ARTICLE nos numeros-chave
3. Quando corrigir algo no ARTICLE.md, verificar se README.md precisa de update

## Verificacao

- [ ] Hierarquia documentada em local visivel
- [ ] Numeros-chave (PER, WER, N palavras) consistentes entre README e ARTICLE
- [ ] Derivados (ICASSP) consistentes com ARTICLE

Dependencias: nenhuma (pode ser feito em paralelo com qualquer outro ticket)
