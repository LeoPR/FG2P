# 002 - Intervalo de Confianca do WER

Status: respondida

## Resultado

Implementado no codigo em `src/analyze_errors.py` com Wilson 95% CI no nivel de palavra.

Formato de saida:

- `WER: X.XX% [L.LL%, U.UU%]`

## Base tecnica

1. WER em G2P e proporcao de palavras com erro (`pred != ref`).
2. O IC e calculado com a mesma `wilson_ci(count, n)` usada para PER.

## Referencias internas

- `src/analyze_errors.py`
- `docs/article/FORMULAS.md` (secao de Wilson CI para WER)
- `docs/evaluations/README.md`