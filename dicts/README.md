dicts/ — Dicionários fonéticos e procedimento de normalização

Objetivo
- Reunir as fontes (ex.: `backups/ipa-dict/`) e fornecer um mecanismo claro e reprodutível para normalizar/atualizar os dicionários usados no pipeline.

Conteúdo esperado
- `pt-br.tsv` — dicionário principal (palavra\tphonemes\t...)
- `pt-br.normalized.tsv` — saída gerada pelo script de normalização
- `README.md` — (este arquivo) com instruções

Comandos úteis
- Normalizar o dicionário PT-BR (gera `pt-br.normalized.tsv` e relatório CSV):

```bash
python scripts/normalize_dicts.py --input dicts/pt-br.tsv --output dicts/pt-br.normalized.tsv --report reports/normalize_dicts.csv
```

Notas técnicas
- O script aplica Unicode NFC e converte o caractere ASCII `g` (U+0067) para o símbolo IPA `ɡ` (U+0261). Outras normalizações podem ser adicionadas em `scripts/normalize_dicts.py`.
- Licenças e origens: ver `backups/ipa-dict/README.md` (origem: open-dict-data/ipa-dict). Ao regenerar o dicionário, preserve metadados de licença.

Próximos passos sugeridos
- Adicionar testes unitários que verifiquem contagem de correções (ex.: 10.252 instâncias g->ɡ no histórico).
- Expor o script como CLI instalável (opcional) ou mover para `src/scripts/` conforme convenção do projeto.
