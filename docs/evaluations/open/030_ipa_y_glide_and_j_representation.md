---
ID: 030
Title: Auditoria: 'y' como glide / representação de 'j' em transcrições IPA
Type: improvement / investigation
Priority: high
Status: open
---

Resumo:
Investigar ocorrências de `y` no corpus (uso como glide palatal / semivogal) e validar se o símbolo correto deveria ser `j` ou outra notação IPA; produzir regras de normalização e exemplos de mapeamento.

Descrição:
- Levantar todas as linhas em `docs/data/*` e `dicts/pt-br.tsv` que usam `y` como token.
- Classificar usos: ditongo, semivogal inicial, marcação de palatalização, erro legado.
- Verificar correspondência ortográfica (ex.: `maio`, `Kaique`, `Brayan`) e contexto fonológico.
- Propor: (a) manter `y` com justificativa fonética; (b) normalizar para `j`; ou (c) mapear para token estruturado (`GLIDE_PALATAL`) com documentação.

Exemplos conhecidos:
- `maio` → `ˈ m a y . ʊ`
- `Kaique` → `k a y ˈ k ɪ`
- `Brayan` → `b ɾ a ˈ y ã`

Critérios de aceite:
- Script listado que extrai ocorrências e gera um CSV com contexto e contador de frequência.
- Decisão documentada (manter `y`, usar `j` ou outra) com justificativa fonética e impactos no treino/inferência.
- Atualização de `IPA_REFERENCE.md` com a decisão e exemplos antes/depois.

Próximos passos:
1. Executar extração e coletar evidências (who/when).
2. Redigir proposta de normalização.
3. Implementar normalização nos dicionários se aprovado.

