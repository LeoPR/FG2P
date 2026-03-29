ID: 046
Title: Agregação PT-BR multi-arquivo (base + overlays)
Type: feature
Priority: High
Status: Open

Descrição:
Permitir que o dataset PT-BR seja composto por múltiplos arquivos em camadas (canônico + regionais/dialetos/domínios), sem quebrar o modo monolítico atual.

Objetivo:
- Suportar expansão incremental do `pt-br.tsv` com novos dicionários.
- Manter determinismo de merge e rastreabilidade.

Escopo:
- Definir ordem de resolução de camadas:
  1. base canônica,
  2. overlays regionais,
  3. overlays de domínio (opcional).
- Definir política para conflitos de entrada (prioridade, log de override, auditoria).
- Manter fallback para o arquivo único legado.

Critérios de aceite:
- Merge determinístico reproduzível.
- Modo legado continua funcional sem alteração obrigatória de config.
- Evidência de compatibilidade com gate PT-BR (037).

Dependências:
- 035, 037, 043, 044
