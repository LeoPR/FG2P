ID: 026
Title: Multilíngue & Tupi / Dialetos
Type: feature
Priority: High
Status: Open

## Papel deste ticket
Este ticket passa a ser o **EPIC** da iniciativa multilíngue.
Execução detalhada fica nos subtickets 035-041.

Subtickets vinculados:
- 035 - Estrutura de dicionários multilíngue com compatibilidade.
- 036 - Workbench de dicionários e pipeline de limpeza.
- 037 - Compatibilidade PT-BR com treino atual (anti-regressão).
- 038 - Reproduzir `pt-br.tsv` a partir de `pt_BR` (ipa-dict) com regras.
- 039 - Tupi como primeiro idioma piloto multilíngue.
- 040 - Inglês como segundo idioma piloto.
- 041 - Leitor de dataset multilíngue com tags de idioma.

## Descrição
Planejar suporte multilíngue e variações dialetais. Definir tags/IDs para variações e um esquema canônico para armazenar dicionários, perfis de normalização e sobreposições.

## Contexto atual
- `dicts/` tem apenas `pt-br/` com um único dicionário TSV.
- A infraestrutura não tem hierarquia definida para variantes regionais ou línguas adicionais.
- Tupi é um caso especial: é língua indígena brasileira, mas não segue o esquema `lingua/região/sub-região` de uma localidade.
- Estado publicado atual: caminho canônico legado de `dicts/` está funcional e correto para quem clona hoje.
- Estado local em progresso: há reorganização em subpastas ainda não finalizada; não deve ser publicada antes dos tickets 035/037.

## Proposta de hierarquia de diretórios

### Critério de subdivisão
Não é simplesmente geográfico. A hierarquia deve refletir o **eixo de variação linguística** que importa para o pipeline G2P:
- Língua (ISO 639): `pt`, `en`, `tpn` (Tupi-Guarani), etc.
- Variante ortográfica/regional (ISO 3166 quando aplicável): `br`, `pt`, `ao` (Angola), etc.
- Sub-variante dialetal optativa: `sp` (São Paulo), `rj`, etc. (apenas se houver dicionário real distinto)

### Estrutura proposta:
```
dicts/
  pt/
    br/              ← Português Brasil (canônico atual, já existe como pt-br/)
      default.tsv
      README.md
      sp/            ← (futuro) variante paulistana, se houver dados
    pt/              ← (futuro) Português de Portugal
      default.tsv
  tpn/               ← Tupi/Tupi-antigo (ISO 639-3: tpn)
    default.tsv
    README.md
  en/
    us/
      default.tsv
```

### Por que Tupi sob `tpn/` e não sob `pt/br/`?
- Tupi não é variante regional do Português — é língua independente com ortografia e fonologia próprias.
- O prefixo ISO 639-3 `tpn` (Tupinambá) ou `tup` (Tupi genérico) identifica sem ambiguidade.
- Manter separado permite normalização independente e evita confusão entre grafemas do Tupi e do Português.

### Regra de sobreposição / merge de dialectos
- Camada base: `dicts/pt/br/default.tsv` (dicionário canônico)
- Camada de sobreposição: `dicts/pt/br/sp/overrides.tsv` (apenas entradas que diferem)
- O pipeline de normalização aplica base + overrides (lexicon layering), não duplica entradas.

## Critérios de aceite
- [ ] Definir e documentar a hierarquia de diretórios e nomeclatura de arquivos (este ticket).
- [ ] Migrar `dicts/pt-br/` existente para `dicts/pt/br/` (novo caminho canônico).
- [ ] Criar `README.md` de nível superior em `dicts/` com o esquema e regras de merge.
- [ ] Garantir que os scripts de normalização (`scripts/normalize_dicts.py`) referem ao novo caminho.
- [ ] Documento com proposta de esquema de tags e exemplos (PT-BR regionais, Tupi).
- [ ] Exemplo de um dicionário de sobreposição com formato e regras de merge.

## Próximos passos (pequenos incrementos)
1. Mover `dicts/pt-br/` → `dicts/pt/br/` (renomear pasta; atualizar referências no código) — baixo risco.
2. Criar `dicts/README.md` de nível superior com a hierarquia documentada.
3. Criar stub `dicts/tpn/README.md` sinalizando o espaço para o Tupi.
4. Atualizar `scripts/normalize_dicts.py` (e configs) para referenciar o novo caminho.
5. Criar subtickets por idioma/variante assim que houver dados concretos disponíveis.

## Dependências
- Ticket 032: organização das fontes `ipa-dict` e scripts de mapeamento.
- Migração de caminho pode impactar configs em `conf/` (verificar antes de mover).

## Descobertas verificadas (2026-03-20)
- O corpus canônico atual (`pt-br.tsv`) está altamente acoplado ao pipeline: há referências diretas em `src/` e em dezenas de configs em `conf/`.
- O arquivo fonte de origem (`pt_BR.txt`, do ipa-dict) está preservado em `dicts/pt-br/` e faz parte da trilha de derivação histórica.
- Já existe um arquivo inicial para Tupi: `dicts/tpw_latn_broad.tsv` (ainda sem integração no pipeline).

## Decisão arquitetural para códigos de língua (antes de mover pastas)

Evitar decidir código ISO por suposição. O nome atual do arquivo sugere `tpw`.

Proposta para o plano:
1. Definir um único código canônico para Tupi neste projeto (`tpw` ou `tpn`) com justificativa documental.
2. Manter um alias documental de compatibilidade no README de `dicts/` enquanto houver incerteza terminológica.
3. Só depois padronizar paths de pasta e nomes de arquivos.

## Plano de migração seguro (sem quebrar o treino atual)

### Fase 0 - Congelamento do canônico PT-BR
- Marcar `dicts/pt-br/pt-br.tsv` como **corpus canônico de treino atual**.
- Não alterar conteúdo deste arquivo nesta fase.
- Registrar checksum SHA256 no ticket 026 para controle de integridade.

### Fase 1 - Estrutura nova sem tocar no pipeline
- Criar estrutura alvo:
  - `dicts/pt/br/`
  - `dicts/tpw/` (ou `dicts/tpn/`, conforme decisão de código)
- Copiar (não mover) os ativos iniciais:
  - `dicts/pt-br/pt-br.tsv` -> `dicts/pt/br/default.tsv`
  - `dicts/pt-br/pt_BR.txt` -> `dicts/pt/br/sources/pt_BR.txt`
  - `dicts/tpw_latn_broad.tsv` -> `dicts/tpw/default.tsv` (ou caminho equivalente)
- Resultado: nova estrutura nasce sem afetar produção/treino.

### Fase 2 - Camada de compatibilidade de path
- Introduzir uma configuração central (ex.: `conf/dicts_registry.json`) com chaves:
  - `pt-br.default` -> caminho antigo (`dicts/pt-br/pt-br.tsv`) inicialmente
  - `pt-br.v2` -> caminho novo (`dicts/pt/br/default.tsv`)
  - `tpw.default` -> `dicts/tpw/default.tsv`
- Atualizar scripts para lerem do registry primeiro, mantendo fallback para path antigo.

### Fase 3 - Migração do código/config em lote
- Atualizar referências em `conf/*.json`, `src/*.py`, `Dockerfile` e docs para usar registry/chave lógica.
- Não remover ainda `dicts/pt-br/`.

### Fase 4 - Validação funcional
- Rodar sanity checks:
  - carga do corpus
  - split train/val/test
  - treino curto smoke test
  - inferência de 20 palavras
- Critério: resultados equivalentes (ou explicados) usando caminho antigo vs novo.

### Fase 5 - Corte controlado
- Só após validação, trocar canônico para `dicts/pt/br/default.tsv`.
- Manter `dicts/pt-br/` como legado por um ciclo curto (depreciação), depois remover.

## Regras de segurança para `pt-br.tsv`
- Não renomear/mover diretamente antes da Fase 4.
- Mudanças de conteúdo do corpus canônico só com checksum antes/depois + justificativa.
- Todo ajuste estrutural deve ser desacoplado de ajuste linguístico (não misturar no mesmo passo).

## Backlog consolidado
Substituído por tickets formais 035-041 para execução incremental e rastreável.

## Ordem recomendada revisada (cleanup-first)

Para reduzir confusão e evitar mexer em caminhos errados, a execução recomendada passa a ser:

1. 034 - Auditoria geral de organização (limpeza e inventário global).
2. 042 - Limpeza de legado e depreciação controlada (reduzir ruído de arquivos antigos).
3. 036 - Workbench de dicionários (`dicts-workbench`) para processamento separado.
4. 032 - Organizar fontes originais e expor pipeline/manifest de normalização.
5. 037 - Compatibilidade PT-BR (congelar referência + smoke tests).
6. 038 - Reproduzir `pt-br.tsv` a partir de `pt_BR` via regras auditáveis.
7. 035 - Estrutura multilíngue com compatibilidade de path.
8. 039 - Tupi como primeiro piloto multilíngue.
9. 040 - Inglês como segundo piloto.
10. 041 - Loader multilíngue com tags.

Racional:
- Primeiro limpar e reduzir ambiguidade documental/estrutural.
- Depois separar claramente consumo (`dicts/`) de proveniência/workbench.
- Em seguida proteger o que já funciona (PT-BR atual) e provar a reconstrução auditável.
- Só depois mexer no path canônico publicado e avançar para expansão multilíngue.
