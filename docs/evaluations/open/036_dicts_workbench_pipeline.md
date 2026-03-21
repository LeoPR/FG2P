ID: 036
Title: Workbench de dicionarios e pipeline de limpeza
Type: feature
Priority: High
Status: Open

Descrição:
Projetar uma área de trabalho separada para estudos e transformação de dicionários brutos até corpus final, sem contaminar a pasta oficial `dicts/`.

**Status de Implementação (2026-03-20)**: Estrutura base criada em `dicts-workbench/` com snapshot do ipa-dict realocado, recipe PT-BR declarada e documentação completa (ver SETUP_LOG.md).

Proposta inicial:
- Pasta de suporte: `dicts-workbench/`.
- Subpastas sugeridas:
  - `sources/` (fontes originais e snapshots compactados)
  - `recipes/` (regras declarativas de correção e manifests)
  - `scripts/` (pipeline de transformação)
  - `reports/` (diferenças/qualidade)
  - `build/` (saídas intermediárias)
  - `tmp/` (extração volátil/temporária; ignorada)

Estrutura recomendada:
```text
dicts-workbench/
  sources/
    ipa-dict/
      snapshots/
        2026-03-20/
          pt_BR.txt.zst
          LICENSE
          README.md
      manifest.json
  recipes/
    pt-br-from-ipa-dict.yaml
  scripts/
  reports/
    pt-br/
  build/
    pt-br/
  tmp/
```

Diretriz central:
- `dicts/` = somente outputs canônicos de consumo.
- `dicts-workbench/` = raw + recipe + build + relatório + proveniência.
- Promoção para `dicts/` deve ser passo explícito, nunca side effect automático silencioso.

Notas de decisão:
- Evitar `dicts/raw/`: funciona no curto prazo, mas mistura insumo bruto com artefato oficial e aumenta risco de publicação/confusão.
- Evitar um único mega-arquivo compactado com todos os idiomas como contrato principal: isso dificulta inspeção seletiva, checksum por língua e evolução futura. Se houver snapshot agregado, ele deve ser opcional, não a unidade lógica principal do pipeline.
- Compressão `zstd` faz sentido para snapshot local, mas é detalhe de armazenamento, não a base do desenho arquitetural.

Objetivo:
- Script genérico: lê corpus bruto + regras -> gera corpus final limpo para `dicts/`.

Fluxo recomendado:
1. Selecionar snapshot bruto (`sources/`).
2. Carregar recipe declarativa (`recipes/`).
3. Extrair para `tmp/` se necessário.
4. Aplicar regras e gerar build intermediário em `build/`.
5. Produzir relatório/checksums em `reports/`.
6. Promover manualmente o resultado aprovado para `dicts/`.

Critérios de aceite:
- Estrutura do workbench definida.
- Contrato de I/O do pipeline documentado.
- Exemplo mínimo de execução ponta-a-ponta.
- Manifest/recipe com input, checksum, data, licença, output esperado e regras associadas.

Dependências:
- 026, 032
