ID: 081
Title: [v2.x] Investigar fontes com silabificacao para EN-US
Type: research
Priority: Medium
Status: Open
Created: 2026-04-14

## Contexto

O ipa-dict (nossa fonte atual) NAO inclui separadores silabicos (`.`)
na transcricao IPA do EN-US, embora inclua no PT-BR.

Isso e decisao da fonte, nao do idioma — ingles tem estrutura silabica
perfeitamente definida. A ausencia de separadores simplifica o treino
inicial mas limita analises de erro por posicao silabica.

## Impacto

- Sem separadores: modelo prediz fonemas puros sem fronteiras
- Com separadores: PER melhora ~17-20% (evidencia do PT-BR Exp101/102)
  mas WER piora ~6-8% (cada separador mal-posicionado = erro de palavra)
- Para TTS downstream: separadores ajudam no alinhamento fonetico

## Fontes potenciais com silabificacao EN-US

| Fonte | Silabificacao | Formato | Palavras | Licenca |
|---|---|---|---|---|
| Wiktionary (via WikiPron) | Parcial (contribuicoes da comunidade) | TSV IPA | 300K+ EN | CC BY-SA |
| CELEX | Sim | Proprietario | 52K | Restritiva |
| MRC Psycholinguistic DB | Sim | Proprietario | 150K | Academica |
| Algoritmo (Maximal Onset) | Geravel automaticamente | Python | Qualquer | N/A |

## Estrategia proposta (para v2.x)

1. Extrair EN-US do WikiPron com broad transcription
2. Filtrar entradas que incluem `.` na transcricao IPA
3. Cruzar com nosso en-us.tsv (ipa-dict) para enriquecimento
4. Para palavras sem `.`, aplicar silabificacao automatica (Maximal
   Onset Principle) e validar por amostra
5. Comparar PER/WER com vs sem separadores em EN-US

## Decisao atual (v2.0)

Treinar EN-US **sem** separadores (`keep_syllable_separators=false`).
Consistente com a maioria dos sistemas G2P em ingles (CMUdict,
Phonetisaurus, LatPhon). Separadores ficam para investigacao futura.

## Dependencias

- Ticket 026 (multilingue): pode ser que outros idiomas no ipa-dict
  tambem nao tenham separadores — precisa investigar por idioma
- WikiPron: avaliar como fonte complementar ao ipa-dict

## Proximos passos

1. [ ] Instalar WikiPron e testar extracao EN-US
2. [ ] Contar % de entradas Wiktionary EN com `.` na transcricao
3. [ ] Implementar silabificacao automatica como regra no pipeline
4. [ ] Treinar e comparar EN-US com vs sem separadores
