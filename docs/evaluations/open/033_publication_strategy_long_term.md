ID: 033
Title: [v1.x→v2.x] Estrategia de publicacao de longo prazo (trilogia de papers)
Type: meta-research
Priority: High
Status: Open

## Contexto

O projeto FG2P foi deliberadamente congelado em v1.x para focar em PT-BR com
rigor estatistico. Isso produziu 5 papers prontos para submissao (arXiv, STIL,
SLT, ICASSP, MLSP). A proxima fase (v2.0+) expande o escopo tecnico em duas
direcoes simultaneas:
1. **Horizontal**: multilingue (alem de PT-BR)
2. **Vertical**: arquitetura moderna (Transformer) + formula de gradiente melhorada

A pergunta central deste ticket: **como publicar essa evolucao sem conflitar
com os papers de v1.x ja submetidos?**

## Resposta resumida (veja ROADMAP.md para detalhes)

Publicar em **tres papers distintos**, cada um com contribuicao cientifica
nova e independente, citando os anteriores explicitamente. Esse e o padrao
estabelecido na literatura (ex: BERT→RoBERTa→DeBERTa; LatPhon evoluiu de 1
lingua para 6).

### Paper A — v1.x (atual, pronto)
- **Titulo**: "Distance-Aware Loss for G2P: Brazilian Portuguese Case Study"
- **Contribuicao**: introduz DA Loss, valida com rigor estatistico em 1 lingua
- **Tese**: *"graduated loss funciona em condicoes controladas"*
- **Venues**: STIL 2026, SLT 2026, ICASSP 2027, MLSP 2026, arXiv
- **Status**: pronto para submissao

### Paper B — v2.0 (multilingue, medio prazo ~2027)
- **Titulo**: "Distance-Aware Loss for Multilingual G2P: Beyond Brazilian Portuguese"
- **Contribuicao**: demonstra transferencia cross-lingual, compara com LatPhon
  no mesmo setup multilingue
- **Tese**: *"DA Loss generaliza alem de uma lingua"*
- **Referencia explicita**: "We previously showed that DA Loss achieves
  PER 0.48% on PT-BR [Marques 2026]. In this work, we extend..."
- **Venues**: INTERSPEECH 2027, ACL 2027, TASLP
- **Subtickets**: 025 (espaco 7D), 026 (multilingue/Tupi)

### Paper C — v2.x+ (nova arquitetura + nova formula, longo prazo ~2028)
- **Titulo**: "Revisiting Distance-Aware Loss: Transformer Architectures and
  Improved Gradient Balancing"
- **Contribuicao**: nova formulacao matematica (ticket 034) + Transformer
  (ticket 035)
- **Tese**: *"eis a versao madura do metodo"*
- **Referencia explicita**: cita Papers A e B como baselines, faz ablation
  contra eles
- **Venues**: ICASSP 2028, TASLP, Computer Speech & Language
- **Subtickets**: 034 (formula), 035 (Transformer)

## Armadilhas a evitar

### Self-plagiarism
Copiar paragrafos literais do paper anterior sem citar.
**Solucao**: sempre citar a versao anterior e reescrever com linguagem nova.

### Salami slicing
Fatiar uma contribuicao unica em multiplos papers pequenos artificialmente.
**Por que nao se aplica ao nosso caso**: cada paper tem pergunta de pesquisa
genuinamente diferente (PT-BR vs multilingue vs arquitetura nova).

### "30%+ de conteudo novo"
Regra universal em editoras: cada paper sucessivo precisa de 30%+ de
contribuicao nova (metricas, experimentos, analises, ou dados).
**No nosso caso**: Paper B tem ~60% novo (linguas adicionais + transferencia
cross-lingual); Paper C tem ~70% novo (arquitetura + formula).

## Precedentes na literatura

- **BERT (2018) → RoBERTa (2019) → DeBERTa (2020)**: mesma equipe, arquitetura
  evoluindo, cada paper cita os anteriores
- **GPT → GPT-2 → GPT-3 → GPT-4**: escalonamento, cada versao e paper distinto
- **Word2Vec → GloVe → fastText**: mesmo problema, metodos sucessivos
- **LatPhon (Chary 2025)**: comecou mono-lingua, expandiu para 6 linguas romance

Em nenhum desses casos houve acusacao de redundancia ou self-plagiarism.

## Infraestrutura de separacao no projeto

Para evitar confusao entre v1 e v2+ no codigo:

- **Paper A (v1.x, congelado)**: branch `main`, usa `dicts/pt-br.tsv` e BiLSTM
- **Paper B (v2.0)**: branch `dev/v2.0`, usa `dicts/*.tsv` multilingue
- **Paper C (v2.x+)**: branch `dev/v2.x` ou posterior, Transformer + nova loss

Cada paper referencia o commit tag correspondente no repositorio para
reprodutibilidade exata.

## Criterios de aceite

- [ ] Cada um dos 3 papers tem escopo claramente definido
- [ ] Cada paper tem subtickets de pesquisa associados (024-028 para v2.0;
      034, 035 para v2.x)
- [ ] ROADMAP.md publicado em `docs/` consolidando a visao
- [ ] README.md da raiz referencia o ROADMAP
- [ ] Nenhum paper sucessivo sobrepoe a contribuicao do anterior

## Subtickets relacionados

- Ticket 022: Meta-ticket de trabalhos futuros v2.0 (ja existe)
- Ticket 024: Pipeline fonotatico (Paper B)
- Ticket 025: Espaco 7D multilingue (Paper B)
- Ticket 026: Multilingue/Tupi/dialetos (Paper B)
- Ticket 028: Morfossintaxe e homografos (Paper B/C)
- **Ticket 034 (novo)**: Melhoria da formula do gradiente (Paper C)
- **Ticket 035 (novo)**: Arquitetura Transformer (Paper C)

## Proximos passos

1. Criar tickets 034 e 035 (dependencias diretas)
2. Criar ROADMAP.md consolidando publicacao + desenvolvimento
3. Apos submissao do STIL, revisar cronograma de Paper B
4. Manter este ticket aberto como referencia permanente da estrategia
