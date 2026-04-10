# FG2P — Roadmap

**Ultima atualizacao**: 2026-04-10
**Proposito**: Consolidar a visao de longo prazo do projeto FG2P, ligando
publicacoes cientificas, desenvolvimento tecnico, e decisoes de escopo.
Este documento e a fonte unica de verdade para "para onde o projeto vai".

---

## Visao geral em 30 segundos

O FG2P nasceu como **projeto congelado em PT-BR** para validar cientificamente
o conceito de **Distance-Aware Loss** — uma funcao de custo que penaliza
erros de G2P proporcionalmente a distancia articulatoria entre fonemas
preditos e alvos.

A v1.x esta concluida (PER 0.48%, WER 5.33%, 5 papers prontos) e o proximo
passo e **evoluir em duas dimensoes**:

1. **Horizontal**: expandir para multilingue
2. **Vertical**: modernizar arquitetura (Transformer) e melhorar formula de gradiente

Essa evolucao sera publicada em **tres papers distintos**, cada um com
contribuicao cientifica independente, seguindo padroes academicos
estabelecidos (ex: BERT→RoBERTa→DeBERTa).

---

## Estrategia de Publicacao (Trilogia de Papers)

### Paper A — v1.x: Prova de conceito em PT-BR (atual)

**Status**: pronto para submissao (abril 2026)

| Item | Detalhe |
|------|---------|
| Titulo | "Distance-Aware Loss for G2P: Brazilian Portuguese Case Study" |
| Contribuicao | Introduz DA Loss, valida com rigor estatistico em 1 lingua |
| Tese central | *"graduated loss funciona em condicoes controladas"* |
| Arquitetura | BiLSTM encoder-decoder + Bahdanau attention (2014) |
| Resultado principal | PER 0.48% (Wilson CI [0.46, 0.51]), WER 5.33% |
| Venues | arXiv, STIL 2026, SLT 2026, ICASSP 2027, MLSP 2026 |
| Branch git | `main` |
| Tag | `v1.1` (Exp104d) |

**Por que BiLSTM e nao Transformer?** Escolha deliberada para isolar a
contribuicao da funcao de custo da contribuicao da arquitetura. Ver
[Paper C](#paper-c--v2x-arquitetura-moderna--formula-melhorada) para a
evolucao arquitetural.

### Paper B — v2.0: Generalizacao multilingue (medio prazo ~2027)

**Status**: planejamento | **Dependencias**: tickets 024, 025, 026, 028

| Item | Detalhe |
|------|---------|
| Titulo provisorio | "Distance-Aware Loss for Multilingual G2P: Beyond Brazilian Portuguese" |
| Contribuicao | Demonstra transferencia cross-lingual, compara com LatPhon |
| Tese central | *"DA Loss generaliza alem de uma lingua"* |
| Arquitetura | BiLSTM (igual Paper A) OU Transformer (ver Paper C) |
| Linguas alvo | PT-BR + {ES, FR, IT, PT-PT, EN, ...} + possivelmente Tupi |
| Venues alvo | INTERSPEECH 2027, ACL 2027, TASLP |
| Branch git | `dev/v2.0` |

**Como se diferencia do Paper A**: cita explicitamente Paper A como baseline
em PT-BR. O contributo novo e mostrar que o metodo transfere — nao apenas
funciona em uma lingua. Inclui experimentos de cross-lingual transfer e
zero-shot em linguas nao vistas.

**Referencia cruzada obrigatoria**:
> "We previously showed that DA Loss achieves PER 0.48% on Brazilian
> Portuguese [Marques 2026a]. In this work, we extend DA Loss to a
> multilingual setting..."

### Paper C — v2.x+: Arquitetura moderna + Formula melhorada (longo prazo ~2028)

**Status**: pesquisa | **Dependencias**: tickets 034, 035

| Item | Detalhe |
|------|---------|
| Titulo provisorio | "Revisiting Distance-Aware Loss: Transformer Architectures and Improved Gradient Balancing" |
| Contribuicao | Nova formulacao matematica + arquitetura moderna |
| Tese central | *"eis a versao madura do metodo"* |
| Arquitetura | Transformer (Vaswani 2017) ou Conformer |
| Nova formula | Ver ticket 034 (opcoes A-D) |
| Venues alvo | ICASSP 2028, TASLP, Computer Speech & Language |
| Branch git | `dev/v2.x` ou posterior |

**Como se diferencia dos Papers A e B**: faz ablation direta contra ambos
como baselines. O contributo novo e a evolucao tecnica — nao mais um
"case study" ou "extensao", mas a **versao madura** do metodo.

**Referencia cruzada obrigatoria**:
> "Our previous work established DA Loss as a phonologically-graded training
> signal [Marques 2026a] and demonstrated multilingual transfer [Marques 2027].
> In this work, we revisit the formulation..."

---

## Estrategia de Desenvolvimento

### v1.x (congelado — NAO tocar ate publicacao dos papers de v1)

Branch: `main`
Escopo: apenas correcoes bibliograficas, formatacao, ajustes de reviewer.
**Nenhuma mudanca de metodo ou arquitetura.**

Arquivos principais:
- `dicts/pt-br.tsv` (corpus unico, congelado)
- `src/g2p.py` (BiLSTM, congelado)
- `docs/article/ARTICLE.md` (meta-artigo raiz, congelado apos Paper A)

### v2.0 (desenvolvimento ativo apos submissao do STIL)

Branch: `dev/v2.0`
Escopo: expansao multilingue **sem tocar na arquitetura**.

Subtickets de pesquisa:
- [024 — Pipeline fonotatico](evaluations/open/024_fonotatica.md)
- [025 — Espaco articulatorio continuo 7D](evaluations/open/025_7d_space.md)
- [026 — Multilingue/Tupi/dialetos](evaluations/open/026_multilingual_tupi.md)
- [027 — Estratificacao de batches multilingue](evaluations/open/027_batch_stratification.md)
- [028 — Morfossintaxe e homografos heterofonos](evaluations/open/028_morphosyntax.md)
- [030 — 'y' como glide, representacao de 'j'](evaluations/open/030_ipa_y_glide_and_j_representation.md)
- [031 — Ditongos nasais auditoria](evaluations/open/031_nasal_diphthongs_ỹ_ʊ̃_audit.md)
- [032 — Fontes originais ipa-dict e scripts](evaluations/open/032_dicts_sources_and_mapping_scripts.md)

Resultado esperado: dataset multilingue estruturado + modelo BiLSTM treinado
em N linguas + analise de transferencia cross-lingual.

### v2.x (desenvolvimento futuro — depende de v2.0 maduro)

Branch: `dev/v2.x` (a criar)
Escopo: modernizacao arquitetural e refinamento da formula de gradiente.

Subtickets de pesquisa:
- **[034 — Melhoria da formula de gradiente](evaluations/open/034_gradient_formula_improvement.md)** (NOVO)
- **[035 — Arquitetura Transformer como substituto do BiLSTM](evaluations/open/035_transformer_architecture_v2.md)** (NOVO)

Resultado esperado: Transformer + nova formula, ablation rigorosa contra
Papers A e B, resultado SOTA competitivo com LatPhon e Transformers modernos.

---

## Cronograma indicativo

| Data | Marco | Responsavel |
|------|-------|-------------|
| **Abr 2026** | Submissao Paper A: STIL, arXiv | Autor |
| **Mai 2026** | Submissao Paper A: MLSP | Autor |
| **Jun 2026** | Submissao Paper A: SLT | Autor |
| **Jul 2026** | Inicio desenvolvimento v2.0 (branch `dev/v2.0`) | Autor |
| **Out 2026** | Submissao Paper A: ICASSP 2027 | Autor |
| **Out-Dez 2026** | v2.0: dataset multilingue consolidado | Autor |
| **Jan-Mar 2027** | v2.0: experimentos multilingues | Autor |
| **Abr-Mai 2027** | Submissao Paper B | Autor |
| **Jun 2027** | Inicio desenvolvimento v2.x (Transformer + nova formula) | Autor |
| **2027-2028** | v2.x: pesquisa em formula e arquitetura | Autor |
| **~Out 2028** | Submissao Paper C | Autor |

Datas sao indicativas e devem ser ajustadas conforme feedback de reviewers
e disponibilidade de tempo.

---

## Como evitar self-plagiarism e salami slicing

Duas armadilhas reais em publicacoes sucessivas:

### Self-plagiarism
Copiar paragrafos literais do paper anterior sem citar.
**Solucao**: sempre citar a versao anterior e reescrever com linguagem nova.
Papers B e C precisam referenciar Paper A explicitamente no texto, nao so
na bibliografia.

### Salami slicing
Fatiar uma contribuicao unica em multiplos papers pequenos artificialmente.
**Por que nao se aplica aqui**: cada paper tem pergunta de pesquisa
genuinamente diferente:
- Paper A: *"DA Loss funciona em PT-BR?"*
- Paper B: *"DA Loss transfere entre linguas?"*
- Paper C: *"Como DA Loss se comporta com arquitetura moderna e formula melhorada?"*

### Regra dos 30%+
Editoras exigem 30%+ de conteudo novo entre papers sucessivos. Nosso caso:
- Paper B: ~60% novo (linguas adicionais + metricas de transferencia)
- Paper C: ~70% novo (arquitetura + formula + ablation contra A e B)

---

## Precedentes na literatura

Esta estrategia nao e inventada — e o padrao academico. Exemplos:

- **BERT (2018) → RoBERTa (2019) → DeBERTa (2020)**: mesma linha de pesquisa,
  cada paper cita os anteriores, sem acusacao de redundancia
- **GPT → GPT-2 → GPT-3 → GPT-4**: escalonamento publicado em sequencia
- **Word2Vec → GloVe → fastText**: mesmo problema, metodos sucessivos
- **LatPhon (Chary 2025)**: comecou mono-lingua, expandiu para 6 linguas romance

---

## Referencias do roadmap

- [Ticket 022 — Meta-ticket v2.0](evaluations/open/022_metrics_and_tf.md)
- [Ticket 033 — Estrategia de publicacao longo prazo](evaluations/open/033_publication_strategy_long_term.md)
- [PUBLICATION_PLAN.md](article/publications/PUBLICATION_PLAN.md) — tickets de publicacao imediata
- [DOUBLE_BLIND_POLICIES.md](article/publications/DOUBLE_BLIND_POLICIES.md) — politicas oficiais das conferencias
- [ARTICLE.md](article/ARTICLE.md) — meta-artigo raiz (fonte canonical v1.x)

---

## Historia de decisoes importantes

| Data | Decisao | Motivacao |
|------|---------|-----------|
| 2026-03 | Congelar projeto em PT-BR para v1.x | Rigor cientifico, isolar contribuicao |
| 2026-03 | Usar BiLSTM deliberadamente (nao Transformer) | Isolar efeito da loss da arquitetura |
| 2026-04-09 | Criar 5 papers para 5 venues diferentes | Maximizar exposicao, adaptar por venue |
| 2026-04-09 | publications/ sai do .gitignore | Versionamento > esconder drafts |
| 2026-04-10 | Adotar trilogia A/B/C para publicacao evolutiva | Evitar self-plagiarism, progression valida |
