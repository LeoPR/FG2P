# Plano de Publicacoes — FG2P

**Ultima atualizacao**: 2026-04-10
**Fonte canonica**: docs/article/ARTICLE.md (meta-artigo) → derivados herdam correcoes
**Politicas de double-blind e repos**: ver DOUBLE_BLIND_POLICIES.md
**Visao de longo prazo (trilogia de papers)**: ver [docs/ROADMAP.md](../../ROADMAP.md)

**Escopo deste documento**: tickets de publicacao **imediata** (v1.x, abril-junho 2026).
Para papers futuros (v2.0 multilingue, v2.x Transformer), ver ROADMAP.md e
tickets 033/034/035 em `docs/evaluations/open/`.

---

## Regra de Propagacao de Correcoes

**Fluxo obrigatorio**: ARTICLE.md (core) → derivados (STIL, SLT, ICASSP, etc.)

1. Ao encontrar um erro em qualquer paper derivado, corrigir PRIMEIRO no ARTICLE.md
2. Propagar a correcao para TODOS os outros derivados que contenham o mesmo trecho
3. Ao encontrar um erro direto no ARTICLE.md, verificar se o erro existe nos derivados

Motivo: todos os papers compartilham vocabulario tecnico, formulas, numeros e claims.
Um erro num derivado provavelmente existe nos outros. Corrigir na raiz evita divergencia.

Checklist de propagacao (campos que devem ser identicos entre papers):
- Formula da DA Loss e descricao dos componentes
- Numeros: PER, WER, CIs, tamanho do test set, contagens de erros
- Nomes das classes de erro (A/B/C/D) e percentuais
- Descricao do LatPhon (params, PER, CI, arquitetura, loss=NLL/CE)
- Descricao do split estratificado (60/10/30, chi2, Cramer V)
- Tabelas de resultados principais e fatorial
- Declaracao de uso de IA (Level 2, mesma redacao base)

---

## Calendario de Deadlines (ordenado por urgencia)

| # | Venue | Tipo | Deadline | Formato | Paginas | Review | Fit | Status |
|---|-------|------|----------|---------|---------|--------|-----|--------|
| 0 | **arXiv** | Preprint | **ASAP** | Livre (LaTeX) | Sem limite | Moderacao | — | DRAFT+TEX PRONTOS |
| 1 | **STIL 2026** | Conf | **20 abr 2026** | SBC | Long 10+refs / Short 6+refs | Double-blind | EXCELENTE | DRAFT+TEX PRONTOS |
| 2 | **IEEE MLSP 2026** | Conf | **15 mai 2026** | IEEE 2-col | 6 incl refs | Double-blind | MEDIO | DRAFT+TEX PRONTOS |
| 3 | **EMNLP 2026** | Conf | **25 mai 2026** (ARR) | ACL | 8+refs | Double-blind | MEDIO | APERTADO |
| 4 | **KDMILE 2026** | Conf | **8 jun 2026** | LaTeX 8pp | 8 max | Single-blind | MEDIO | POSSIVEL |
| 5 | **IEEE SLT 2026** | Conf | **17 jun 2026** | IEEE 2-col | 6+2 refs | Double-blind | EXCELENTE | DRAFT PRONTO |
| 6 | **IberSPEECH 2026** | Conf | TBD (~jun?) | ISCA-like | 5pp | Double-blind | ALTO | MONITORAR |
| 7 | **ICASSP 2027** | Conf | ~out 2026 | IEEE 2-col | 4+1 ref | Single-blind | EXCELENTE | DRAFT PRONTO |
| 8 | **TSD 2026** | Conf | TBD | Springer LNAI | TBD | TBD | MEDIO | MONITORAR |
| 9 | **INTERSPEECH 2027** | Conf | ~fev 2027 | ISCA | 4+1 ref | Double-blind | EXCELENTE | FUTURO |
| 10 | **IEEE/ACM TASLP** | Journal | Rolling | IEEE Trans | 10-14 | Single-blind | EXCELENTE | LONGO PRAZO |
| 11 | **Phonetica** | Journal | Rolling | De Gruyter | Sem limite | Peer review | MEDIO | OA GRATIS 2026! |
| 12 | **IEEE Access** | Journal | Rolling | IEEE | Sem limite | Peer review | MEDIO | OA, rapido |
| 13 | **Comp. Speech & Lang** | Journal | Rolling | Elsevier | Sem limite | Peer review | ALTO | APC $3.600 |

### Deadlines ja passadas

| Venue | Deadline | Nota |
|-------|----------|------|
| INTERSPEECH 2026 | 25 fev 2026 | Sydney, set 2026 |
| SEPLN 2026 | 23 mar 2026 | NLP iberico |
| PROPOR 2026 | ~9 jan 2026 | Salvador, abr 2026. Proxima: PROPOR 2028 |
| ACL 2026 | mar 2026 (ARR) | San Diego, jul 2026 |
| EACL 2026 | out 2025 (ARR) | Rabat, mar 2026 |
| LREC 2026 | out 2025 | Palma de Mallorca, mai 2026 |
| IEEE ASRU 2025 | mai 2025 | Bienal impar. Proximo: ASRU 2027 |

---

## arXiv — Preprint (Prioridade Zero)

**Por que publicar primeiro no arXiv:**
1. **Timestamp de prioridade**: Marca quando o trabalho foi concluido (LatPhon apareceu set 2025)
2. **Visibilidade imediata**: Indexado por Google Scholar, Semantic Scholar, citations antes de conferencia
3. **Nao bloqueia nada**: Nenhum venue proibe submissao com preprint no arXiv (ver DOUBLE_BLIND_POLICIES.md)
4. **Gratuito e permanente**: Sem APC, identificador permanente
5. **Rapidez**: Disponivel em 1-2 dias uteis

**Categoria recomendada**: cs.CL (Computation and Language) como primaria, cross-list para eess.AS e cs.SD
- LatPhon (arXiv:2509.03300) usou cs.CL
- Maioria dos papers G2P usa cs.CL ou eess.AS

**Requisito: endorsement**
Desde janeiro 2026, arXiv exige endorsement para novos autores:
- Caminho 1 (automatico): email institucional + publicacao anterior no arXiv
- Caminho 2 (endorsement pessoal): pedir a um autor estabelecido na categoria
  (orientador, coautor, ou autor de paper citado — cada pagina de abstract tem link "Which authors are endorsers?")

**Formato**: LaTeX preferido (o main.tex do ICASSP/SLT serve como base)

**Acao**: Publicar no arXiv ANTES de qualquer deadline de conferencia.

---

## Prioridade Recomendada

### URGENTE — STIL 2026 (deadline 20 abr — 12 dias!)

**STIL 2026** — Cuiaba, 19-22 out 2026 (co-located BRACIS)
- Deadline: **20 abril 2026** (extendida)
- Formato: SBC. Long paper: 10 paginas + refs. Short paper: 6 paginas + refs.
- Linguas: PT-BR, EN, ES
- Double-blind
- Site: https://bracis.sbc.org.br/2026/stil/
- Pasta: `publications/stil/`
- Base: comprimir ARTICLE.md — pode ser em portugues!
- Fit: EXCELENTE — venue brasileira de PLN, audiencia perfeita, publicado no ACL Anthology
- **DECISAO NECESSARIA**: vale tentar em 12 dias? O ARTICLE.md ja tem tudo em PT-BR.

### Tier 1 — Acoes imediatas (abr-jun 2026)

**IEEE SLT 2026** — Palermo, 13-16 dez 2026
- Deadline: **17 junho 2026** (~10 semanas)
- Formato: IEEE 2-colunas, 6 paginas + 2 de referencias
- Double-blind (anonimizar)
- Pasta: `publications/slt/`
- **DRAFT + LATEX PRONTOS** (SLT_DRAFT.md + main.tex compilando)
- Fit: EXCELENTE

**KDMILE 2026** — Cuiaba, 19-22 out 2026 (co-located BRACIS)
- Deadline: **8 junho 2026**
- Formato: LaTeX, 8 paginas max, PT/EN
- Single-blind
- Site: https://bracis.sbc.org.br/2026/kdmile/
- Topicos: NLP listado explicitamente
- Fit: MEDIO — venue brasileira, menor prestígio que STIL mas menos competitivo
- Nota: se STIL for aceito, nao submeter a KDMILE (mesmo evento BRACIS)

**IEEE MLSP 2026** — Atlanta, 28 set - 1 out 2026
- Deadline: **15 maio 2026** (~5 semanas)
- Formato: IEEE 2-col, 6 paginas incluindo refs
- Double-blind (anonimizar)
- Pasta: `publications/mlsp/`
- **DRAFT + LATEX PRONTOS** (main.tex compilando, 6pp)
- Reframing: "Domain-Informed Loss Functions for Structured Sequence Prediction"
- Site: https://mlsp26.ieeesps.org/
- Fit: MEDIO — enfase em metodologia ML, G2P como instanciacao

### Tier 2 — Medio prazo (jul 2026 - fev 2027)

**IberSPEECH 2026** — TBD
- Deadline: TBD (~jun 2026 estimativa)
- Formato: ISCA-like, 5 paginas (5a so refs)
- Double-blind
- Site: https://iberspeech.tech/2026/call-for-papers/
- Fit: ALTO — speech iberico, PT-BR e core scope
- Monitorar CFP

**ICASSP 2027** — deadline ~outubro 2026
- Pasta: `publications/icassp/`
- **DRAFT + LATEX PRONTOS** (ICASSP_DRAFT.md + main.tex compilando)
- Single-blind (incluir nomes)

**TSD 2026** — Brno, 1-4 set 2026
- Deadline: TBD (historicamente ~mar-abr, pode ja ter passado)
- Formato: Springer LNAI proceedings
- Site: https://www.tsdconference.org/tsd2026/
- Fit: MEDIO — speech/dialogue
- Monitorar

**INTERSPEECH 2027** — deadline ~fevereiro 2027
- Pasta: `publications/interspeech/`
- Formato ISCA: 4+1 ref (ou Long Paper track)
- Double-blind

### Tier 3 — Journals (rolling, sem deadline)

**IEEE/ACM TASLP** — mais prestigioso para este topico
- Pasta: `publications/taslp/`
- Formato: IEEE Transactions, 10-14 paginas
- Single-blind
- IF: ~4.1
- Turnaround: 3-6 meses
- Estrategia: submeter apos publicacao de conferencia (SLT/ICASSP) + expandir

**Phonetica** (De Gruyter)
- **OA gratuito desde 2026** (CC BY 4.0) — sem APC!
- Foco em ciencia fonetica e modelagem computacional
- Bom se enfatizar contribuicao linguistica/fonologica
- Rolling submission

**IEEE Access**
- Open access, APC ~$1.750
- Turnaround rapido (~1-2 meses)
- Multidisciplinar, qualquer topico IEEE
- Menor prestigio mas boa visibilidade

**Computer Speech & Language** (Elsevier)
- IF: ~5.6
- APC: ~$3.600 (se OA)
- Escopo direto: speech, NLP, G2P
- Alternativa ao TASLP

**Natural Language Processing** (Cambridge UP, antigo "NLE")
- OA via acordos transformativos (pode ser gratis)
- Rolling submission
- Bom se reframing como metodologia NLP

---

## Mapeamento: Formato por Venue

| Venue | Template | Cols | Lingua | Anonimo | Derivado de |
|-------|----------|------|--------|---------|-------------|
| arXiv | Livre (IEEE ok) | 2 | EN | Nao | icassp/ ou slt/ |
| STIL 2026 | SBC | 1 | PT-BR | Sim | ARTICLE.md (comprimir) |
| SLT 2026 | IEEEtran conference | 2 | EN | Sim | **PRONTO** |
| KDMILE 2026 | LaTeX | TBD | PT/EN | Nao | ARTICLE.md (comprimir) |
| MLSP 2026 | IEEEtran conference | 2 | EN | Sim | **PRONTO** |
| IberSPEECH | ISCA-like | 2 | EN | Sim | slt/ (comprimir) |
| ICASSP 2027 | IEEEtran conference | 2 | EN | Nao | **PRONTO** |
| INTERSPEECH 2027 | ISCA | 2 | EN | Sim | slt/ (comprimir) |
| TASLP | IEEEtran transactions | 2 | EN | Nao | ARTICLE.md (expandir EN) |
| Phonetica | De Gruyter | TBD | EN | Nao | ARTICLE.md (foco fonologico) |

---

## Restricoes de Dual Submission

**REGRA GERAL**: arXiv preprint e permitido em TODOS os venues acima.

**Conferencias IEEE** (SLT, ICASSP, MLSP): O mesmo paper NAO pode estar
sob review em duas conferencias IEEE simultaneamente. Mas versoes diferentes
(ex: SLT 6pp vs ICASSP 4pp) sao tratadas como trabalhos distintos se
houver contribuicao adicional substancial.

**Journals**: Aceitam versoes expandidas de papers de conferencia (30%+ novo).
Citar a versao de conferencia. TASLP, CSL e Phonetica todos permitem isso.

**BRACIS** (STIL + KDMILE): NAO submeter o mesmo paper a ambos no mesmo ano.

---

## Uso de IA Generativa — Resumo

**Referencia completa**: ver [DOUBLE_BLIND_POLICIES.md](DOUBLE_BLIND_POLICIES.md)
secao "Uso de IA Generativa — Politicas Oficiais e Enquadramento".

Nosso uso se enquadra como **Nivel 2 (assistencia editorial)** com componente
de **Nivel 3 (compressao de drafts)** — declaramos por precaucao.
Todos os 5 papers tem secao de AI disclosure consistente e dentro das politicas.

| Venue | Onde declarar | Obrigatorio? |
|-------|---------------|-------------|
| IEEE (SLT, ICASSP, MLSP, TASLP) | Acknowledgments ou secao dedicada | Nivel 3 sim |
| SBC/STIL | Secao propria | Qualquer nivel |
| ISCA (INTERSPEECH) | Acknowledgments | Nivel 3 sim |
| ACL/EMNLP | Checklist + apendice | Nivel 3 sim |
| arXiv | Corpo do paper | Recomendado |

---

## Pre-requisitos no ARTICLE.md (tickets pendentes)

| Ticket | Descricao | Prioridade | Impacto |
|--------|-----------|------------|---------|
| 057 | §7.2 codigo → pseudocodigo | Medium | Todos os derivados |
| 062 | §10 limpeza, refs internas | Medium | Todos |
| 060 | §8.4 convergencia | Low | TASLP (journal) |
| 063 | Fluxo narrativo | Low | TASLP |
| 065 | ~~Mover material complementar para subpasta `supplementary/`~~ | ~~Low~~ | **FECHADO 2026-04-09** |
| 066 | Reorganizar ARTICLE.md em capitulos separados (chapters/) | Low | Organizacao — risco alto, avaliar apos STIL |
| 067 | Limpar docs/ raiz: mover artefatos soltos para subpastas | Low | Organizacao |
| 068 | ~~Criar pasta mlsp/ e gerar draft MLSP 2026~~ | ~~High~~ | **FECHADO 2026-04-09** |
| 069 | Sincronizar .bib: remover entradas nao-citadas em cada paper | Low | Higiene bibliografica |
| 070 | ~~Auditoria de contextualizacao de citacoes~~ | ~~High~~ | **FECHADO 2026-04-09** |
| 071 | ~~Consolidar politicas (double-blind, IA, .gitignore) em DOUBLE_BLIND_POLICIES.md~~ | ~~High~~ | **FECHADO 2026-04-09** |
| 072 | ~~Revisao final pre-submissao STIL (deadline 20 abr)~~ | ~~**URGENTE**~~ | **FECHADO 2026-04-09** |
| 073 | ~~Revisao final pre-submissao MLSP (deadline 15 mai)~~ | ~~High~~ | **FECHADO 2026-04-09** |
| 074 | ~~Revisao final pre-submissao SLT (deadline 17 jun)~~ | ~~High~~ | **FECHADO 2026-04-09** |
| 075 | ~~Revisao final pre-submissao arXiv (ASAP)~~ | ~~High~~ | **FECHADO 2026-04-09** |
| 076 | ~~Revisao final pre-submissao ICASSP (deadline ~out 2026)~~ | ~~Medium~~ | **FECHADO 2026-04-09** |
| 077 | ~~Pesquisa: nivel de detalhe exigido na declaracao de IA por venue~~ | ~~High~~ | **FECHADO 2026-04-09** |
| 078 | Ligar termos "near-miss"/"catastrofico" a taxonomia Classe A-D explicitamente | Low | Cosmetico — apos feedback de reviewers |

**Ticket 065 — Material complementar para subpasta**:
Mover DA_LOSS_ANALYSIS.md, EXPERIMENTS.md, FORMULAS.md, ORIGINALITY_ANALYSIS.md, PIPELINE.md
para `docs/article/supplementary/`. Estes arquivos contem fundamentacao teorica profunda
que alimenta o ARTICLE.md mas nao sao derivados de publicacao. Passos:
1. Criar `docs/article/supplementary/`
2. Mover os 5 arquivos (git mv)
3. Atualizar todas as referencias cruzadas no ARTICLE.md
4. Verificar que nenhum paper derivado referencia esses arquivos diretamente
5. REFERENCES.bib fica em `docs/article/` (usado por papers via TEXINPUTS)

**Ticket 066 — Capitulos separados**:
Proposta futura: quebrar ARTICLE.md (~1400 linhas) em chapters/01_intro.md, etc.
com ARTICLE.md como indice/TOC. Risco: Markdown nao tem \include{}, entao o
"indice" seria manual. Avaliar apos STIL quando nao houver deadline proximo.

**Ticket 067 — Limpar docs/ raiz**:
Arquivos soltos em `docs/` que deveriam ir para subpastas:
- `2509.03300v1.pdf` + `latphon_2509_03300v1_extracted.txt` → `docs/article/supplementary/references/` (material LatPhon)
- `FG2P_Consolidated_Report.html` + `FG2P_Report.docx` → `docs/report/` (ja existe)
- `INTEGRATION.md` → `docs/` OK (guia de integracao, acessivel da raiz)
Risco baixo. Atualizar referencias se necessario.

**Ticket 068 — MLSP 2026 draft** (FECHADO 2026-04-09):
Criado `publications/mlsp/` com main.tex (6pp IEEE, double-blind), mlsp_refs.bib
(12 entradas incl. szegedy2016rethinking), latexmkrc. Reframing ML-methodology:
"Domain-Informed Loss Functions for Structured Sequence Prediction".
Compila limpo, 6 paginas incluindo refs. Inclui: framework geral, worked example,
2x2 factorials (Sep×DA e PanPhon×DA), error taxonomy, OOV qualitativo, design space.

**Ticket 069 — Sincronizar .bib**:
Achados da varredura 2026-04-09:
- arXiv/SLT: `bottou2010large`, `kohavi1995crossvalidation` no .bib sem \cite → kohavi corrigido, bottou avaliar
- STIL: `neto2006brazilian` no .bib sem \cite → avaliar se intro deveria citar
Impacto: nenhum (BibTeX ignora entradas nao-citadas), mas higiene.

**Ticket 070 — Auditoria de contextualizacao de citacoes** (FECHADO 2026-04-09):
Verificacao sistematica de que todas as 13 referencias citadas nos 5 papers sao
contextualizadas no texto e relevantes. Resultado: 100% OK. Ver secao abaixo.

**Ticket 071 — Consolidar politicas em DOUBLE_BLIND_POLICIES.md** (FECHADO 2026-04-09):
Acoes realizadas:
1. Adicionados MLSP e STIL/BRACIS ao DOUBLE_BLIND_POLICIES.md com citacoes oficiais
2. Movida secao completa de IA generativa (framework 3 niveis, enquadramento, fontes)
   do PUBLICATION_PLAN.md para DOUBLE_BLIND_POLICIES.md
3. Atualizada secao .gitignore com decisao de 2026-04-09 (publications/ agora versionado)
4. Adicionada tabela de verificacao de AI disclosure em cada paper
5. PUBLICATION_PLAN.md agora aponta para DOUBLE_BLIND_POLICIES.md como referencia

**Ticket 072 — Revisao final STIL** (FECHADO 2026-04-09):
Checklist pre-submissao completo:
- [x] Formato SBC correto (sbc-template.sty, 1-col, 12pt)
- [x] 10 paginas de conteudo + refs ilimitadas
- [x] Double-blind: [ANONIMO], sem afiliacoes, sem self-refs identificaveis
- [x] Secao "Uso de IA Generativa" (§8) presente e conforme SBC
- [x] Secao "Limitacoes" (§6) presente com 5 itens (exigida por STIL 2026)
- [x] Secao "Declaracao de Etica" (§7) presente (exigida por STIL 2026)
- [x] Todas as citacoes resolvidas (0 undefined)
- [x] Numeros consistentes com ARTICLE.md e outros 4 papers (PER, WER, CIs, LatPhon)
- [x] Referencias bibliograficas contextualizadas (ticket 070)
- [x] Politica de anonimato conferida (DOUBLE_BLIND_POLICIES.md §7)
- [x] Compila limpo (0 erros, 0 warnings — corrigido babel brazil→brazilian)
- [x] Lingua: PT-BR consistente, abstract PT + resumo EN (padrao SBC)

**Ticket 073 — Revisao final MLSP** (deadline 15 mai 2026):
Mesmo checklist adaptado para IEEE double-blind, 6pp incl. refs, ingles.

**Ticket 074 — Revisao final SLT** (deadline 17 jun 2026):
Mesmo checklist adaptado para IEEE double-blind, 6+2pp refs, ingles.

**Ticket 075 — Revisao final arXiv** (ASAP):
Mesmo checklist adaptado para preprint: autor visivel, sem limite de paginas.

**Ticket 076 — Revisao final ICASSP** (deadline ~out 2026):
Mesmo checklist adaptado para IEEE single-blind, 4+1pp ref, ingles.

**Ticket 078 — Ligar near-miss/catastrofico a Classe A-D** (baixa prioridade):
Os termos "near-miss" e "catastrofico" usados no exemplo numerico sao
tecnicamente defensaveis (estabelecidos na literatura ML: catastrophic
forgetting/interference, near-miss pairs em metric learning) mas nao
estao explicitamente ligados a taxonomia Classe A-D definida na secao
de Metricas. Um reviewer pedante pode pedir essa ligacao explicita.
Solucao proposta (se necessario apos feedback): adicionar coluna "Classe"
na Tabela de exemplo numerico OU contextualizar no texto introdutorio
com algo como "...um erro Classe B (near-miss, par minimo) e um erro
Classe D (catastrofico, classes articulatorias distantes)".
Aplicar apenas se vier comentario de reviewer OU antes de camera-ready.
Todos os 5 papers (arXiv, STIL, SLT, ICASSP, MLSP) teriam a mesma
correcao propagada.

**Ordem recomendada**: **072** (STIL) → 073 (MLSP) → 074 (SLT) → 075 (arXiv) → 057 → 062 → 067 → tag v1.3 → 069 → 076 → 060+063+066 para TASLP

---

## Auditoria de Citacoes — Ticket 070 (FECHADO 2026-04-09)

Verificacao sistematica de que todas as referencias citadas nos 5 papers:
(a) sao discutidas no texto com contexto, nao "largadas" sem explicacao;
(b) sao relevantes ao trabalho; (c) seguem boas praticas academicas.

### Criterio de qualidade

A pratica academica padrao exige que toda citacao seja contextualizada:
- **Citacao narrativa**: "Rao et al. [2015] demonstrated that..." — padrao ouro
- **Citacao de recurso/metodo**: "PanPhon [Mortensen 2016]" — aceitavel quando
  o recurso ja foi descrito antes no texto
- **Citacao de fundamentacao**: "[Wilson 1927, Brown 2001]" apos explicar
  a escolha metodologica — padrao em estatistica

Citacoes sem contexto ("citation dropping") sao consideradas ma pratica.

### Resultado por referencia

| # | Chave BibTeX | Referencia | Tipo de uso | Contextualizacao |
|---|-------------|-----------|-------------|-----------------|
| 1 | `barbosa2004brazilian` | Barbosa & Albano (2004), JIPA — Ilustracao IPA oficial PT-BR | Fatos fonologicos: alofones roticos, assimilacao | Multiplas citacoes com contexto especifico |
| 2 | `bahdanau2014neural` | Bahdanau et al. (2014) — Mecanismo de atencao | Arquitetura: "BiLSTM with Bahdanau attention" | Citacao classica de arquitetura, nomeada |
| 3 | `bisani2008joint` | Bisani & Ney (2008), Speech Comm. — Joint-sequence G2P | (1) Baseline na Related Work; (2) Definicao da normalizacao PER | Dois contextos distintos |
| 4 | `chary2025latphon` | Chary et al. (2025) — LatPhon, Transformer multilingual G2P | Comparacao principal: arch, params, PER, CI, dados | Paragrafo dedicado com descricao completa. PDF baixado e verificado |
| 5 | `mortensen2016panphon` | Mortensen et al. (2016), COLING — PanPhon features | Metrica de distancia core: "24 binary features (voicing, nasality, place, manner)" | Multiplas citacoes com descricao funcional |
| 6 | `rao2015g2p` | Rao et al. (2015), ICASSP — LSTM G2P | Related Work: "demonstrated LSTM outperforms n-gram" | Citacao narrativa com resultado comparativo |
| 7 | `wilson1927probable` | Wilson (1927), JASA — Wilson score interval | Metodo estatistico: "used in place of Wald intervals, which underestimate..." | Justificativa de escolha metodologica |
| 8 | `brown2001interval` | Brown, Cai & DasGupta (2001), Stat. Sci. — Analise de CIs | Complementa Wilson: "recommends Wilson over Wald for small p" | Co-citado com Wilson, suporte estatistico |
| 9 | `neto2006brazilian` | Neto et al. (2006), ICASSP — Recursos G2P PT-BR | Related Work: "established early baselines with decision trees and HMMs" | Citacao narrativa como baseline historico |
| 10 | `kohavi1995crossvalidation` | Kohavi (1995), IJCAI — Cross-validation e estratificacao | Fundamentacao: "41% PER reduction attributable to protocol, not model" | Citado apos evidencia empirica de split bias |
| 11 | `bottou2010large` | Bottou (2010), COMPSTAT — SGD convergencia | Treino: "shuffling ensures efficient optimization" | Citacao de fundamentacao em ICASSP/MLSP |
| 12 | `szegedy2016rethinking` | Szegedy et al. (2016), CVPR — Inception/label smoothing | Contraste: "uniformly redistributes, ignoring output structure" | Citacao contrastiva na Related Work do MLSP |
| 13 | `morris2004cer` | Morris et al. (2004), Interspeech — Definicao formal de PER | Metrica: co-citado com Bisani na definicao de PER | Citacao tecnica no STIL |

### Uso por paper

| Paper | Refs citadas no texto | Refs no .bib nao citadas | Nota |
|-------|----------------------|-------------------------|------|
| arXiv | 10 | `bottou2010large` (no .bib, nao \cited) | Ref presente para eventual uso |
| SLT | 10 | `bottou2010large` | Idem |
| ICASSP | 11 | nenhuma | Limpo |
| MLSP | 11 | nenhuma | Limpo |
| STIL | 10 | `neto2006brazilian` (no .bib, nao \cited) | Avaliar se intro deveria citar |

### Materiais de referencia baixados

| Referencia | Arquivo local | Verificado |
|-----------|--------------|-----------|
| LatPhon (Chary 2025) | `docs/2509.03300v1.pdf` + `docs/latphon_2509_03300v1_extracted.txt` | Sim — conteudo conferido, numeros consistentes |

### Conclusao

Todas as 13 referencias usadas nos papers sao:
1. **Contextualizadas** no texto com pelo menos uma frase explicativa
2. **Diretamente relevantes** ao trabalho (fonologia PT-BR, arquitetura, metricas, loss design, avaliacao)
3. **De qualidade**: journals/conferencias de primeira linha (JIPA, ICASSP, COLING, JASA, Stat. Sci., CVPR, Interspeech)

Nenhuma citacao esta "largada" sem contexto. O padrao segue boas praticas academicas.

---

## Estrutura de Pastas

```
docs/article/                        ← "Biblioteca" de referencia
├── ARTICLE.md                       ← Meta-artigo (fonte canonica)
├── REFERENCES.bib                   ← BibTeX (usado por papers via TEXINPUTS)
├── supplementary/                   ← Material complementar aprofundado
│   ├── DA_LOSS_ANALYSIS.md          ← Teoria DA Loss + formulas alternativas
│   ├── EXPERIMENTS.md               ← Log Exp0-107 com configs JSON
│   ├── FORMULAS.md                  ← Derivacoes matematicas
│   ├── ORIGINALITY_ANALYSIS.md      ← Pesquisa de originalidade
│   ├── PIPELINE.md                  ← Pipeline tecnico de dados
│   └── BENCHMARK.md                 ← Metodologia de benchmark de velocidade
│
└── publications/                    ← Derivados (no .gitignore)
    ├── PUBLICATION_PLAN.md          ← ESTE ARQUIVO
    ├── README.md                    ← Convencoes e regras gerais
    ├── DOUBLE_BLIND_POLICIES.md     ← Politicas oficiais com fontes
    ├── icassp/                      ← ICASSP 2027 — DRAFT+TEX PRONTOS
    ├── slt/                         ← IEEE SLT 2026 — DRAFT+TEX PRONTOS
    ├── mlsp/                        ← IEEE MLSP 2026 — DRAFT+TEX PRONTOS
    ├── stil/                        ← STIL 2026 — DRAFT+TEX PRONTOS
    ├── interspeech/                 ← INTERSPEECH 2027
    └── taslp/                       ← TASLP journal
```

### Convencoes

- Cada subpasta = 1 venue (sem ano no nome)
- README.md: especificacoes, checklist, historico
- *_DRAFT.md: fonte markdown → gera LaTeX
- main.tex + *_refs.bib + latexmkrc: compilacao
- build/: artefatos intermediarios (latexmk $out_dir)
