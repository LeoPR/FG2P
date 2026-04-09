# Politicas Oficiais de Double-Blind e Repositorios Publicos

**Ultima atualizacao**: 2026-04-09
**Proposito**: Registrar o que as editoras/conferencias DIZEM OFICIALMENTE sobre
double-blind review, repositorios publicos (GitHub) e preprints (arXiv).
Apenas citacoes de fontes primarias — sem deducoes logicas.

---

## Resumo Executivo

| Venue | Tipo Review | GitHub publico OK? | arXiv OK? | Periodo de anonimato? | Linkar repo NO paper? |
|---|---|---|---|---|---|
| **IEEE SLT** | Double-blind | Sim (na web) | Provavel (silencioso) | Nao explicito | NAO (usar Anonymous GitHub) |
| **IEEE ICASSP** | **Single-blind** | Sim | Sim (explicito) | N/A | Sim |
| **IEEE MLSP** | Double-blind | Sim (na web) | Sim (explicito) | Nao explicito | NAO (usar Anonymous GitHub) |
| **STIL/BRACIS** | Double-blind | Sim (na web) | Sim (anonimo) | **~1 mes antes do deadline** | NAO (usar Anonymous GitHub) |
| **INTERSPEECH** | Double-blind | Sim (na web) | Sim (explicito) | **REMOVIDO** | So anonimizado |
| **ACL/ARR** | Double-blind | Sim (na web) | Sim (explicito) | **REMOVIDO** (desde fev 2024) | So via Anonymous GitHub |
| **IEEE/ACM TASLP** | Single-blind | Sim | Sim (explicito) | N/A | Sim |

**Conclusao**: NENHUM venue exige tornar o repositorio GitHub privado.
A restricao e sobre o que vai DENTRO do paper submetido.
O STIL/BRACIS e o mais restritivo: preprint deve ser anonimo e nao pode ser
atualizado durante o periodo de anonimato.

---

## 1. IEEE SLT 2026

**Fonte**: https://attend.ieee.org/slt-2026/call-for-papers/

**Tipo**: Double-blind

**Citacao oficial SLT 2026** (unica referencia disponivel ate agora):
> "The review process is double-blind. Authors must ensure that their
> submissions do not reveal their identities in any way."

O site do SLT 2026 e minimalista e nao detalha politica de preprints ou repos.

**Referencia: SLT 2024** (edicao anterior, mais detalhada):
Fonte: https://2024.ieeeslt.org/paper_submission/

> "please refrain from adding acknowledgements, grant numbers as well as
> **public github repository links** to your submitted paper"

> "Submission of code through **anonymous github repositories** is allowed;
> however, they have to be on a branch that will not be modified after the
> submission deadline."

> "The availability of information on the web that may allow reviewers to
> infer the authors' identities **does not constitute a breach** of the
> double-blind submission policy."

**O que isso significa**:
- NAO colocar link do GitHub identificavel no paper
- PODE usar Anonymous GitHub para enviar codigo aos revisores
- Ter o repo publico na web NAO e violacao
- SLT 2024 e silencioso sobre arXiv/preprints (nao proibe explicitamente)

---

## 2. IEEE ICASSP

**Fonte**: https://2026.ieeeicassp.org/sps-policies/
**Paper kit**: https://cmsworkshops.com/ICASSP2026/papers/paper_kit.php

**Tipo**: Single-blind (NAO e double-blind)

**Citacao oficial**:
> "**ICASSP does not perform blind reviews**, so be sure to include the
> author list in your submitted paper."

**Sobre preprints**:
> "Authors may post their preprints in the following locations: Author's
> personal website, Author's employer's website, arXiv.org, TechRxiv.org,
> Funder's repository"
>
> "**This does not count as a prior publication.**"

**Sobre GitHub**: Silencioso — nenhuma mencao. Como e single-blind, nao ha
restricao sobre repos publicos ou links no paper.

**O que isso significa**:
- Incluir nome do autor no paper (obrigatorio)
- Pode linkar GitHub, arXiv, etc. livremente
- Zero restricoes de anonimato

---

## 3. INTERSPEECH 2026 (ISCA)

**Fonte principal**: https://interspeech2026.org/en-AU/pages/author-resources/submission-policy
**Politica ISCA**: https://isca-speech.org/Interspeech-Policy

**Tipo**: Double-blind

**MUDANCA CRITICA — periodo de anonimato REMOVIDO**:
> "**Interspeech no longer enforces an anonymity period for submissions.**
> While uploading a version online is permitted, your official submission
> to Interspeech must not contain any author-identifying information."

**Sobre preprints/arXiv** (politica ISCA):
> "Authors are allowed to submit manuscripts posted in institutional and
> public repositories (such as arXiv) to Interspeech conferences for
> consideration"
>
> "Authors are required to mention 'submitted to Interspeech' with any
> paper posted on public repositories (such as arXiv)."

**Sobre repos/GitHub** (politica de submissao):
> "Supplementary materials and online resources like repositories
> (if applicable) **do not reveal the authors' identities**."

**Sobre referencias a fontes nao peer-reviewed**:
> "papers submitted to INTERSPEECH should refer to peer-reviewed publications,
> and references to non-peer-reviewed publications (including public repositories
> such as arXiv, Preprints, and HAL, software, and personal communications)
> should only be made if there is no peer-reviewed publication available,
> and should be kept to a minimum."

**O que isso significa**:
- PODE manter repo publico e preprint no arXiv durante review
- O paper submetido deve ser anonimizado (sem nomes, sem GitHub identificavel)
- Se linkar repo no paper, deve ser anonimizado
- Marcar preprint com "submitted to Interspeech"

---

## 4. ACL Rolling Review (ARR) — aplica-se a ACL, EMNLP, NAACL, EACL, STIL*

**Fonte**: https://aclrollingreview.org/cfp
**Politica de anonimato**: https://aclrollingreview.org/anonymity
**Guidelines**: https://aclrollingreview.org/authors

*STIL historicamente publica no ACL Anthology e pode seguir politicas ACL.
Verificar CFP especifico do STIL 2026 quando publicado.

**Tipo**: Double-blind ("two-way anonymized")

**MUDANCA IMPORTANTE (efetiva desde 15 fev 2024)**:
> "The ACL has adopted a new anonymity policy effective for all future
> submissions, including to ARR."

> "**Authors are free to post and discuss non-anonymous preprints at
> any time.**"

> "Beginning with the February 15, 2024 ARR deadlines, there is **no
> anonymity period or limitation on posting or discussing non-anonymous
> preprints** while the work is under peer review."

**Mecanismo de incentivo** (nao obrigatorio):
> A politica "incentivizes anonymous submissions by **special paper awards
> and priority in acceptance decisions for borderline papers**."
>
> "Authors select the preprint status of the submission in the submission
> form. If you choose the 'binding' non-preprint option, you commit to not
> preprinting until after the review cycle is over, under the penalty of
> desk rejection."

**Sobre repos/GitHub**:
> "Supplementary materials, including any **links to repositories, should
> also be anonymized**."
>
> "If supplementary software is provided through a link to an online
> repository, it should be properly anonymized (e.g., **Anonymous GitHub**)."
>
> "Links to file hosting services that can track downloads, such as
> Dropbox, are not allowed."

**Sobre o paper**:
> "Papers must not include authors' names and affiliations. Furthermore,
> self-references that reveal the authors' identities, e.g., 'We previously
> showed (Smith, 1991)...' must be avoided."

**O que isso significa**:
- PODE manter GitHub publico e arXiv durante review (periodo de anonimato REMOVIDO)
- O paper deve ser anonimizado (sem nomes, sem self-references identificaveis)
- Links a repos no paper devem usar Anonymous GitHub
- Opcao de declarar "non-preprint" para vantagem em papers borderline (nao obrigatorio)

---

## 5. IEEE/ACM TASLP

**Fonte**: https://signalprocessingsociety.org/publications-resources/information-authors

**Tipo**: Single-blind (confirmado)

**Citacao oficial**:
> "**Single-anonymized peer review process**, where the identities of the
> reviewers are not known to the authors, but the reviewers know the
> identities of the authors."

**Sobre preprints**:
> "Authors may post preprints of the submitted manuscript [...] on their
> personal website, on their employer's website, and/or on **approved
> third-party preprint servers (such as arXiv)**."
>
> Authors must "provide a **complete list (including URLs) of all posted
> preprints** of the submitted manuscript" during submissao.

**Sobre GitHub**: Silencioso — nenhuma mencao. Como e single-blind,
nao ha restricao sobre repos publicos.

**O que isso significa**:
- Reviewers ja sabem quem voce e
- Pode ter repo publico, arXiv, etc. sem restricao
- Declarar preprints existentes no formulario de submissao

---

## 6. IEEE MLSP 2026

**Fonte**: https://mlsp26.ieeesps.org/paper-submission-instructions/
**Review process**: https://mlsp26.ieeesps.org/paper-review-process/

**Tipo**: Double-blind

**Citacao oficial**:
> "Please observe that MLSP uses a double-blind review process."
>
> "We conduct a double-blind review process."
>
> "Each paper will be evaluated anonymously by at least three reviewers."

**Sobre preprints/arXiv**:
> "Uploading a preprint to arXiv or other public repositories does not
> constitute grounds for rejection."
>
> "authors are encouraged to mask or modify the preprint to the best of
> their ability during the review period so that reviewers cannot easily
> link the preprint to the MLSP submission."

**Sobre repos/GitHub**:
> Code submission is "highly recommended but not required" and must be
> "completely anonymized using tools like https://anonymous.4open.science/."

**Paginas**: 6 paginas incluindo refs e figuras.
> "Papers must not be longer than 6 pages, including all text, figures,
> and references."

**Plataforma**: OpenReview.

**O que isso significa**:
- PODE manter GitHub publico e arXiv durante review (explicito: "not grounds for rejection")
- O paper deve ser anonimizado (template com Anonymous/Anonymous)
- Se compartilhar codigo, usar Anonymous GitHub
- Recomendado (nao obrigatorio) mascarar preprint durante review

---

## 7. STIL 2026 (SBC/BRACIS)

**Fonte STIL**: https://bracis.sbc.org.br/2026/stil/
**Fonte BRACIS**: https://bracis.sbc.org.br/2026/bracis/

**Tipo**: Double-blind ("double-anonymous")

**Citacao oficial STIL**:
> "The review process will be double-blind, therefore papers must not
> contain any information revealing the identity of the authors."

**Citacao oficial BRACIS** (politica guarda-chuva):
> "BRACIS employs a double-anonymous review process. This means that both
> the reviewer's and author's identities and institutions are concealed
> from the reviewers, and vice versa, throughout the review process."

**Sobre preprints/arXiv**:
> "It is allowed to make an anonymous version of the paper publicly
> available (for example, on OpenReview or arXiv), even during the
> anonymity period."
>
> "During the anonymity period, it is not allowed to publicly release
> a non-anonymous version of the paper."

**Periodo de anonimato**:
> "from one month before the submission deadline until the acceptance
> or rejection notification."

Deadline 20 abr → anonimato comeca ~20 mar 2026.

**IMPORTANTE**: nao atualizar preprint nem postar em redes sociais durante review:
> "You cannot update the online version nor publish information regarding
> the work on social media during the paper review period, as it can
> compromise the double-anonymous review process."

**Sobre repos/GitHub**:
> Authors are "strongly encourage[d]" to make "code and data available
> anonymously (e.g., in an anonymous GitHub repository via Anonymous
> GitHub or in a Dropbox folder)."

**Sobre IA generativa (BRACIS/STIL)**:
> "Authors using LLMs take full responsibility for all content, including
> checking for plagiarism."
>
> Disclosure deve estar na secao Acknowledgments, "without prejudice to
> the evaluation process."

STIL exige secao dedicada:
> "Submissions must include a Limitations section, Ethics Statement, and
> a disclosure of generative AI usage (extra page allowed)."

**Paginas**:
- Long paper: 10 paginas + refs ilimitadas
- Short paper: 6 paginas + refs ilimitadas

**O que isso significa**:
- PODE ter repo publico, mas NAO linkar no paper
- PODE ter preprint anonimo no arXiv durante review
- NAO pode publicar versao nao-anonima durante periodo de anonimato (~20 mar a notificacao)
- NAO atualizar preprint durante review
- Secao de disclosure de IA OBRIGATORIA (pagina extra permitida)
- Se compartilhar codigo, usar Anonymous GitHub

---

## Implicacoes Praticas para o Projeto FG2P

### O repositorio GitHub precisa ser privado?

**NAO.** Nenhum venue exige isso. Todos permitem explicitamente (ou sao
silenciosos sobre) a existencia de repos publicos na web durante review.

A citacao mais clara e do SLT 2024:
> "The availability of information on the web that may allow reviewers to
> infer the authors' identities **does not constitute a breach** of the
> double-blind submission policy."

### O que precisa ser anonimizado?

Apenas o CONTEUDO DO PAPER SUBMETIDO:
1. Remover nomes de autores e afiliacoes
2. Nao linkar o GitHub identificavel no texto do paper
3. Se quiser fornecer codigo aos revisores, usar Anonymous GitHub
4. Evitar self-references identificaveis ("nosso trabalho anterior [Smith, 2025]")

### A pasta publications/ esta no git?

**SIM (desde 2026-04-09).** Decisao: privilegiar controle de versao e seguranca
historica sobre a conveniencia de esconder drafts.

Considerando as politicas oficiais: **nenhum venue proibe ter conteudo na web**.
A restricao e sobre o que vai DENTRO do paper submetido (sem nomes, sem links
identificaveis). Ter o .tex no GitHub publico nao viola nenhuma politica.

Os artefatos LaTeX (*.aux, *.log, *.pdf, build/) continuam no .gitignore —
sao efemeros e nao tem valor de versionamento.

Historico de decisoes:
- **Antes de 2026-04-09**: publications/ inteiro no .gitignore (por precaucao)
- **2026-04-09**: removido do .gitignore apos auditoria de politicas de todos
  os venues. Nenhum exige repo privado. Controle de versao priorizado.

### Resumo de acoes por venue

| Venue | Acao no paper | Acao no repo | Acao no arXiv |
|---|---|---|---|
| SLT | Anonimizar, Anonymous GitHub | Nenhuma | Pode postar (silencioso) |
| ICASSP | Incluir nome | Nenhuma | Pode postar (explicito) |
| MLSP | Anonimizar, Anonymous GitHub | Nenhuma | Pode postar (explicito, "not grounds for rejection") |
| STIL/BRACIS | Anonimizar, Anonymous GitHub | Nenhuma | Pode postar anonimo; NAO atualizar durante review |
| INTERSPEECH | Anonimizar, Anonymous GitHub | Nenhuma | Pode postar + marcar "submitted to IS" |
| ACL/ARR | Anonimizar, Anonymous GitHub | Nenhuma | Pode postar (explicito) |
| TASLP | Incluir nome | Nenhuma | Pode postar + declarar na submissao |

---

## Uso de IA Generativa — Politicas Oficiais e Enquadramento

### O que "geracao de conteudo" significa na pratica

Nenhuma editora define com precisao absoluta onde termina "edicao" e comeca
"geracao de conteudo". Mas a literatura academica e as politicas convergem
em um framework de 3 niveis (adaptado de Resnik & Hosseini 2025, AMEE Guide 192,
e revisao comparativa de PMC12170296):

#### Nivel 1 — Disclosure DESNECESSARIO
- Corretor ortografico e gramatical (Grammarly, Word spellcheck)
- Gerenciamento de referencias (Zotero, Mendeley)
- Teclado preditivo / autocomplete curto
- Busca de literatura (usar IA para *encontrar* papers — desde que o autor leia e cite)

#### Nivel 2 — Disclosure OPCIONAL (recomendado)
- Parafrasear ou polir texto **ja escrito pelo autor** (melhorar clareza, fluxo)
- Correcao de estilo e reestruturacao de frases
- Revisao de consistencia terminologica
- Formatacao (LaTeX, conversao de formatos)
- Sugestoes de organizacao estrutural

> **IEEE**: "editing and grammar enhancement is common practice and generally
> outside the intent of [the disclosure] policy" — isento, mas recomendado.
>
> **ACL**: "paraphrasing or polishing the author's original content, rather
> than suggesting new content" — permitido sem disclosure.
>
> **SAGE**: Classifica como "Assistive AI" — distinto de "Generative AI".

#### Nivel 3 — Disclosure OBRIGATORIO
- Gerar texto novo (paragrafos, secoes) que o autor nao escreveu primeiro
- Gerar ideias de pesquisa ou hipoteses
- Criar figuras, tabelas ou codigo
- Resumir literatura para inclusao direta no paper
- Brainstorming e ideacao de contribuicoes cientificas
- Interpretar dados ou resultados

> **IEEE**: "shall be disclosed in the acknowledgments section. The AI system
> used shall be identified, and specific sections [...] shall be identified."
>
> **ACL**: Exige especificar onde o texto foi usado; publicado como apendice.
>
> **SBC/STIL**: Secao dedicada obrigatoria no paper.

#### A zona cinza (onde especialistas divergem)

A fronteira mais ambigua esta entre "polir texto existente" (Nivel 2) e
"reescrever para melhor clareza e fluxo" (potencialmente Nivel 3). Pontos
de tensao identificados na literatura:

- **Onde termina "correcao" e comeca "reescrita"?** Springer Nature isenta
  "AI-assisted copy editing" mas nao define o limite. The Lancet permite
  apenas "readability and language" — exclui ate sumarizacao.
- **Proporcionalidade** (ACM): disclosure deve ser proporcional a "the
  proportion of new text or content generated". Uma frase != uma secao.
- **Tipo de documento importa**: arXiv e mais rigoroso com surveys/reviews
  do que com papers originais — preocupacao com submissoes em massa.

#### O criterio consensual: quem contribuiu intelectualmente?

> "The distinction isn't truly about *who typed* the text, but rather
> who contributes intellectually and accepts responsibility for it."
> — PMC12170296 (Comparative Review of Editorial Policies, 2025)

O teste pratico: se voce remover a IA do processo, o conteudo cientifico
(hipoteses, design, analise, conclusoes) muda? Se nao, e assistencia.
Se sim, e geracao de conteudo.

#### Recomendacao pratica: "When in doubt, disclose"

Todas as fontes convergem neste principio. O custo de declarar e zero
(nenhum venue penaliza disclosure honesto — SBC/BRACIS explicita "without
prejudice to the evaluation process"). O custo de nao declarar pode ser
retratacao.

### O que EXATAMENTE declarar? Nivel de detalhe por venue (ticket 077, pesquisa 2026-04-09)

A questao: basta dizer "usamos IA", ou precisa nomear a ferramenta, versao, secoes?

**IEEE** (SLT, ICASSP, MLSP, TASLP) — o MAIS especifico:
> "The AI system used **shall be identified**, and **specific sections** of
> the article that use AI-generated content **shall be identified** and
> accompanied by a **brief explanation regarding the level** at which the
> AI system was used to generate the content."
>
> Fonte: https://journals.ieeeauthorcenter.ieee.org/.../submission-and-peer-review-policies/

**O que o IEEE pede**: (1) nome do sistema; (2) secoes especificas; (3) nivel de uso.
**O que NAO pede**: versao, parametros, prompts.
Excecao: "editing and grammar enhancement" e isento, mas recomendado.

**SBC/BRACIS** (STIL, KDMILE) — recomendado, nao prescritivo:
> "We suggest that this use be properly mentioned in the Acknowledgements
> section."
>
> Fonte: https://bracis.sbc.org.br/2026/bracis/

STIL 2026 exige secao dedicada mas nao especifica o formato:
> "Submissions must include [...] a disclosure of generative AI usage
> (extra page allowed)."

**O que o SBC pede**: mencionar que usou. Sem formato prescrito.
**Recomendacao**: seguir o padrao IEEE (mais rigoroso) para nao ter problema.

**ACL/ARR** (EMNLP, STIL se via ARR) — scope and nature:
> Authors must "elaborate on the **scope and nature** of their use."
>
> Fonte: https://2023.aclweb.org/blog/ACL-2023-policy/

Respostas ficam visiveis como apendice. Sem template rigido.

**ISCA** (INTERSPEECH) — sem politica especifica de formato de disclosure.

**arXiv** — autor assume responsabilidade; recomendado declarar no corpo.

#### Resumo comparativo: o que declarar

| Elemento | IEEE | SBC/STIL | ACL | ISCA | arXiv |
|----------|------|----------|-----|------|-------|
| Nome da ferramenta | **Obrigatorio** | Recomendado | Nao especificado | Nao especificado | Recomendado |
| Versao | Nao exigido | Nao exigido | Nao exigido | Nao exigido | Nao exigido |
| Secoes afetadas | **Obrigatorio** | Nao exigido | "scope and nature" | Nao especificado | Recomendado |
| Nivel de uso | **Obrigatorio** | Nao exigido | "scope and nature" | Nao especificado | Recomendado |

#### Decisao para nossos papers

Adotamos o padrao **IEEE** (o mais rigoroso) em TODOS os papers por consistencia
e seguranca. Nosso disclosure declara:
1. **Nome**: "Claude, Anthropic" (ferramenta identificada)
2. **Secoes**: "manuscript preparation" (escopo: todo o texto)
3. **Nivel**: "exclusively as editorial assistance: grammar and spell checking,
   bibliographic reference search, citation management, terminology consistency
   verification across sections, and LaTeX formatting"
4. **Negativa explicita**: "The tool did not generate new text, hypotheses,
   figures, code, or data interpretations"

Versao do modelo NAO e exigida por nenhum venue. Incluir opcionalmente
seria boa pratica mas nao e necessario.

### Enquadramento do nosso uso

| Atividade | Nivel | Disclosure |
|-----------|-------|------------|
| Design experimental, codigo, execucao | Humano | N/A |
| Analise de resultados e conclusoes | Humano | N/A |
| Redacao inicial do ARTICLE.md | Humano | N/A |
| Organizacao estrutural (secoes, fluxo) | 2 (assistencia) | Recomendado |
| Revisao de consistencia terminologica | 2 (assistencia) | Recomendado |
| Formatacao LaTeX e conversao de formatos | 2 (assistencia) | Recomendado |
| Busca de inconsistencias entre secoes | 2 (assistencia) | Recomendado |
| Pesquisa de venues, deadlines, politicas | 1 (busca) | Desnecessario |
| Compressao ARTICLE.md → drafts de conferencia | 2-3 (zona cinza) | **Obrigatorio** |

A compressao do ARTICLE.md para drafts derivados e o caso mais proximo do
Nivel 3: a IA seleciona o que incluir/excluir e reformula para caber no
limite de paginas. Embora o conteudo cientifico seja inteiramente do autor,
a *selecao e reformulacao* tem componente generativo. Declaramos como
Nivel 3 por precaucao.

### Politicas de IA por venue — resumo

| Venue | Onde declarar | Nivel 1 | Nivel 2 | Nivel 3 |
|-------|---------------|---------|---------|---------|
| **IEEE** (SLT, ICASSP, MLSP, TASLP) | Acknowledgments ou secao dedicada | Isento | Isento (recomendado) | Obrigatorio |
| **SBC/STIL** | Secao propria (obrigatoria) | Obrigatorio* | Obrigatorio* | Obrigatorio |
| **ISCA** (INTERSPEECH) | Acknowledgments | Isento | Isento (recomendado) | Obrigatorio |
| **ACL/EMNLP** | Checklist + apendice | Isento | Isento | Obrigatorio |
| **arXiv** | Corpo do paper | Recomendado | Recomendado | Obrigatorio |

*STIL 2026 exige secao dedicada de disclosure para qualquer uso de IA, independente do nivel.

### Verificacao: como cada paper declara uso de IA (auditoria 2026-04-09)

| Paper | Secao presente? | Titulo da secao | Conteudo |
|-------|----------------|-----------------|----------|
| arXiv | Sim | "Use of Generative AI Tools" | Claude/Anthropic, editorial, Level 2 |
| SLT | Sim | "Use of Generative AI Tools" | Claude/Anthropic, editorial only |
| ICASSP | Sim | "Use of Generative AI Tools" | Claude/Anthropic, editorial only |
| MLSP | Sim | "Use of Generative AI Tools" | Claude/Anthropic, editorial only |
| STIL | Sim | "Uso de IA Generativa" (PT-BR) | Claude/Anthropic, assistencia editorial |

Todos os 5 papers declaram uso de IA de forma consistente e dentro das
politicas de cada venue.

### Fontes primarias consultadas

- [IEEE Author Center — Submission Policies](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/)
- [ACL 2023 Policy on AI Writing Assistance](https://2023.aclweb.org/blog/ACL-2023-policy/)
- [BRACIS 2026 CFP — On the usage of Generative AI tools](https://bracis.sbc.org.br/2026/bracis/)
- [ISCA 2026 Submission Guidelines](https://www.iscaconf.org/isca2026/submit/guidelines.php)
- [arXiv Policy on ChatGPT (Jan 2023)](https://blog.arxiv.org/2023/01/31/arxiv-announces-new-policy-on-chatgpt-and-similar-tools/)
- [PMC12170296 — Defining Boundaries of AI Use in Scientific Writing (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12170296/)
- [AMEE Guide No.192 — When and How to Disclose AI Use (2025)](https://pubmed.ncbi.nlm.nih.gov/41467560/)
- [Monperrus — Policies on Generative AI for Scholarly Writing](https://www.monperrus.net/martin/generative-ai-scientific-writing)
- [Thesify — AI Policies in Academic Publishing 2025](https://www.thesify.ai/blog/ai-policies-academic-publishing-2025)
