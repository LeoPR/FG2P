# Publications — Derivados para Submissao

Esta pasta contem os derivados do meta-artigo (ARTICLE.md) formatados para
cada venue de publicacao. O ARTICLE.md e a fonte canonica — correcoes la
propagam para todos os derivados.

## Regras de Anonimato (resumo — detalhes em DOUBLE_BLIND_POLICIES.md)

**O repositorio GitHub NAO precisa ser privado.** Nenhum venue exige isso.

A restricao e apenas sobre o conteudo do paper submetido:
- **Double-blind** (SLT, STIL, INTERSPEECH, ACL/ARR): nao incluir nomes, afiliacoes
  ou link identificavel do GitHub no paper.
- **Single-blind** (ICASSP, TASLP): incluir nomes (obrigatorio). Sem restricoes.

Periodos de anonimato foram **eliminados** pelo INTERSPEECH (2026) e ACL/ARR
(desde fev 2024). Preprints no arXiv sao permitidos durante review.

---

## Politicas de Uso de IA Generativa — Consolidacao por Venue

Todos os venues permitem o uso de IA como ferramenta auxiliar. Nenhum permite
IA como autor. Autores assumem total responsabilidade pelo conteudo.

| Venue | Onde declarar | Edicao/gramatica | Geracao de conteudo | Fonte |
|-------|---------------|------------------|---------------------|-------|
| **IEEE** (SLT, ICASSP, TASLP) | Acknowledgments | Isento (recomendado) | Obrigatorio: identificar IA e secoes | [IEEE Author Center](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/) |
| **STIL/BRACIS** (SBC) | Secao propria obrigatoria | Obrigatorio (pagina extra permitida) | Obrigatorio | [BRACIS 2026 CFP](https://bracis.sbc.org.br/2026/bracis/) |
| **INTERSPEECH** (ISCA) | Acknowledgments | Isento (recomendado) | Obrigatorio | [ISCA 2026 Guidelines](https://www.iscaconf.org/isca2026/submit/guidelines.php) |
| **ACL/EMNLP** | Checklist + apendice | Isento | Obrigatorio (publicado como apendice) | [ACL 2023 Policy](https://2023.aclweb.org/blog/ACL-2023-policy/) |
| **arXiv** | No corpo do paper | Recomendado | Obrigatorio ("significant use") | [arXiv Blog](https://blog.arxiv.org/2023/01/31/arxiv-announces-new-policy-on-chatgpt-and-similar-tools/) |

### O que declarar no nosso caso

Uso de IA neste projeto:
- Organizacao estrutural de texto e publicacoes
- Revisao de consistencia terminologica e cientifica
- Formatacao LaTeX e conversao de formatos
- Busca de inconsistencias entre secoes
- Tarefas administrativas (pesquisa de venues, deadlines, politicas)

**NAO** usamos IA para: design experimental, implementacao do codigo, execucao
de experimentos, analise dos resultados, ou conclusoes cientificas.

### Modelo de declaracao para cada tipo de venue

**IEEE (Acknowledgments)**:
> Generative AI (Claude, Anthropic) was used as an assistive tool for
> manuscript organization, terminology review, and LaTeX formatting. All
> scientific content — experimental design, implementation, execution,
> analysis, and conclusions — is the sole work of the authors.

**SBC/STIL (secao propria)**:
> IA generativa (Claude, Anthropic) foi utilizada como ferramenta auxiliar
> em tarefas de: organizacao estrutural do texto, revisao de consistencia
> terminologica e formatacao LaTeX. Todo o conteudo cientifico e de autoria
> exclusiva dos autores.

**ISCA/INTERSPEECH (Acknowledgments)**: Mesmo texto do IEEE.

**arXiv**: Incluir no corpo do paper ou acknowledgments.

---

## Estrutura

```
publications/
├── README.md                  ← ESTE ARQUIVO
├── PUBLICATION_PLAN.md        ← Calendario, prioridades, pre-requisitos
├── DOUBLE_BLIND_POLICIES.md   ← Politicas oficiais com citacoes e URLs
├── icassp/                    ← Target: ICASSP 2027 (~out 2026) — single-blind
├── slt/                       ← IEEE SLT 2026 (deadline 17 jun) — double-blind
├── stil/                      ← STIL 2026 (deadline 20 abr) — double-blind
├── interspeech/               ← INTERSPEECH 2027 (~fev 2027) — double-blind
└── taslp/                     ← IEEE/ACM TASLP journal (rolling) — single-blind
```

## Convencoes

- Cada subpasta = 1 venue (sem ano no nome — README interno tem historico)
- README.md de cada venue: especificacoes, processo de submissao, politica de IA, checklist
- *_DRAFT.md: draft em markdown (intermediario para LaTeX)
- main.tex: LaTeX final para compilacao
- *_refs.bib: subset de REFERENCES.bib

## Privacidade

Esta pasta esta no .gitignore por higiene (evitar drafts incompletos no GitHub),
nao por exigencia das editoras. Apos aceitacao de um paper, pode-se remover
a entrada correspondente do .gitignore para publicar no repositorio.
