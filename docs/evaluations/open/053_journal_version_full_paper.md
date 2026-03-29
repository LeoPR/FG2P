ID: 053
Title: Versão journal — Computer Speech and Language (paper completo)
Type: publication
Priority: Medium
Status: Open

Descrição:
O ICASSP paper (5p) é necessariamente comprimido. Uma versão journal de ~20-25 páginas permitiria incluir: todas as ablações, análise de erros completa, corpus pipeline, fonologia PT-BR, discussão de generalização, perceptual validation e multilingual extension. O venue natural é Computer Speech and Language (CSL, Elsevier) — principal journal de processamento de fala e linguagem.

Alternativas de venue:
- **Computer Speech and Language (CSL)**: principal journal da área, indexed, impacto ~4.0. Foco em ASR/TTS/G2P é perfeito.
- **Speech Communication (Elsevier)**: similar, foco mais em acústica/fala.
- **TACL (Transactions of ACL)**: se foco em NLP, mas G2P tem menos relevância aqui.
- **JMLR**: se contribuição metodológica da DA Loss for o foco central (machine learning + fonologia).

Recomendação: CSL ou Speech Communication. CSL tem track record de papers G2P (Bisani & Ney 2008 — ref [5] do paper — foi publicado lá).

Conteúdo adicional vs. ICASSP:
1. **§2 expandido**: Pipeline completo de corpus (Unicode, PanPhon corrections, stratification details)
2. **§3 expandido**: Full ablation table (todos os 14 experimentos Exp0–Exp107)
3. **§4 expandido**: DA_LOSS_ANALYSIS.md completo — análise numérica, λ bounds, interaction with BiLSTM
4. **§5 expandido**: Métricas graduadas PER_w e WER_g com evolução cronológica completa
5. **§6 expandido**: Análise estatística da neutralização e↔ɛ (tabela de razões por posição prosódica)
6. **§7 expandido**: Generalização OOV com análise qualitativa completa por categoria
7. **§8 novo**: Perceptual validation (se 052 estiver concluído)
8. **§9 novo**: Multilingual DA Loss (se extensão estiver disponível)
9. **Appendix**: Corpus statistics, phoneme frequency distribution, full error taxonomy

Pré-requisitos para submissão forte:
- 049 (ARTICLE.md consistente como fonte)
- 052 (perceptual validation como diferencial)
- Ao menos 1 idioma adicional validando generalização do DA Loss (tickets 026/039)

## Venue recomendado: IEEE/ACM TASLP

**IEEE/ACM TASLP** (Transactions on Audio, Speech, and Language Processing):
- Submissão rolling — **pode submeter agora, sem deadline fixo**
- Formato: 8–12 páginas, double-column IEEE, inglês
- Portal: https://mc.manuscriptcentral.com/tasl-ieee

Alternativa se quiser português ou volume maior: **Computer Speech and Language (CSL, Elsevier)**.

---

## Passos incrementais (verificáveis individualmente)

**Pré-requisito**: ticket 049 Passo 1 (C2-D framing §1.1) completo.

**Passo 1 — Verificar Guide for Authors TASLP**:
- Acessar: https://signalprocessingsociety.org/publications-resources/ieee-transactions-audio-speech-and-language-processing/author-information-taslp
- Anotar page limit, double-blind policy, template LaTeX disponível.
- Verificar: anotações adicionadas em "Notas TASLP" abaixo.

**Passo 2 — Listar gaps entre ARTICLE.md e requisitos TASLP**:
- Ler §1–§9 do ARTICLE.md com olhar de revisor IEEE.
- Listar no máximo 5 pontos que precisam de expansão (ex: §Background, ablações completas).
- Verificar: lista documentada neste ticket.

**Passo 3 — Criar outline de journal version**:
- Seções e tamanho estimado com base em ARTICLE.md.
- Identificar quais seções são expansão vs. tradução EN direta.
- Verificar: outline existe neste ticket.

**Passo 4 — Rascunho §1–§4** (EN, Introdução + Background + Corpus + Método):
- Base: ARTICLE.md §1–§4 traduzido e expandido conforme outline.
- Verificar: arquivo `docs/article/taslp/draft_sections_1_4.md` existe.

**Passo 5 — Rascunho §5–§7** (EN, Resultados + Generalização):
- Tabela de ablações completa (14 experimentos), análise e↔ɛ.
- Verificar: arquivo `docs/article/taslp/draft_sections_5_7.md` existe.

**Passo 6 — Rascunho §8–§9 + Referências** (EN, Discussão + Conclusões):
- Reprodutibilidade, limitações, trabalhos futuros.
- Referências: REFERENCES.bib revisadas para formato TASLP.
- Verificar: paper completo compilável em LaTeX.

**Passo 7 — Submissão**:
- Submeter em: https://mc.manuscriptcentral.com/tasl-ieee
- Registrar número de manuscript ID neste ticket.

---

## Notas TASLP (preencher no Passo 1)

- Page limit: ___
- Double-blind: ___
- Template LaTeX: ___

---

Critérios de aceite:
- Passo 1 completo (Guide for Authors verificado).
- Passo 3 completo (outline aprovado).
- Passo 7: número de manuscript ID registrado.

Dependências obrigatórias:
- 049 (ARTICLE.md com abstract EN, reprodutibilidade — C2 completo)

Dependências que fortalecem mas não bloqueiam:
- 052 (perceptual validation como diferencial)
- 026/039 (extensão multilíngue — seção opcional)
