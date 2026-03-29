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

Passos:
- [ ] Mapear diferenças concretas entre ICASSP draft e journal version (conteúdo adicional)
- [ ] Verificar Guide for Authors do CSL (formato, página mínima/máxima, single-blind vs. double-blind)
- [ ] Definir momento de submissão (após ICASSP response ou em paralelo?)
- [ ] Adaptar ARTICLE.md como base para journal version

Critérios de aceite:
- Decisão sobre venue documentada.
- Outline de journal version aprovado.
- Timeline de submissão definida.

Dependências:
- 048, 049 (ICASSP e ARTICLE.md)
- 052 (perceptual validation)
- 026, 039, 040 (multilingual extension para versão mais forte)
