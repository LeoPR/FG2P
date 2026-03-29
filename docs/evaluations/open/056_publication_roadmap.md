ID: 056
Title: Roadmap de publicações — plano integrado com v2.0
Type: reference
Priority: Critical
Status: Open

Descrição:
Plano completo de publicação, priorizando ARTICLE.md como fonte de verdade. Publicação e v2.0 são prioridades paralelas — ARTICLE.md perfeito beneficia ambas.

---

## Princípio arquitetural

```
ARTICLE.md (fonte de verdade, PT-BR, ~1100 linhas)
    ↓ adaptar
    ├── ICASSP 2027 (4+1p, EN, double-blind, IEEEtran)
    ├── IEEE/ACM TASLP (8-12p, EN, rolling, IEEE journal)
    ├── Computer Speech & Language (20p, EN, rolling, Elsevier)
    └── PROPOR 2028 (12-15p, PT/EN, LNCS, abr 2028 estimado)
```

**Regra**: NÃO editar derivados diretamente sem antes corrigir ARTICLE.md.
Se ARTICLE.md estiver correto → derivados ficam corretos por construção.

---

## Status de deadlines verificados (março 2026)

### ❌ Conferências 2026 — todos passados

| Venue | Deadline | Conferência | Portal |
|-------|---------|-------------|--------|
| ICASSP 2026 | 17 set 2025 | Barcelona, 4–8 mai 2026 | [2026.ieeeicassp.org](https://2026.ieeeicassp.org/) |
| PROPOR 2026 | 16 nov 2025 | Salvador BR, 13–16 abr 2026 | [propor2026.ufba.br](https://propor2026.ufba.br/) |
| Interspeech 2026 | 25 fev 2026 | Sydney, 28 set–1 out 2026 | [interspeech2026.org](https://interspeech2026.org/) |
| EACL 2026 | jan 2026 | Rabat, 24–29 mar 2026 | [2026.eacl.org](https://2026.eacl.org/) |
| LREC 2026 | nov 2025 | Palma, 11–16 mai 2026 | [lrec2026.info](https://lrec2026.info/) |

### ✅ Opções abertas AGORA — journals rolling

| Venue | Impacto | Páginas | Deadline | Portal |
|-------|---------|---------|---------|--------|
| **IEEE/ACM TASLP** | ~CiteScore 9 | 8–12p IEEE | Rolling | [mc.manuscriptcentral.com/tasl-ieee](https://mc.manuscriptcentral.com/tasl-ieee) |
| **Computer Speech & Language** | IF 3.4 / CiteScore 6.7 | ~20p | Rolling | [sciencedirect.com/journal/computer-speech-and-language](https://www.sciencedirect.com/journal/computer-speech-and-language) |
| **Speech Communication** | IF ~3 | ~20p | Rolling | [sciencedirect.com/journal/speech-communication](https://www.sciencedirect.com/journal/speech-communication) |

**Recomendação**: IEEE/ACM TASLP é o target mais forte. Mesmo publisher do ICASSP. G2P papers publicados regularmente. Reproducible code requerido → preparar.

### 📅 Conferências futuras — planejar agora

| Venue | Local | Deadline esperado | Formato | Portal |
|-------|-------|-----------------|---------|--------|
| **ICASSP 2027** | Toronto, 16–21 mai | ~set 2026 | **4p+1p refs** IEEEtran | [2027.ieeeicassp.org](https://2027.ieeeicassp.org/) |
| **Interspeech 2027** | TBD | ~fev 2027 | 4-5p ISCA | interspeech.org |
| **SSW14** (Speech Synthesis Workshop) | TBD 2027 | ~mai 2027 | workshop | isca-speech.org |
| **PROPOR 2028** | Portugal (estimado) | ~set 2027 | 12-15p LNCS | propor.org.br |

⚠️ **Bug de formato no draft atual**: ICASSP_DRAFT.md foi escrito com 5p de conteúdo, mas o limite real do ICASSP é **4p de conteúdo + 1p de referências**. Precisa de corte antes de ICASSP 2027.

---

## Plano de execução — 4 fases

### FASE 1: ARTICLE.md perfeito (pré-requisito de tudo)
*Estimativa: 3–4 horas | Responsável: ticket 049*

```
Semana 1:
[P0] C0-A: §7.3 table: 14/31 → 17/31 (inconsistência factual crítica)
[P0] C0-B: §7.4: header (4/9) → (6/9)
[P0] C0-C: §9 limites: remover/refrasecar "ɣ→x erro sistemático" (contradição com §6.2)
[P0] C0-D: §5.1: remover @yao2015sequence da citação de WER

Semana 1 (cont.):
[P1] C1-A: §8.3: restaurar acentuação (encoding corruption)
[P1] C1-B: §5.5: remover parágrafo coloquial de lab notebook
[P1] C1-C: §9: "empiricos" → "empíricos"
[P1] C1-D: §4.2: mover link do título para nota da seção

Semana 2:
[P2] C2-A: adicionar abstract em inglês (adaptar de ICASSP_DRAFT.md)
[P2] C2-D: refrasear §1.1 comparação LatPhon (usar framing ICASSP_DRAFT.md)
[P2] C2-B: adicionar reproducibility statement em §9
[P2] C2-C: adicionar seção Agradecimentos mínima
[P2] C2-E: marcar §10 como "apêndice técnico não-publicável"
```

### FASE 2: Derivado TASLP (rolling — submeter assim que Fase 1 concluída)
*Estimativa: 8–12 horas de redação | Responsável: ticket 053*

Expansões desde ARTICLE.md:
- §2.25 (Memorização vs Aprendizado) → seção 2.3 no TASLP
- §2.4 (cross-validation protocol) → inline em §2
- §5.3 (PER_w/WER_g completo) → seção 5.3 no TASLP
- Tabela de ablações completa (14 experimentos)
- Análise estatística e↔ɛ (tabela de posições)
- Benchmark de throughput (breve, §5 ou Appendix)
- Reproducibility statement: código + dados + modelos
- Remover: §10 (CLI guide), referências a arquivos .md internos

### FASE 3: Derivado ICASSP 2027 (preparar quando deadline anunciado ~set 2026)
*Estimativa: 4–6 horas | Responsável: ticket 048*

Cortes desde ICASSP_DRAFT.md (5p → 4p):
- §3 Architecture: remover equações LSTM, manter apenas Bahdanau + tabela de configs
- §4.3: λ sweep table → inline (1 frase)
- §6.1: top-3 erros em vez de top-5
- §6.2: manter tabela, condensar análise e↔ɛ em 2 linhas
Verificar novo template quando disponível (não usar template 2026).

### FASE 4: Derivados de longo prazo (após Fase 2)
- PROPOR 2028: versão portuguesa expandida com análise fonológica completa
- Interspeech 2027: versão multilíngue (depende tickets 039/040/054)
- SSW14 2027: versão TTS-focused (depende ticket 052 — perceptual validation)

---

## Relação com v2.0 do projeto

Publicação e v2.0 são PARALELAS, não sequenciais:

| Trabalho de Publicação | Trabalho v2.0 |
|------------------------|---------------|
| Corrigir ARTICLE.md (Fase 1) | Não interfere com código |
| Escrever TASLP version (Fase 2) | Não interfere com código |
| Perceptual validation proxy (ticket 052) | Usa src/analyze_errors.py existente |
| Geminate corpus (ticket 055) | Usa dicts-workbench (já em v2.0) |
| DA Loss multilingual (ticket 054) | Resultado natural de tickets 039/040/041 |

O que v2.0 produz que fortalece publicação futura:
- Sistema multilíngue → paper Interspeech 2027 / TASLP v2
- Corpus expandido (geminadas) → melhora OOV section
- Perceptual validation → fortalece DA Loss claim em journal
- 7D articulatory space → futuro paper metodológico (ticket 025)

---

## Checklist de publicação rápida (agora)

Para submeter ao TASLP nas próximas semanas:
- [ ] Fase 1 completa (ticket 049)
- [ ] Abstract em inglês adicionado ao ARTICLE.md
- [ ] Reproducibility statement redigido
- [ ] TASLP draft criado a partir de ARTICLE.md
- [ ] Revisão por pares informal (colega ou co-autor)
- [ ] Conta no ScholarOne criada: https://mc.manuscriptcentral.com/tasl-ieee
- [ ] Código anonimizado em repositório (GitHub privado ou Zenodo)

Dependências:
- 048 (ICASSP 2027), 049 (ARTICLE.md fixes), 050 (PROPOR), 051 (Interspeech), 052 (perceptual), 053 (TASLP), 054 (multilingual paper), 055 (geminate)
