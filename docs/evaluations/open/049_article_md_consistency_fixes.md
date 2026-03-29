ID: 049
Title: Auditoria completa ARTICLE.md — correções para publicação
Type: documentation
Priority: Critical
Status: Open

Descrição:
ARTICLE.md é a fonte de verdade do projeto. Todos os derivados (ICASSP, TASLP, PROPOR, CSL) devem ser gerados a partir dele. Este ticket documenta TODOS os problemas encontrados na auditoria de março 2026, organizados por camada de criticidade.

**Princípio**: consertar ARTICLE.md uma vez → todos os formatos derivados ficam corretos.

---

## CAMADA 0 — Inconsistências factuais (bloqueiam qualquer publicação)

### [C0-A] §7.3 Tabela de Generalização — 14/31 → 17/31
**Arquivo**: ARTICLE.md, linha ~849–856
**Problema**: tabela não foi atualizada após auditoria ɣ/x (§6.2), que corrigiu 4 entradas
**Estado atual** (errado):
```
| Generalização PT-BR | 4/9 (44%)  | 97% | Near-misses: ɣ→x, ĩ→i |
| Controles (em treino)| 3/4 (75%) | 98% | borboleta: ɣ→x |
| Total: 14/31 corretas (45%) |
```
**Correção** (verificado em §6.2 do próprio ARTICLE.md):
```
| Generalização PT-BR  | 6/9 (67%)  | 97% | Near-misses: ĩ→i, fantabulástico |
| Controles (em treino)| 4/4 (100%) | 100%| borboleta ✓ (ɣ/x correto) |
| Total: 17/31 corretas (55%) |
```
**Fonte da correção**: ARTICLE.md §6.2 documenta a auditoria; ICASSP_DRAFT.md §6.3 tem a versão correta.

### [C0-B] §7.4 Parágrafo — header desatualizado
**Linha ~860**: "**Generalização PT-BR (4/9)**" → deve ser "(6/9)"

### [C0-C] §9 Conclusões — "limite" inconsistente com §6.2
**Linha ~1054**: "Confusão `ɣ`→`x` em coda → erro sistemático de vozeamento velar"
**Problema**: §6.2 demonstrou que o modelo está CORRETO na distribuição ɣ/x — não é erro sistemático.
**Correção**: remover este item dos "limites" ou refraseá-lo: "Distinção ɣ/x em coda: verificada como comportamento correto após auditoria (§6.2). Casos residuais podem persistir em edge cases não auditados."

### [C0-D] §5.1 WER — referência @yao2015sequence incorreta
**Linha ~563**: `Referências: [@bisani2008joint; @yao2015sequence]`
**Problema**: yao2015sequence é um paper de tradução automática (machine translation), não G2P/WER.
**Correção**: remover @yao2015sequence desta linha; manter apenas @bisani2008joint (que define WER/SER para G2P).
**Nota**: @yao2015sequence está em REFERENCES.bib mas o título é errado para este contexto.

---

## CAMADA 1 — Qualidade de texto (necessário para publicação externa)

### [C1-A] §8.3 Encoding corruption
**Linhas ~922–935**: acentos faltantes em todo o parágrafo
**Exemplos**: "sobre-interpretacao" → "sobre-interpretação", "hipotese" → "hipótese", "e usada" → "é usada", "fonetico" → "fonético", "avaliacao" → "avaliação", "nao" → "não", "fonetica" → "fonética"
**Ação**: revisar linha a linha todo o §8.3 e restaurar acentuação.

### [C1-B] §5.5 — Parágrafo de lab notebook no corpo do artigo
**Linhas ~681–682**: "Aqui a estratificação também seguia a lógica de fazer um random do corpus, e quebrar em pedaços como a maioria dos G2P ciram nos artigos. No experimento 1, foi feito..."
**Problema**: texto coloquial e em primeira pessoa misturado com seção de métodos IMRaD.
**Ação**: remover completamente. O conteúdo relevante já está em §2.2 (Divisão e Estratificação).

### [C1-C] §9 Conclusões — typo
**Linha ~1031**: "empiricos" → "empíricos"

### [C1-D] §4.2 — Título de seção incompleto / mal formatado
**Linha ~372**: `### 4.2 Distance-Aware Phonetic Loss (DA Loss) [DA_LOSS_ANALYSIS.md](./DA_LOSS_ANALYSIS.md).`
O link está no título da seção, que não é o padrão correto. Mover link para nota de rodapé no final da seção.

---

## CAMADA 2 — Completude científica (necessário para journal/TASLP)

### [C2-A] Abstract em inglês (ausente)
ARTICLE.md tem apenas resumo em português. Para submission a venues internacionais (TASLP, CSL, ICASSP), o abstract em inglês é obrigatório.
**Ação**: adicionar abstract em inglês ao início do documento (pode copiar/adaptar do ICASSP_DRAFT.md §Abstract, que está em inglês e revisado).

### [C2-B] Statement de reprodutibilidade (ausente)
IEEE TASLP exige declaração de reprodutibilidade: "o código e os dados usados para produzir os resultados estão disponíveis em [URL]".
**Ação**: adicionar parágrafo em §9 ou nova seção "Reprodutibilidade": documentar que modelos estão em `models/`, código em `src/`, e corpus em `dicts/pt-br.tsv`. Adicionar instruções mínimas de reprodução.

### [C2-C] Acknowledgements / Agradecimentos (ausente)
Para submissão a venues formais, é necessário declarar: financiamento (se houver), instituição afiliada, contribuições. Para a versão de pesquisa pessoal, pode ser mínimo: "Este trabalho foi desenvolvido como projeto de pesquisa independente."
**Ação**: adicionar seção "Agradecimentos" antes das Referências.

### [C2-D] §1.1 Framing da comparação com LatPhon
**Linha ~68**: "**Resultado estatístico**: O limite **superior** do IC de FG2P (0,51%) está **abaixo** do limite **inferior** do IC de LatPhon (0,56%) — **diferença estatisticamente significativa a 95% de confiança**."
**Problema**: os dois ICs foram calculados em amostras completamente diferentes (28.782 vs ~500 palavras). ICs não-sobrepostos de amostras independentes NÃO implicam diferença significativa no sentido de teste de hipóteses clássico. A afirmação técnica dos ICs está correta; o label "estatisticamente significativa" é impreciso.
**Correção** (já feita no ICASSP_DRAFT.md; trazer para ARTICLE.md): "Os ICs de Wilson não se sobrepõem no cenário reportado (limite superior FG2P: 0,51% < limite inferior LatPhon: 0,56%). Esta comparação é indicativa, não confirmatória: os test sets diferem substancialmente em tamanho e amostragem."

### [C2-E] §10 Guia de Uso — não pertence ao artigo científico
§10 (linhas ~1090–fim) documenta o CLI `inference_light.py` em detalhe operacional. Este conteúdo pertence a um README técnico/documentação do projeto, não a um artigo científico.
**Ação**: marcar §10 explicitamente como "Apêndice técnico — não incluir em versões para publicação" ou mover para README do projeto.
**Nota**: esta seção NÃO deve ser incluída em nenhuma versão derivada (ICASSP, TASLP, CSL).

---

## CAMADA 3 — Formatos derivados (adaptar após C0+C1 concluídos)

### [C3-A] ICASSP 2027 — corte de conteúdo necessário
**Problema**: ICASSP_DRAFT.md atual tem 5 páginas de conteúdo. Limite real do ICASSP é **4 páginas de conteúdo + 1 página de referências**.
**Identificação do corte** (~1 coluna IEEEtran de conteúdo):
- §3 Architecture: remover equações LSTM detalhadas (manter apenas Bahdanau), cortar descrição dos vocabulários
- §4.3 λ sweep: converter tabela para 1 linha de texto inline
- §6.1 Error patterns: manter top-3 substituições em vez de top-5
- §6.2 Error Quality: manter apenas tabela, cortar análise e↔ɛ
**Deadline**: ~setembro 2026 (ICASSP 2027 Toronto, 16–21 mai 2027)
**Portal**: https://2027.ieeeicassp.org/

### [C3-B] IEEE/ACM TASLP — versão expandida (rolling submission)
**Formato**: 8–12 páginas, double-column IEEE, inglês
**Expansões a partir de ARTICLE.md**:
- Incluir §2.25 (Memorização vs Aprendizado) como seção completa
- Incluir §2.4 (Cross-validation protocol)
- Incluir §5.3 (Métricas graduadas PER_w/WER_g) completo
- Incluir tabela de ablações completa (todos os 14 experimentos)
- Incluir análise estatística neutralização e↔ɛ (§6.1 do ARTICLE.md)
- Adicionar reproducibility statement e código disponível
- Remover §10 (CLI guide)
**Portal**: https://mc.manuscriptcentral.com/tasl-ieee
**Ação imediata**: submissão possível agora, sem deadline fixo.

### [C3-C] Computer Speech & Language (CSL) — alternativa TASLP
**Formato**: ~20 páginas, inglês ou português
**Diferencial vs TASLP**: pode incluir análise fonológica PT-BR mais profunda (§2.3 auditoria corpus, PHONOLOGICAL_ANALYSIS.md), discussão linguística, exemplos OOV
**Portal**: https://www.sciencedirect.com/journal/computer-speech-and-language
**IF**: 3.4 (Q2); TASLP é mais prestígio mas CSL tem G2P como tema central

### [C3-D] PROPOR 2028 — versão PT-BR completa
**Formato**: LNCS 12–15 páginas, português ou inglês
**Estimativa de deadline**: ~setembro 2027
**Diferencial**: audiência PT-BR, pode escrever em português, análise fonológica expandida

---

## Status de execução

### ✅ CONCLUÍDO — C0 (Factual) e C1 (Texto)
Commits `66b127c` → restaurado em `74a36a4` (v1.0/pub) e `8324fc3` (cleanup) após reversão acidental.

- [x] C0-A: §7.3 tabela → 17/31 (55%), 6/9, 4/4
- [x] C0-B: §7.4 header → (6/9)
- [x] C0-C: §9 — removido "ɣ→x erro sistemático"
- [x] C0-D: §5.1 — removido @yao2015sequence
- [x] C1-A: §8.3 — encoding restaurado
- [x] C1-B: §5.5 — lab notebook removido
- [x] C1-C: §9 — "empiricos" → "empíricos"
- [x] C1-D: §4.2 — link movido do título para nota
- [x] §6.4: "DA Loss com embeddings PanPhon" → "DA Loss"
- [x] §10: marcado como apêndice técnico não-publicável
- [x] Referências: seção substituída por ponteiro para REFERENCES.bib

**Branches atualizadas**: `cleanup/initial-prune-20260318`, `v1.0/publication`.

---

### ⬜ PENDENTE — C2 (Completude para journal/TASLP)

Cada passo é independente e verificável antes de prosseguir:

**Passo 1 — C2-D** (framing crítico, fazer primeiro):
- §1.1 linha ~68: substituir "diferença estatisticamente significativa a 95%" pela formulação do ICASSP_DRAFT.md: "Esta comparação é indicativa, não confirmatória: os test sets diferem substancialmente em tamanho e amostragem."
- Verificar: `grep "indicativa, não confirmatória" docs/article/ARTICLE.md`

**Passo 2 — C2-A** (abstract EN):
- Adicionar abstract em inglês antes de "## 1. Introdução", adaptado do §Abstract em `icassp/ICASSP_DRAFT.md`.
- Verificar: seção "Abstract" (inglês) existe antes de "## 1."

**Passo 3 — C2-B** (reprodutibilidade):
- Adicionar em §9: "Reprodutibilidade: código em `src/`, corpus em `dicts/pt-br.tsv`, modelos em `models/`. Experimento reproduzível via `python src/inference_light.py --index 18 --word <palavra>`."
- Verificar: `grep "Reprodutibilidade" docs/article/ARTICLE.md` retorna linha em §9.

**Passo 4 — C2-C** (agradecimentos):
- Adicionar "## Agradecimentos" antes de "## Referências": "Este trabalho foi desenvolvido como projeto de pesquisa independente."
- Verificar: seção existe imediatamente antes de Referências.

---

### ⬜ PENDENTE — C3 (Derivados)
C3-A (ICASSP 2027), C3-B (TASLP), C3-C (CSL), C3-D (PROPOR) — ver tickets 048, 053.

---

## Ordem de execução

```
✅ C0 + C1   FEITO — branches atualizadas
⬜ C2: passos 1→2→3→4 (cada um verificável)
→ ARTICLE.md completo para qualquer submissão
⬜ C3: derivados por venue (tickets 048, 053)
```

Dependências:
- docs/article/ARTICLE.md (fonte)
- docs/article/icassp/ICASSP_DRAFT.md (referência para C2-A, C2-D)
- docs/article/REFERENCES.bib (fonte canônica de referências)

Notas sobre REFERENCES.bib:
- @byt5g2p ✓ existe
- @yao2015sequence removido do texto principal (paper de MT, não G2P)
- @vasilевски2012phonologic tem caractere cirílico na chave — pode causar problemas em BibTeX/LaTeX
