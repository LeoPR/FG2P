# Avaliacoes do Projeto

Esta pasta organiza avaliacoes tecnicas do FG2P em formato incremental.

Objetivo:
- registrar perguntas de avaliacao que surgem durante a pesquisa;
- separar o que ja foi respondido do que ainda precisa de evidencias ou calculos;
- manter um historico de raciocinio que possa depois ser consolidado no artigo ou removido se ficar obsoleto.

Convencao:
- `answered/`: perguntas com resposta documentada, ainda que sujeita a refinamento;
- `open/`: perguntas, riscos, hipoteses e pendencias que merecem investigacao adicional.

## Indice de Perguntas

| ID | Tema | Status | Observacao |
|----|------|--------|------------|
| 001 | Robustez da pesquisa: pontos fortes e fracos | Respondida parcialmente | Sintese inicial ja consolidada; pode ser refinada depois |
| 002 | Intervalo de confianca do WER | Respondida | Implementado em analyze_errors.py; formulas centralizadas em docs/article/FORMULAS.md |
| 003 | Tamanho minimo de amostra para teste e treino | Respondida | Fechada como criterio metodologico pratico; minimo universal fica como trabalho futuro |
| 004 | Validade externa, comparacao classica e generalizacao | Respondida | Formula final: alta para comparacao no universo `ipa-dict`, moderada para generalizacao ampla |
| 005 | Estratificacao incompleta e fatores ocultos | Respondida | Fator de seguranca mantido como principio prudencial, sem valor universal fixado |
| 006 | Curva de estabilidade com menos dados | Respondida | Evidencia descritiva consolidada com Exp104b/105/106; pontos intermediarios ficam para estudo futuro |
| 007 | Por que LatPhon nao reporta WER | Respondida | Paper usa PER como metrica principal e estatistica no nivel de fonema |
| 008 | DA Loss: forca, limites e narrativa justa | Respondida | Narrativa final separa ganho da DA Loss em comparacoes limpas do efeito combinado com override estrutural |
| 009 | Regras para comparacoes empiricas justas | Respondida | Checklist aplicado nas comparacoes principais de README e ARTICLE |
| 010 | Matriz consolidada de reivindicacoes | Respondida | Claims frageis principais foram matizados e a delimitacao formal de originalidade foi registrada |
| 011 | Questionario de fechamento por etapas | Respondida | Decisoes editoriais principais aplicadas no README/ARTICLE e no indice de avaliacoes |
| 012 | Clareza narrativa e nomenclatura | Respondida | Regra geral para ordem da historia do projeto e uso consistente de rotulos (comparacao externa vs interna) |
| 013 | Checklist final de submissao (Ciclo 3.2) | Respondida | Coerencia final validada; consistencia numerica e referencias quebradas ajustadas |
| 014 | Auditoria da narrativa de velocidade (Exp106) | Respondida | Fechada com narrativa por regime (hardware+batches+modelo), sem claim universal de speedup isolado |
| 015 | Consistencia e ganho real: Exp104d vs Exp104b | Respondida | Exp104d melhora estimativas pontuais de qualidade, mas nao domina em todos os eixos (IC95 de WER sobreposto e custo maior em CPU) |
| 016 | Estudo de performance cross-platform (train + inference) | Respondida | Auditoria tecnica sem alteracao de codigo; prioriza AMP, batching real de inferencia e trilha de quantizacao CPU |
| 017 | Benchmark por regime (latencia unitária vs volume CPU/GPU) | Respondida | Define decomposicao de overhead (Python/H2D/sync/decode), protocolo R1-R4 e plano de otimizacao para aceleracao real |
| 018 | Guia prático de inferência: CPU, batch e métricas de desempenho | Respondida | Sweeps formais CPU (2026-03-15) + GPU (2026-03-14): GPU superior a partir de batch≥4 em todos os modelos; em batch=1 depende do modelo (0.78×–1.45×); Exp104d pico GPU 1.106 w/s / CPU 190 w/s |
| 019 | Real-World Use Case do G2P: escopo, taxonomia e metricas | Respondida | Escopo fechado: G2P como camada fonetica em pipeline maior, com limites de inferencia explicitos e metrica nova mantida como trabalho futuro |
| 020 | Proporcionalidade chars/s e condições de pico por hardware | Respondida | Benchmark consolidado: chars/s não linear em batch=1; picos operacionais definidos (GPU≈512, CPU≈64–128); profiling fino de núcleos fica como micro-otimização opcional |
| 021 | Auditoria de publicação v1.1 — consistência, links, sincronização | Aberta | 4 itens de alta prioridade (registry Exp9 duplo, PER origin, 104c vs 104d, citações); 6 itens de média; correções imediatas já aplicadas em README/BENCHMARK |
| 022 | Trabalhos Futuros v2.0: Métricas, Fonotática, Espaço 7D, Multilíngue | Aberta | Consolidação de 6 tópicos principais da seção 9.1 de ARTICLE.md: métricas especializadas, pipeline fonotático 4-fases, espaço articulatório 7D (transfer learning zero/few-shot para novos idiomas incluindo Tupi), geminadas, batch stratification, morfossintaxe. Documentação canônica nesta avaliação. Ver seção "Trabalhos Futuros Consolidados" abaixo. |

## Proximas acoes (estado atual)

1. Estado global: pronto para publicação no GitHub; 4 pendências abertas (ver avaliação 021) antes de submissão formal a conferência/revista.
2. Manter MOS/ABX como trabalho futuro explicito (nao bloqueante).
3. Benchmark formal completo (CPU 2026-03-15 + GPU 2026-03-14) integrado em todos os documentos principais.
4. Pendências de 021: verificar registry Exp9 (dois checkpoints); confirmar PER canônico de Exp9; documentar 104c vs 104d; converter citações para \cite{} antes de submissão formal.
4. Registrar Exp104d como candidato de melhor qualidade media vs Exp104b, sem promover como "melhor em todos os sentidos" ate replicacao com menor ruido e politica de custo/latencia definida.

## Publicacao: bloqueadores e pendencias

Leitura pragmatica para submissao/publicacao:

### Bloqueadores reais de publicacao

1. Nenhum bloqueador experimental aberto no estado documental atual.

Observacao: ajustes editoriais de velocidade (nota 014) e caso de uso real (nota 019) foram encerrados e registrados como respondidos.

Observacao: a delimitacao formal de originalidade da DA Loss foi registrada em `docs/article/ORIGINALITY_ANALYSIS.md` e consolidada na nota 010.

### Pendencias importantes, mas que podem virar limitacao explicita

1. **Validacao perceptual (MOS/ABX)**
	- nao precisa bloquear o texto se ficar explicitamente como trabalho futuro e se as afirmacoes auditivas forem suavizadas

### O que da para fechar agora, sem novo experimento

1. Registrar MOS/ABX e curva de estabilidade intermediaria como trabalho futuro, nao como requisito imediato.
2. Fazer revisao cruzada curta de wording de limitacoes entre README, ARTICLE e evaluations.
3. Alinhar wording de speed do Exp106 (README/ARTICLE/EXPERIMENTS) com o status de evidencia atual (nota 014).

### O que exige trabalho adicional antes de chamar de "pronto para publicar"

1. Nenhum item bloqueante adicional no estado atual.

## Plano em ciclos ate concluir tudo

### Ciclo 1 - concluido

1. Revisao editorial de claims no README e ARTICLE.
2. Separacao explicita entre efeito da DA Loss e efeito do override estrutural.
3. Fechamento formal de 008, 009 e 010.

### Ciclo 2 - fechamento sem novo experimento (concluido)

1. Fechar 004 com redacao final de validade externa:
	- alta para comparacao no universo `ipa-dict`;
	- moderada para generalizacao ampla.
2. Fechar 003, 005 e 006 como bloco de limitacoes metodologicas + futuro trabalho.
3. Fechar 011 registrando as decisoes editoriais finais ja aplicadas.

### Ciclo 3 - pronto para publicar

1. Revisao curta de originalidade da DA Loss com fontes primarias no related work. (concluido)
2. Checklist final de submissao (coerencia de claims, limitacoes, reproducibilidade, referencias). (concluido; nota 013)
3. Marcar status global como "pronto para submissao". (concluido)

## Uso esperado

Quando surgir uma nova pergunta, adicionar primeiro uma nota em `open/` com:
- problema;
- por que importa;
- evidencias ja conhecidas;
- o que falta para responder com mais rigor.

Quando a pergunta estiver madura, mover o conteudo para `answered/` ou consolidar no artigo.

---

## Trabalhos Futuros Consolidados (§ 9.1 ARTICLE.md)

**Nota de canonicidade**: Esta seção consolida todos os próximos passos mencionados em docs/article/ARTICLE.md§9.1, docs/article/EXPERIMENTS.md e comparativos sobre arquiteturas (Transformer, LatPhon, ByT5-Small). **Este é o local de referência única** — outros arquivos devem apenas referenciar este documento em vez de repetir detalhes.

### TF-001: Métricas Especializadas

**Objetivo**: Capturar aspectos de qualidade invisíveis ao PER/WER padrão.

- **Stress Accuracy**: % de acentos posicionados corretamente (sílaba tônica no lugar certo)
- **Boundary F1**: Precisão/revocação de fronteiras silábicas quando separadores são gerados

**Importância**: Algumas palavras podem ter PER baixo mas stress no lugar errado, quebrando compreensão em TTS.

**Implementação**: Adicionar métricas em `src/metrics.py` com suporte a avaliação por palavra completa vs por segmento.

---

### TF-002: Pipeline Fonotático em 4 Fases

**Objetivo**: Integrar restrições explícitas de fonotática do PT-BR para reduzir erros estruturais (separadores mal posicionados, conflitos com regras silábicas).

**Contexto de domínio**: PT-BR permite apenas ~4 consoantes em coda (/s/, /ɾ/, /l/, /N/), com ~78% das sílabas sendo abertas (CV).

**Ordem de implementação**:

1. **Diagnóstico** (risco zero): Quantificar quantos erros atuais violam regras fonotáticas conhecidas sem modificar o modelo
   - Dataset: 28.782 palavras de teste estratificado
   - Métrica: % de violações por categoria (inserção de sep, confusão positional, etc)

2. **N-gram Fonotático** (risco baixo): Treinar bigrama/trigrama de fonemas como modelo de linguagem fonológico
   - Treino: corpus de 95.937 pares (grapheme, IPA)
   - Objetivo: aprender sequências válidas de fonemas

3. **Autômato de Estados Finitos** (risco médio): FSA encoding estrutura silábica
   - Estados: ONSET → NUCLEUS → CODA → BOUNDARY
   - Válidade: rejeita transições ilegais em tempo real

4. **Integração** (risco médio-alto): Reranking dos N-melhores beams do LSTM
   - Abordagem 1 (baixo risco): Post-process reranking pelo modelo fonotático
   - Abordagem 2 (médio risco): Penalização direta como termo adicional na loss durante treino

**Referência técnica**: [docs/article/ARTICLE.md § 9.1 Pipeline fonotático](../article/ARTICLE.md#91-trabalhos-futuros)

---

### TF-003: Espaço Articulatório Contínuo 7D (Multilíngue + Transfer Learning)

**Objetivo**: Substituir PanPhon discreto (24 features binárias) por espaço contínuo 7D fundamentado em análise do trato vocal.

**Vantagem chave**: Espaço é **multilinguisticamente universal** — cada idioma é uma quantização (conjunto de fonemas) do mesmo espaço contínuo subjacente.

**Mapeamento de dimensões**:

| Dimensão | Range | Semântica | Correlato Acústico |
|----------|-------|-----------|--------------------|
| HEIGHT | [0,1] | Altura da língua | F1 (inverso) |
| BACKNESS | [0,1] | Posição anterior-posterior | F2 (inverso) |
| ROUNDING | [0,1] | Arredondamento labial | Timbre, F2 |
| CONSTR_LOC | [0,1] | Local de constrição (labial→glotal) | Transições F2/F3 |
| CONSTR_DEG | [-0.1,1] | Grau de constrição | Energia, fricção |
| NASALITY | [0,1] | Nasalização | Antirressonâncias em F1 |
| VOICING | [0,1] | Vozeamento binário | F0, estrutura harmônica |

**Conservação de variância**: 7D preservam 95-99% da variância do trato vocal (Birkholz et al., em prep).

**Resolução de problema estrutural**: `.` (boundary) e `ˈ` (stress) receberiam coordenadas distintas:
  - Ambos: CONSTR_DEG = -0,1 (estrutural, não-fonético)
  - Diferença: VOICING = 0,0 vs 1,0
  - Resultado: d(., ˈ) ≠ 0,0 naturalmente, sem necessidade de override

**Aplicações**:
- **Transfer Learning Zero-Shot**: modelo PT-BR puro pode fazer predições baseline para idiomas novos quantizando no espaço 7D
- **Few-Shot**: adaptar com N exemplos de novo idioma usando espaço contínuo como inicialização
- **Multilíngue Nativo**: treinar modelo único sobre múltiplos idiomas no espaço 7D compartilhado

**Correlação com referências comparativas**:
- LatPhon (2025): Transformer multilíngue para 6 idiomas, sem loss fonológica — pode ganhar com espaço 7D
- ByT5-Small (299M): zero-shot para 100 idiomas — seria complementar a transfer learning 7D para few-shot
- Este projeto (PT-BR): espaço 7D permitiria estender FG2P para Tupi-Guarani, outros idiomas indígenas, e variantes dialetais do PB

**Referência técnica**: [docs/article/ARTICLE.md § 9.1 Espaço articulatório contínuo 7D](../article/ARTICLE.md#91-trabalhos-futuros)

---

### TF-004: Ampliar Corpus de Geminadas

**Objetivo**: Cobrir gap identificado em avaliação de generalização (Exp OOV em ARTICLE.md).

**Contexto**: PT-BR têm geminadas principalmente em empréstimos italianos e ingleses:
- Italianos: *zz* (/dz/), *tt* (/t:/) em borrowings como "pizza", "patente"
- Ingleses: *pp*, *ss*, *tt* em palavras como "sipping", "kissing"

**Ação**: Expandir `dicts/pt-br.tsv` com ~500-1000 exemplos de geminadas certificadas de fonte fonética confiável.

**Validação pós-inclusão**: 
- Treinar Exp com dados estendidos
- Comparar OOV accuracy (geminadas) vs baseline Exp104d
- Esperado: redução de erro no subset de geminadas

---

### TF-005: Estratificação de Batches Durante Treinamento

**Objetivo**: Reduzir variância em loss curves mantendo distribuição fonológica balanceada em cada mini-batch.

**Setup atual**: Batches aleatórios simples (`batch_size=32`)

**Proposta**: Implementar stratified batching (`batch_size=96`)

**Mecânica**:
1. Agrupar corpus por estratos fonológicos (mesmo critério do split train/val/test)
2. Amostrar proporcionalmente em cada batch:
   - 6 exemplos de "monossilábicos"
   - 24 exemplos de "2–3 silábicas"
   - 48 exemplos de "4+ silábicas"
   - etc (proporções = dataset)

**Benefício esperado**:
- Redução de variância em curvas de loss: ~50% menos ruído
- Convergência mais estável
- Sem penalidade significativa em tempo de treino: +6% overhead de indexação

**Referência técnica**: [docs/article/ARTICLE.md § 9.1 Estratificação de batches](../article/ARTICLE.md#91-trabalhos-futuros)

---

### TF-006: Análise de Homógrafos Heterófonos (Morfossintaxe)

**Objetivo**: Resolver ambiguidade onde ortografia é idêntica mas pronúncia depende da categoria gramatical.

**Exemplos PT-BR**:
- *jogo* (substantivo /ˈʒɔgʊ/ vs verbo /ˈʒogʊ/) — diferença em altura/abertura de segunda vogal
- *gosto* (substantivo /ˈɡɔstʊ/ vs verbo /ɡɔˈstʊ/) — stress diferente
- *acordo* (substantivo /aˈkɔrdʊ/ vs verbo /ɑˈkɔrdʊ/) — slight vowel difference

**Natureza do limite**: Qualquer sistema G2P em **isolamento de palavra** (word-level, sem contexto) nunca resolverá isso. É um limite arquitetural irredutível, não da implementação.

**Solução**: Pipeline em série (não sequencial local):
```
Texto pleno → Analisador Morfossintático (POS-tagging) → G2P Condicionado → IPA
```

**Referência técnica**: [docs/article/ARTICLE.md § 9.1 Morfossintaxe](../article/ARTICLE.md#91-trabalhos-futuros)

---

## Referências Cruzadas e Documentação Existente

Para evitar duplicação, todos os documentos devem referenciar este arquivo quando mencionarem "próximos passos", "trabalhos futuros", "extensão multilíngue", "Transformer", "Tupi", etc:

- **README.md**: Referenciar seção TF-003 (espaço 7D) ao falar sobre LatPhon, ByT5, transfer learning
- **ARTICLE.md**: Já contém texto original em § 9.1; este arquivo é o resumo executivo canonizado
- **EXPERIMENTS.md**: Referenciar TF-001 (métricas) ao mencionar "análise PanPhon graduada"
- **src/inference_light.py**: Documentação de uso pode referenciar TF-004 (geminadas) se dataset for expandido

**Convenção**: Quando outro arquivo menciona um futuro trabalho:
1. Descrição breve inline se < 2 parágrafos
2. Referência cruzada para esta seção se > 2 parágrafos ou se for tópico central