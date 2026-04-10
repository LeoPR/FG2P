ID: 035
Title: [v2.x] Arquitetura Transformer como substituto do BiLSTM
Type: research
Priority: Medium
Status: Open

## Contexto

A v1.x do FG2P usa **deliberadamente** um BiLSTM encoder-decoder com atencao
de Bahdanau (arquitetura de 2014). A escolha foi justificada cientificamente:

> "A arquitetura deliberadamente escolhida para isolar a contribuicao da
> funcao de custo de novidades arquiteturais."

Isto permitiu atribuir os ganhos de PER/WER a DA Loss, nao a arquitetura.
Foi a decisao correta para validar o conceito em v1.x.

A v2.x deve **remover essa limitacao**, adotando arquitetura moderna:

## Motivacao

### Por que mudar agora
1. **Paper A (v1.x) ja validou** que DA Loss funciona — o conceito esta provado
2. **LatPhon (Chary 2025)** usa Transformer e e a referencia SOTA competitiva
3. **Transfer learning multilingue** funciona melhor com Transformers (atencao cross-lingual)
4. **Pre-training** so e viavel com Transformers (BERT-style masking)

### Por que NAO foi feito em v1.x
- Foco em isolar contribuicao da loss (ver acima)
- BiLSTM e mais estavel com dataset pequeno (95k palavras)
- Facilita reproducibility (menos hiperparametros)

## Direcoes exploratorias

### Opcao A: Transformer encoder-decoder classico (Vaswani 2017)
- Linha de base moderna
- Comparavel diretamente com LatPhon (7.5M, 4 layers, RoPE)
- Risco: pode nao convergir bem com dataset pequeno

### Opcao B: Transformer + pre-training (BERT-style)
- Pre-treinar em grafemas (masked grapheme modeling)
- Fine-tune em G2P supervised
- Aproveita dados nao-rotulados de corpora grandes

### Opcao C: Character-level Transformer (ByT5-style)
- Byte-level input, sem tokenizer
- Melhor generalizacao para OOV e chars raros (k, w, y)
- Cita: Xue et al. 2022 (ByT5)

### Opcao D: Hibrido — Conformer ou Lite-Transformer
- Combina atencao + convolucao
- Melhor para sequencias curtas (palavras de 5-15 chars)
- Usado em ASR moderno

## Desafios especificos

### Integracao com DA Loss
A DA Loss opera por token na saida. Transformer produz logits por token
igual ao BiLSTM — integracao deve ser direta. Verificar:
- Teacher forcing funciona igual?
- Atencao cruzada precisa de mudancas?
- Beam search na inferencia muda comportamento?

### Separadores silabicos e tokens estruturais
Exp104d depende de separadores (`.`) e stress marker (`ˈ`) como tokens
especiais. Transformer lida bem com isso, mas precisa verificar:
- Posicionamento aprendido vs fixo
- Interacao com override de distancia (ver ticket 034)

### Tamanho do modelo
- BiLSTM 17.2M (Exp104d) e eficiente — Transformer equivalente seria
  provavelmente menor em capacidade util
- Referencia: LatPhon 7.5M chega a PER 0.86% multilingual
- Target: PER <= 0.48% monolingual com menos parametros

## Criterios de aceite

- [ ] Pelo menos 2 arquiteturas Transformer implementadas
- [ ] Baseline direto (Vaswani 2017) com mesmas condicoes do Exp104d
- [ ] Comparacao PER/WER/throughput em PT-BR (reproduzindo v1.x)
- [ ] Integracao testada com DA Loss (atual e nova formula do ticket 034)
- [ ] Ablation: BiLSTM + DA vs Transformer + DA (isolando efeito arquitetura)
- [ ] Contribuicao publicavel no Paper C (v2.x)

## Dependencias

- Ticket 034 (formula de gradiente): pode interagir com arquitetura
- Ticket 026 (multilingue): pre-training so vale a pena multilingue

## Proximos passos

1. Implementar Transformer encoder-decoder baseline (Vaswani puro)
2. Treinar em PT-BR com DA Loss v1 (formula atual)
3. Comparar com Exp104d (mesmo split, mesmos hiperparametros)
4. Explorar variantes (ByT5, Conformer) se baseline for competitivo
5. Integrar com formula nova (ticket 034) quando madura
