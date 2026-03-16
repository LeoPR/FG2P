# 019 - Real-World Use Case do G2P: escopo, taxonomia de falhas e metricas

Status: respondida

Objetivo:
- documentar perguntas para revisar o trecho de Real-World Use Case no README;
- garantir que o exemplo fique estritamente alinhado ao escopo de G2P;
- estruturar uma ponte clara entre erro de fala, erro fonetico percebido e erro de escrita/digitacao;
- preparar base conceitual para possivel metrica nova, sem overclaim.

## Contexto do problema

O exemplo atual ("cinto muito" vs "sinto muito") e comunicativo, mas pode parecer fora do escopo estrito de G2P se for lido como tarefa de semantica, ASR, correcao gramatical ou NLP geral.

A revisao desejada e:
- manter um exemplo claro e limpo de realidade de uso;
- explicitar o que o G2P explica diretamente e o que depende de pipeline maior;
- evitar ambiguidade entre falha de fala, falha de escrita e falha de digitacao.

## Bloco A - Escopo cientifico do caso de uso

Perguntas para responder:

1. Qual formulacao de caso de uso fica mais fiel ao escopo de G2P?
- A) Normalizacao fonologica para TTS
- B) Suporte a recuperacao fonetica em busca/IR
- C) Camada fonetica auxiliar para deteccao de confusoes ortograficas plausiveis
- D) Outra formulacao (especificar)

2. Qual frase de fronteira deve aparecer no README para evitar overclaim?
- A) "G2P nao resolve semantica por si so"
- B) "G2P e componente de um pipeline maior"
- C) Ambas

3. O exemplo principal deve ser:
- A) somente fonologico (sem semantica social)
- B) fonologico com nota curta de contexto social
- C) duas versoes lado a lado (baseline + ampliada)

Criterio de fechamento do bloco:
- redacao curta aprovada para "o que G2P faz" e "o que G2P nao faz".

## Bloco B - Taxonomia de falhas (fala, escrita, digitacao)

Perguntas para responder:

1. Qual taxonomia minima deve entrar no texto?
- A) 3 niveis: fala, fonetico-percebido, escrita/digitacao
- B) 4 niveis: fala, percepcao fonetica, ortografia, digitacao
- C) Outra estrutura (especificar)

2. Para cada nivel, qual unidade observavel minima sera usada?
- fala: telefone/silaba/acento?
- percepcao: confusao de classe A/B/C/D?
- ortografia: substituicao de grafema?
- digitacao: distancia de teclado/edicao?

3. Quais relacoes causais podem ser afirmadas com seguranca?
- A) apenas associacao
- B) associacao com hipotese mecanistica
- C) causalidade forte (somente com evidencias adicionais)

Criterio de fechamento do bloco:
- tabela simples "tipo de falha -> observavel -> limite de inferencia".

## Bloco C - Exemplo do README (clareza e aderencia)

Perguntas para responder:

1. O exemplo "cinto muito" deve:
- A) ser removido
- B) ser mantido com reformulacao estrita de escopo
- C) ser substituido por exemplo menos ambiguo

2. Se for substituido, qual tipo de exemplo priorizar?
- A) confusao fonetica de coda/final (mais G2P puro)
- B) acento tonico com impacto em inteligibilidade
- C) separador silabico e fronteiras prosodicas

3. Qual formato deixa mais claro o papel do G2P?
- A) entrada ortografica -> saida IPA -> interpretacao
- B) par minimo com duas saidas IPA
- C) mini-caso com erro esperado e erro evitado

Criterio de fechamento do bloco:
- um exemplo aprovado, curto, reproduzivel e estritamente G2P-oriented.

## Bloco D - Fair comparison para familias de saida

Perguntas para responder:

1. No trecho de Real-World Use Case, devemos reforcar explicitamente as familias?
- A) sem separador (fonemas + tonica)
- B) com separador (fonemas + tonica + fronteira)
- C) ambas com frase de comparacao justa

2. Qual regra de leitura deve aparecer junto do exemplo?
- A) "nao comparar WER bruto entre familias"
- B) "comparar dentro da mesma estrutura de saida"
- C) ambas

3. Como citar Exp104c sem invalidar o resultado?
- A) WER valido no regime sem separador
- B) nao usar para claim de superioridade sobre modelos com separador
- C) A + B no mesmo paragrafo

Criterio de fechamento do bloco:
- paragrafo curto de fairness aprovado para README e replicavel no ARTICLE.

## Bloco E - Teoria e ancoragem bibliografica

Perguntas para responder:

1. Quais familias teoricas devem sustentar o texto?
- A) fonologia/fonotatica e percepcao da fala
- B) psicolinguistica do erro ortografico
- C) dinamica sociolinguistica da inteligibilidade
- D) combinacao A+B+C com escopo delimitado

2. Qual profundidade bibliografica cabe no README?
- A) apenas mencao conceitual
- B) 1-2 referencias chave
- C) detalhe maior apenas no ARTICLE

3. Onde registrar o aprofundamento teorico?
- A) no ARTICLE
- B) em nota de avaliacao respondida
- C) em ambos

Criterio de fechamento do bloco:
- lista curta de referencias-alvo e frase de delimitacao metodologica.

## Bloco F - Possivel metrica nova (exploratoria)

Perguntas para responder:

1. A proposta de metrica nova entra como:
- A) ideia futura (sem numero)
- B) prototipo descritivo
- C) metrica formal com validacao

2. Qual eixo a metrica deve capturar primeiro?
- A) severidade fonetica ponderada por classe A/B/C/D
- B) risco de confusao ortografica a partir da saida fonetica
- C) combinacao (custo fonetico + custo ortografico)

3. Qual requisito minimo para nao virar overclaim?
- A) protocolo de anotacao claro
- B) conjunto de teste dedicado
- C) avaliacao de confiabilidade entre anotadores
- D) A+B (minimo)

Criterio de fechamento do bloco:
- decisao explicita: "metrica nova agora" vs "trabalho futuro".

## Bloco G - Entregas editoriais (README -> ARTICLE)

Perguntas para responder:

1. Qual nivel de mudanca no README nesta iteracao?
- A) microedicao de 1-2 paragrafos
- B) reescrita da subsecao inteira
- C) manter texto e adicionar box de escopo

2. O ARTICLE deve refletir essa revisao em qual secao primeiro?
- A) Discussao (limitacoes e escopo)
- B) Aplicacoes praticas
- C) ambos

3. Qual criterio de consistencia entre README e ARTICLE?
- A) mesma tese central, niveis de detalhe diferentes
- B) README mais pratico, ARTICLE mais formal
- C) A + B

Criterio de fechamento do bloco:
- checklist de sincronizacao README/ARTICLE aprovado antes da proxima versao.

## Resultado esperado desta nota

Ao responder estas perguntas, o projeto deve chegar em:
1. um Real-World Use Case claro, limpo e estritamente coerente com G2P;
2. uma taxonomia pratica de falhas (fala/percepcao/escrita/digitacao) com limites de inferencia;
3. um texto com fair comparison consistente com as familias de saida e com a leitura de Exp104c/Exp104d;
4. um plano realista para metrica nova (ou decisao de deixar como trabalho futuro).

## Respostas consolidadas (2026-03-15)

1. Escopo cientifico adotado:
	- formulacao principal: G2P como camada fonetica auxiliar para recuperar confusoes ortograficas plausiveis em pipelines maiores;
	- fronteira explicita: G2P nao resolve semantica por si so.

2. Taxonomia minima adotada:
	- quatro niveis: fala, percepcao fonetica, ortografia, digitacao;
	- regra de inferencia: no README, afirmar associacao e mecanismo plausivel, sem claim de causalidade forte.

3. Exemplo editorial:
	- manter exemplo curto e didatico, mas com redacao estritamente G2P-oriented;
	- formato aprovado: entrada ortografica -> saida IPA -> interpretacao do papel fonetico no pipeline.

4. Fair comparison (familias de saida):
	- reforcar regra de comparacao dentro da mesma estrutura de saida;
	- manter Exp104c como resultado valido no regime sem separador, sem usar como superioridade global sobre modelos com separador.

5. Metrica nova:
	- decisao: manter como trabalho futuro (sem introduzir numero novo nesta versao);
	- prerequisitos minimos para versao futura: protocolo de anotacao + conjunto dedicado.

## Decisao final da avaliacao

- Avaliacao encerrada como respondida.
- O caso de uso real foi delimitado para o escopo de G2P e desacoplado de claims semanticos amplos.
- A narrativa final fica consistente com os limites metodologicos do projeto e com as regras de comparacao justa ja adotadas.
- Revisao editorial aprofundada aplicada em README e ARTICLE: foco em "o que foi usado e como", com teoria detalhada remetida para `docs/article/REFERENCES.bib` e documentos tecnicos complementares.
