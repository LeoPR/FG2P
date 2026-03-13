# 012 - Clareza narrativa e nomenclatura

Status: respondida

## Pergunta

Como manter clareza quando o texto mistura historia do projeto (fases/experimentos internos) e comparacoes com trabalhos externos?

## Principio geral

Documentacao deve preservar a ordem da historia do projeto e usar nomenclatura consistente por contexto.

1. Comparacao externa (entre trabalhos):
- usar o mesmo eixo de rotulo para todas as linhas da tabela (ex.: Nome + Ano)
- deixar detalhes de configuracao em nota/caption/coluna de contexto

2. Comparacao interna (entre variantes do proprio projeto):
- usar IDs internos no formato canonico `ExpN` ou `ExpN[a-z]`, com uma frase curta explicando a convencao

3. Fluxo narrativo:
- primeiro "quem compara com quem" (contexto externo)
- depois "qual configuracao interna foi usada" (rastreabilidade)

## Exemplo aplicado

No ajuste FG2P vs LatPhon:
- a tabela externa passou a usar o mesmo eixo para ambos (Nome + Ano)
- o identificador interno (Exp104b) ficou como referencia de configuracao, nao como rotulo principal da comparacao

Esse exemplo deve servir de padrao para outras secoes que misturem baseline externo com fases internas.

## Regra reutilizavel para futuras revisoes

Padrao recomendado para IDs internos:
- formato canonico: `ExpN` ou `ExpN[a-z]`
- regex pratica: `^Exp[0-9]+[a-z]?$`
- exemplo: `Exp0`, `Exp9`, `Exp104b`, `Exp104d`
- quando houver variacao de hiperparametro dentro da mesma familia, preferir texto contextual, por exemplo `Exp7 (lambda=0.20)`, em vez de embutir o valor no ID principal

Antes de publicar uma tabela, verificar:
- o eixo de nomeacao e o mesmo para todas as linhas?
- a ordem da explicacao ajuda um leitor novo a entender a historia sem conhecer os IDs internos?
- os detalhes tecnicos ficaram no lugar certo (contexto), sem poluir o rotulo principal?

## O que ainda precisa ser explicado na ordem feliz

Mesmo com a nomenclatura ajustada, ainda ha pontos que um leitor novo so entende por inferencia.

1. Configuracoes complementares antes dos numeros principais:
- explicar cedo que FG2P nao tem um unico "melhor modelo" universal
- Exp104b e referencia para PER/TTS
- Exp9 e referencia para WER/NLP

2. Por que o topo escolhe Exp104b como vitrine:
- deixar explicito por que o resumo inicial abre com PER 0,49% e WER 5,43%
- explicar que a abertura privilegia o modelo de referencia principal para comparacao externa, enquanto a secao de selecao de modelos mostra as alternativas por caso de uso

3. Qual metrica ancora a comparacao com literatura:
- dizer antes da tabela que a comparacao externa com LatPhon e feita principalmente em PER
- justificar isso pelo fato de WER nao estar disponivel na referencia externa

4. O que significa "same source family (ipa-dict)":
- esclarecer que significa mesma linhagem de recurso lexical, nao subconjunto identico nem split identico

5. Quando a dificuldade da tarefa muda:
- avisar cedo que alguns modelos predizem separadores silabicos e outros nao
- deixar claro que isso altera a dificuldade e a leitura de PER/WER

6. Onde entra o override estrutural:
- explicar cedo que o melhor PER (Exp104b) depende da correcao de distancias para tokens estruturais
- isso evita que o leitor atribua todo o ganho apenas a DA Loss de forma generica

7. Ordem ideal para leitor novo:
- o que o projeto faz
- quais sao as duas referencias praticas (Exp104b e Exp9)
- qual metrica compara com literatura
- por que a comparacao e limitada ao setup
- so depois: detalhes de loss, split, arquitetura e ablacoes
