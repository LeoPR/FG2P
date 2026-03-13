# 011 - Questionario de Fechamento por Etapas

Status: em aberto

Objetivo:
- transformar pendencias abertas em decisoes objetivas;
- evitar retrabalho e overclaim;
- fechar as notas 003, 004, 005, 006, 008, 009 e 010 com trilha audivel.

## Etapa 1 - Claims e Escopo (prioridade alta)

Perguntas para decisao:

1. Sobre a frase "FG2P supera LatPhon em PT-BR", qual redacao final voce prefere?
   - A) "supera em PER no setup reportado" (condicional)
   - B) "apresenta PER menor com IC nao sobreposto no setup reportado" (mais neutra)
   - C) outra redacao (especificar)

2. A frase "metodo > arquitetura" deve:
   - A) ser removida do README/ARTICLE
   - B) ser reescrita como hipotese condicional
   - C) ser mantida com caveat forte

3. Em comparacoes com literatura, qual nivel de prudencia voce quer como padrao?
   - A) conservador (apenas o que e estritamente comparavel)
   - B) balanceado (comparavel + contexto)
   - C) agressivo (com ressalvas)

Criterio de fechamento da etapa:
- redacao definitiva aprovada para claims parciais/frageis da nota 010.

## Etapa 2 - DA Loss (prioridade alta)

Perguntas para decisao:

1. Qual tese principal da DA Loss no texto final?
   - A) melhora de qualidade do erro (Classe D -> B)
   - B) melhora de PER classico
   - C) ambas, com separacao clara de quando cada uma aparece

2. Voce quer destacar explicitamente que o ganho de Exp104b inclui override estrutural?
   - A) sim, no README e no ARTICLE
   - B) sim, so no ARTICLE
   - C) nao

3. Sobre validacao perceptual (MOS/ABX):
   - A) registrar como trabalho futuro obrigatorio
   - B) iniciar plano minimo de experimento agora
   - C) deixar apenas como nota de limitacao

Criterio de fechamento da etapa:
- nota 008 com narrativa final aprovada e sem causalidade indevida.

## Etapa 3 - Comparacoes empiricas justas (prioridade alta)

Perguntas para decisao:

1. Regra obrigatoria para qualquer claim numerico no README/ARTICLE:
   - A) metrica + condicao + calculo + limite (obrigatorio)
   - B) metrica + condicao (mais leve)

2. Para throughput/hardware, padrao de frase deve incluir razao explicita?
   - A) sim (ex.: 31.4/7.8 = 4.0x)
   - B) nao, apenas valores absolutos

3. Quando nao houver metrica equivalente na referencia externa (ex.: WER):
   - A) usar `n/d` e evitar inferencia cruzada
   - B) estimar por aproximacao

Criterio de fechamento da etapa:
- nota 009 aplicada em README/ARTICLE e checklist marcado como concluido.

## Etapa 4 - Tamanho minimo e estabilidade (prioridade media)

Perguntas para decisao:

1. Conclusao desejada para tamanho minimo sera:
   - A) teorica
   - B) empirica
   - C) hibrida

2. Fator de seguranca para `N_pratico = N_minimo * fator`:
   - A) 2x
   - B) 4x
   - C) 8x
   - D) por faixa (ex.: 2x-4x)

3. Curva de estabilidade (% treino):
   - A) apenas pontos existentes (50/60/95)
   - B) incluir novos pontos intermediarios (ex.: 55, 65, 75)

Criterio de fechamento da etapa:
- notas 003, 005 e 006 com criterio final fechado e narrativa unica.

## Etapa 5 - Validade externa (prioridade media)

Perguntas para decisao:

1. Qual formula final para validade externa no texto principal?
   - A) alta para comparacao classica, moderada para generalizacao ampla
   - B) alta para ambas (com caveats)
   - C) outra (especificar)

2. Teste OOV/neologismos no texto deve ser descrito como:
   - A) evidencia promissora
   - B) evidencia robusta
   - C) evidencia preliminar

Criterio de fechamento da etapa:
- nota 004 com redacao final integrada ao ARTICLE.

## Resultado esperado do questionario

Ao responder as etapas acima, cada resposta vira uma decisao editorial ou experimental concreta. Isso permite fechar o backlog sem loops e com rastreabilidade.
