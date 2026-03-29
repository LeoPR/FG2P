ID: 045
Title: Canonização fonética para treino e estabilidade da DA Loss
Type: feature
Priority: High
Status: Open

Descrição:
Definir camada explícita de canonização fonética para treino, garantindo que equivalências representacionais (Unicode/forma visual) não gerem classes fonéticas duplicadas quando o alvo fonético for o mesmo.

Objetivo:
- Preservar o princípio segmental da DA Loss (classe por segmento fonético).
- Evitar acoplamento indevido entre forma visual do IPA e classe de treino.

Escopo:
- Definir política de canonização de segmentos para PT-BR.
- Documentar regras de alias representacional quando foneticamente equivalentes.
- Garantir compatibilidade com o pipeline atual (`g2p.py`, `losses.py`, `phonetic_features.py`).
- Definir como símbolos estruturais continuam tratados na distância.

Critérios de aceite:
- Regras de canonização documentadas com justificativa linguística.
- Pipeline de treino mantém comportamento funcional no baseline PT-BR.
- Sem regressão na inicialização/cálculo da DA Loss para vocabulário PT-BR atual.

Dependências:
- 037, 038, 043, 044
