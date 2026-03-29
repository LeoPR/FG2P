ID: 052
Title: Validação perceptual — MOS/ABX para qualidade de erros (Class B vs D)
Type: research
Priority: Medium
Status: Open

Descrição:
A hipótese central da DA Loss é que erros Classe B (substituições de 1 feature articulatória, ex: e↔ɛ) são perceptualmente menos salientes que erros Classe D (cross-class, ex: vogal↔stop). Esta hipótese, embora intuitivamente motivada e fonologicamente fundamentada, nunca foi validada perceptualmente no contexto do FG2P. O paper ICASSP (§6.2) já flag isso explicitamente como "future work".

Objetivo:
- Confirmar empiricamente que erros Classe B produzem síntese de menor degradação perceptual que Classe D, usando avaliação humana.
- Fortalecer o argumento central do paper: reduzir Classe D não é apenas uma métrica — tem impacto real em aplicações TTS.

Métodos possíveis:

**MOS (Mean Opinion Score) — listening test**:
- Sintetizar pares de palavras: (transcrição DA Loss) vs. (transcrição CE baseline)
- Avaliar preferência por juízes humanos (AMT ou voluntários linguistas)
- Custo: requer sintetizador TTS externo (Coqui TTS, ESPnet, ou similar para PT-BR)

**ABX test**:
- A = pronúncia correta (dicionário), B = pronúncia com erro Classe B, X = pronúncia com erro Classe D
- Pergunta: "X está mais próximo de A ou B?"
- Se X=B for preferido sobre X=D, valida a hierarquia articulatória

**Proxy rápido sem TTS**:
- Comparar distribuição de erros por posição prosódica: erros Classe D em sílabas tônicas causam mais degradação perceptual que em átonas. Análise quantitativa da distribuição posicional como proxy fonológico.

Escopo mínimo viável:
1. Análise de distribuição de erros Classe D por posição silábica (tônica vs. átona) — proxy fonológico sem estudo de usuário. Implementável com dados existentes.
2. Relatório documentando distribuição observada como suporte à hipótese perceptual.

Escopo completo (paper futuro):
1. Integração com sintetizador TTS PT-BR
2. Estudo perceptual com N≥20 ouvintes nativos
3. Análise estatística com IC95

Critérios de aceite (escopo mínimo):
- Tabela de erros Classe D por posição prosódica documentada.
- Discussão atualizada em ARTICLE.md §6.4.

Critérios de aceite (escopo completo):
- Protocolo MOS/ABX desenhado e executado.
- Resultados integrados ao paper como evidência de aplicabilidade.

Dependências:
- src/analyze_errors.py (extração de classe por posição)
- Sintetizador TTS PT-BR (para escopo completo — componente externo)
