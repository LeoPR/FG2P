ID: 054
Title: Paper de extensão multilíngue da DA Loss
Type: research
Priority: Low
Status: Open

Descrição:
A DA Loss foi desenvolvida para PT-BR, mas sua formulação é independente do idioma: qualquer fonema com cobertura PanPhon pode ser usado. Este ticket rastreia a pesquisa necessária para publicar a generalização do método para outros idiomas, separado da extensão do sistema G2P (tickets 026/039/040 cobrem o sistema; este ticket cobre o claim metodológico para publicação).

Hipótese:
DA Loss é uma contribuição metodológica transferível: seu benefício (redistribuição de erros Classe D→B) deve aparecer em qualquer idioma onde (1) existe corpus G2P com transcrições IPA e (2) os fonemas têm cobertura PanPhon. O paper de extensão validaria isso empiricamente.

Pré-requisitos técnicos:
- Sistema G2P multilíngue funcional (tickets 041, 043)
- Pelo menos 2 idiomas adicionais com resultados reproduzíveis (sugestões: inglês, espanhol, italiano — todos com boa cobertura PanPhon e dicionários disponíveis)
- Análise de distribuição de erros por classe (A–D) para cada idioma com/sem DA Loss

Potencial de contribuição científica:
- Validação cross-lingual: DA Loss é método robusto, não overfitting PT-BR?
- Análise comparativa: quais propriedades fonológicas por idioma afetam o ganho de DA Loss?
- Questão interessante: idiomas com menos ambiguidade grafema-fonema (ex: espanhol, italiano) teriam menos ganho de DA Loss que PT-BR/inglês?

Venues alvo para este paper:
- Interspeech 2027 (se extensão multilíngue pronta em 2026)
- ACL Findings (se foco em NLP multilíngue)
- CSL journal (seção de multilingual G2P)

Critérios de aceite:
- Experimentos DA Loss em ≥2 idiomas além do PT-BR com resultados documentados.
- Análise comparativa de distribuição de erros por classe entre idiomas.
- Paper draft com claim de generalização validado empiricamente.

Dependências:
- 026, 039, 040, 041, 043 (sistema multilíngue)
- 052 (perceptual validation como evidência adicional)
- 053 (journal version pode absorver esta extensão)
