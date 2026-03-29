ID: 041
Title: Leitor de dataset multilíngue com tags de idioma
Type: feature
Priority: High
Status: Open

Descrição:
Planejar leitor de dataset misto para treinar um supermodelo multilíngue, mantendo também opção de modelos por idioma de alta performance.

Escopo:
- Definir formato canônico de tag de idioma/variante no input (ex.: `<lang:pt-br-sp>palavra`).
- Definir contrato de parsing e validação dessas tags.
- Definir modo single-language vs multi-language no loader.

Critérios de aceite:
- Especificação de tag documentada e validada.
- Protótipo de loader para dataset misto.
- Compatibilidade preservada com dataset monolíngue existente.

Dependências:
- 035, 037, 039, 040

Dependências adicionais (2026-03-26):
- 043 (piloto PT-BR first)
- 044 (contrato de corpus)
- 046 (agregação PT-BR multi-arquivo)
- 047 (taxonomia lexical opcional)

Fase de prontidão para implementação:
1. Primeiro garantir estabilidade PT-BR (037) e contrato de corpus (044).
2. Depois integrar agregação PT-BR multi-arquivo (046) como caso de transição.
3. Só então habilitar parsing de tags multilíngues em modo misto.

Critério adicional de aceite:
- Modo single-language deve reproduzir baseline PT-BR sem regressão antes de ativar modo multi-language.
