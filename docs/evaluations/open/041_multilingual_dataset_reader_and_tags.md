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
