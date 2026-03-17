ID: 026
Title: Multilíngue & Tupi / Dialetos
Type: feature
Priority: High
Status: Open

Descrição:
Planejar suporte multilíngue e variações dialetais. Definir tags/IDs para variações (ex.: `<sp-pt-br>`, `<tupi-variant-A>`) e um esquema canônico para armazenar perfis de normalização e dicionários sobrepostos.

Exemplos de requisitos:
- Cada idioma/dialeto/variante recebe uma tag canônica e um manifesto com: fonte, normalização, preferências de tokenização.
- Mecanismo de sobreposição em camadas para lexicons (canonical + additions + dialect-overrides).

Critérios de aceite:
- Documento com proposta de esquema de tags e exemplos (PT-BR regionais, Tupi variants)
- Exemplo de um dicionário de sobreposição com formato e regras de merge

Próximos passos:
1. Propor esquema de tags e convencões de arquivo
2. Listar idiomas/variações prioritárias
3. Criar ticket para cada idioma/dialeto prioritário (subtickets)
