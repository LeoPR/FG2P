ID: 048
Title: Checklist de submissão ICASSP 2026 → migrado para ICASSP 2027
Type: publication
Priority: High
Status: Open

⚠️ ATENÇÃO: O deadline do ICASSP 2026 foi 17 de setembro de 2025 — já passou.
Portal real: https://2026.ieeeicassp.org/ (conferência: Barcelona, 4–8 maio 2026)
Este ticket agora cobre a preparação para ICASSP 2027 (Toronto, 16–21 maio 2027).

Descrição:
O paper está compilado (main.pdf em docs/article/icassp/). O ICASSP 2026 já não é possível. Foco agora: adaptar e submeter ao ICASSP 2027, cujo deadline deve abrir em ~setembro 2026.

⚠️ PROBLEMA DE FORMATO IDENTIFICADO:
O ICASSP 2026 usa formato 4 páginas de conteúdo + 1 página opcional de referências (5p total).
O ICASSP_DRAFT.md atual foi estruturado como 5 páginas de conteúdo + 1 de referências (6p total).
O paper precisa ser reduzido em ~1 página de conteúdo antes da submissão ao ICASSP 2027.

Objetivo:
- Garantir conformidade técnica e editorial antes de submeter ao portal oficial.
- Verificar anonimização completa.
- Confirmar portal de submissão real e deadline.

Itens do checklist:

**Compilação e formato**:
- [ ] Verificar que main.pdf compila sem erros/warnings (latexmk -pdf main em docs/article/icassp/)
- [ ] Confirmar que main.pdf não excede 6 páginas (5 conteúdo + 1 referências)
- [ ] Verificar que todas as referências ([1]–[11]) resolvem no PDF (sem ??)
- [ ] Verificar que todos os símbolos IPA renderizam corretamente (sem □ ou ?)
- [ ] Confirmar que figuras/tabelas não ultrapassam margens IEEEtran

**Anonimização (double-blind)**:
- [ ] Nenhuma referência ao nome do sistema interno (FG2P removido)
- [ ] Nenhum caminho de arquivo interno (dicts/pt-br.tsv removido)
- [ ] Nenhuma referência a .md files internos do projeto
- [ ] Seção "Revision Notes" ausente na versão final
- [ ] Authors/Affiliation como [ANONYMOUS]

**Adaptação para ICASSP 2027**:
- [ ] Reduzir paper de 5p conteúdo para 4p conteúdo (cortar ~1 coluna IEEEtran)
  - Candidatos a corte: parte da §3 (Architecture), parte da tabela λ sweep, §6.1 (error patterns menos crítico)
- [ ] Atualizar para template ICASSP 2027 quando disponível (NÃO usar template 2026)
- [ ] Portal ICASSP 2027: https://2027.ieeeicassp.org/ (deadline esperado ~setembro 2026)
- [ ] Confirmar categorias/tracks: "Speech and Language Processing" ou "Machine Learning for Audio"
- [ ] Preparar metadados: título, keywords (G2P, Brazilian Portuguese, Distance-Aware Loss, PanPhon), abstract

**Materiais complementares**:
- [ ] Decidir se submete código como suplemento (anonimizado)
- [ ] Se sim: preparar repositório anônimo (GitHub privado ou anonymous.4open.science)
- [ ] Verificar política de arXiv do ICASSP 2026 (pré-print permitido ou embargado?)

Critérios de aceite:
- PDF submetido no portal com receipt de confirmação.
- Todos os itens do checklist verificados.

Dependências:
- docs/article/icassp/main.tex (compilado — main.pdf já existe)
- docs/article/icassp/ICASSP_DRAFT.md (fonte de verdade do conteúdo)
- docs/article/icassp/icassp_refs.bib (11 referências)
