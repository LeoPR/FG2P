# Guia BCP 47 — FG2P v2.0+

**Ultima atualizacao**: 2026-04-11
**Proposito**: Registrar por que o FG2P v2.0 adota BCP 47 (IETF RFC 5646)
como padrao de tags de idioma, com foco nos casos relevantes ao projeto:
Portugues Brasileiro, linguas indigenas do Brasil (Tupinamba, Nheengatu,
Guarani), e variantes sub-regionais (sotaques como Sao Paulo, Rio de Janeiro).

Este documento **nao substitui** a especificacao oficial BCP 47 — serve
apenas como referencia operacional rapida para o projeto, para evitar
pesquisar os mesmos pontos a cada sessao de trabalho.

---

## Por que BCP 47

### Decisao (2026-04-11)
O FG2P v2.0 adota **BCP 47** como padrao canonico de tags de idioma em
todo o pipeline: arquivos de regras, datasets, manifest, inputs do modelo,
e documentacao.

### Justificativa

1. **Padrao guarda-chuva**: BCP 47 nao e uma alternativa ao ISO 639-3 —
   e a especificacao IETF que **usa** ISO 639-1, 639-3, e ISO 3166-1 como
   suas bases. Quem adota BCP 47 automaticamente usa ISO.

2. **Amplo**: usado por ICU (Unicode), CLDR, HTML `lang=`, navegadores,
   sistemas operacionais, bibliotecas de internacionalizacao (i18n), todo
   o ecossistema de localizacao de software relevante.

3. **Moderno**: RFC 5646 de setembro/2009, ainda em uso corrente sem
   substituto.

4. **Extensivel**: suporta variantes registradas (com IANA) e extensoes
   privadas (`-x-`) para casos nao contemplados no registry oficial.

5. **Interoperabilidade**: sistemas modernos de TTS/ASR/NMT usam BCP 47
   como entrada padrao. CharsiuG2P, por exemplo, usa a sintaxe
   `<eng-us>: word` compativel com BCP 47.

### Fonte primaria

- **RFC 5646 — Tags for Identifying Languages**:
  https://www.rfc-editor.org/rfc/rfc5646.html
  (Tambem conhecido como BCP 47 — Best Current Practice 47)

- **IANA Language Subtag Registry**:
  https://www.iana.org/assignments/language-subtag-registry
  (Lista oficial de todos os subtags validos)

- **CLDR (Common Locale Data Repository)** — implementacao de referencia:
  https://cldr.unicode.org/

---

## Regras de escolha (pratico)

Siga esta ordem para construir uma tag BCP 47:

### 1. Codigo primario de idioma

| Situacao | Regra |
|---|---|
| Idioma com codigo ISO 639-1 (2 letras) | **Usa o de 2 letras**. Ex: `pt`, `en`, `es`, `gn` |
| Idioma so com ISO 639-3 (3 letras) | Usa o de 3 letras. Ex: `tpn`, `yrl` |
| Nunca mistura | Mesma tag nao pode ter parte 2-letra e parte 3-letra |

### 2. Regiao (pais) — opcional

Adiciona codigo ISO 3166-1 alpha-2 com hifen, **em maiusculas**:
- `pt-BR` (Portugues do Brasil)
- `pt-PT` (Portugues de Portugal)
- `en-US` (Ingles americano)
- `en-GB` (Ingles britanico)

### 3. Variante sub-regional — private use

BCP 47 **nao tem codigo oficial** para sub-dialetos como "Portugues de
Sao Paulo". A solucao padrao e usar **private use** com prefixo `-x-`:
- `pt-BR-x-sp` (variante Sao Paulo)
- `pt-BR-x-rj` (variante Rio de Janeiro)
- `pt-BR-x-ne` (variante Nordeste)

Os subtags apos `-x-` sao **livres** (desde que tenham de 1 a 8 caracteres
alfanumericos cada). Nao precisa registrar com IANA.

---

## Idiomas relevantes ao FG2P

### Portugues e variantes

| Tag | Descricao | Status |
|---|---|---|
| `pt` | Portugues (macrolanguage, raro usar sem regiao) | ISO 639-1 |
| `pt-BR` | Portugues Brasileiro | **Canonical v1.x e v2.x** |
| `pt-PT` | Portugues Europeu | Futuro |
| `pt-BR-x-sp` | Portugues Brasileiro, variante Sao Paulo | Private use |
| `pt-BR-x-rj` | Portugues Brasileiro, variante Rio de Janeiro | Private use |
| `pt-BR-x-ne` | Portugues Brasileiro, variante Nordeste | Private use |
| `pt-BR-x-sul` | Portugues Brasileiro, variante Sul | Private use |

**Nota**: a granularidade das variantes (SP, RJ, etc.) e decisao do
projeto, nao do padrao. O `-x-` garante que nao ha conflito com codigos
futuros do IANA.

### Linguas indigenas do Brasil (familia Tupi e outras)

| Tag | Descricao | Status |
|---|---|---|
| `tpn` | **Tupinamba** (Old Tupi) | ISO 639-3 ativo. Extinto como L1, usado em revitalizacao |
| `tpw` | Tupi | **DEPRECATED em 2023**. Usar `tpn` |
| `yrl` | **Nheengatu** (Lingua Geral Amazonica) | ISO 639-3 ativo. Vivo |
| `gn` | Guarani (macrolingua) | ISO 639-1 |
| `gug` | Guarani Paraguaio (Jopara) | ISO 639-3, mais especifico que `gn` |
| `gun` | Guarani Mbya | ISO 639-3 |
| `gub` | Guajajara | ISO 639-3 (povo Tenetehara) |
| `kgp` | Kaingang | ISO 639-3 |
| `xav` | Xavante | ISO 639-3 |
| `ter` | Terena | ISO 639-3 |

**Nota cientifica**: muitas dessas linguas tem pouco material IPA
disponivel em dicionarios abertos. O suporte no FG2P dependera da
existencia de corpora adequados, possivelmente via crowdsourcing ou
parceria com linguistas especializados.

### Linguas para comparacao/transfer multilingue (Paper B futuro)

| Tag | Descricao | Fonte |
|---|---|---|
| `en-US` | Ingles Americano | ipa-dict `en_US.txt` |
| `en-GB` | Ingles Britanico | ipa-dict `en_UK.txt` |
| `es-ES` | Espanhol Castelhano | ipa-dict `es_ES.txt` |
| `es-MX` | Espanhol Mexicano | ipa-dict `es_MX.txt` |
| `fr-FR` | Frances de Franca | ipa-dict `fr_FR.txt` |
| `fr-CA` | Frances de Quebec | ipa-dict `fr_QC.txt` |
| `de` | Alemao | ipa-dict `de.txt` |
| `it` | Italiano | (nao presente no ipa-dict atualmente) |

---

## Como o FG2P usa BCP 47

### 1. No `.rules.tsv` do pipeline

Futuro (Fase 2.5) — acao `tag` declara a tag BCP 47 associada ao grupo:
```
# pt-br.rules.tsv
pt-br    tag       pt-BR                                         # ← BCP 47
pt-br    src       ../sources/ipa-dict/data/pt_BR.txt
pt-br    dst       ../output/pt-br.tsv
pt-br    regex     /
pt-br    regex     g                                 ɡ
```

Note que o **nome do grupo** no `.rules.tsv` pode ser qualquer string
(ex: `pt-br` minusculo, sem hifen de pais), mas a tag **BCP 47 canonica**
vai na acao `tag` (com maiusculas e hifen corretos: `pt-BR`).

### 2. No input do modelo (sintaxe CharsiuG2P-compativel)

```
<pt-BR>: palavra
<pt-BR-x-sp>: palavra
<tpn>: palavra
```

Componentes:
- `<` e `>` delimitam a tag
- `:` apos o `>`
- **espaco obrigatorio** depois do `:`
- palavra-alvo em seguida

### 3. No manifest (Fase 2.5)

`dicts/manifest.yaml` mapeia path de arquivo → tag BCP 47:
```yaml
entries:
  - path: pt/br/pt-br.tsv
    tag: pt-BR
    description: "Portugues Brasileiro canonical"
    source: "ipa-dict via dicts-workbench"
```

### 4. Filesystem vs tag (desacoplamento)

O FG2P adota **desacoplamento** entre nome de pasta e tag BCP 47:
- **Pastas**: nomes minusculos simples (ex: `dicts/pt-br/sp/`) — Windows/Linux compat
- **Tags**: sempre com maiusculas corretas (`pt-BR-x-sp`) — vem do manifest

Isso evita problemas de case-sensitivity no Windows e permite renomear
pastas sem quebrar o sistema de tags.

---

## Perguntas frequentes

### Por que nao usar `pt-br` minusculo como tag oficial?

Porque BCP 47 especifica que:
- Subtag primario (idioma): minusculas
- Subtag de regiao: **MAIUSCULAS**
- Subtag de variante: minusculas

Entao `pt-BR` e **canonico**, `pt-br` e tecnicamente incorreto. Sistemas
tolerantes aceitam ambos, mas o FG2P usa a forma canonica para
interoperabilidade.

### Qual a diferenca entre `gn` e `gug`?

- `gn` (Guarani) e uma **macrolanguage** — engloba varias variantes
- `gug` (Guarani Paraguaio/Jopara) e uma variante especifica, ISO 639-3

Para o FG2P, se o dataset e especifico do Guarani Paraguaio, use `gug`.
Se for generico ou ainda nao foi decidido, use `gn`.

### E se eu quiser uma variante de variante, tipo "SP interior"?

Extensoes privadas permitem multiplos subtags:
```
pt-BR-x-sp-interior
pt-BR-x-sp-capital
```

Cada subtag apos `-x-` deve ter 1-8 caracteres alfanumericos.

### Como distinguir pronuncia fonemica vs fonetica?

BCP 47 tem a variante registrada `fonipa` para transcricao fonetica em
IPA. Exemplo:
```
pt-BR-fonipa    # Portugues Brasileiro em IPA (transcricao fonetica)
```

Isso e IANA-registrado, nao private use. Pode ser util quando o FG2P
precisar distinguir "representacao ortografica vs representacao IPA" na
mesma tag. Nao e usado atualmente, mas bom saber que existe.

---

## Referencias

- RFC 5646 (BCP 47): https://www.rfc-editor.org/rfc/rfc5646.html
- IANA Registry: https://www.iana.org/assignments/language-subtag-registry
- ISO 639-3: https://iso639-3.sil.org/
- CLDR: https://cldr.unicode.org/
- CharsiuG2P (exemplo de uso de tags compativeis): https://github.com/lingjzhu/CharsiuG2P
- ipa-dict (fonte de dados): https://github.com/open-dict-data/ipa-dict
