# Avaliacoes do Projeto

Compendio simples de perguntas e status.

Regra da pasta:
- `open/` contem TODOs (um ticket por arquivo, com todos os detalhes).
- `answered/` contem perguntas respondidas (arquivo completo com contexto e evidencia).
- Este `README.md` mantem apenas o indice e links.

## Indice de Perguntas

| ID | Tema | Status | Arquivo |
|----|------|--------|---------|
| 001 | Robustez da pesquisa: pontos fortes e fracos | Respondida | [answered/001-robustez-da-pesquisa.md](answered/001-robustez-da-pesquisa.md) |
| 002 | Intervalo de confianca do WER | Respondida | [answered/002-ci-do-wer.md](answered/002-ci-do-wer.md) |
| 003 | Tamanho minimo de amostra para teste e treino | Respondida | [answered/003-tamanho-minimo-de-amostra.md](answered/003-tamanho-minimo-de-amostra.md) |
| 004 | Validade externa, comparacao classica e generalizacao | Respondida | [answered/004-validade-externa-e-generalizacao.md](answered/004-validade-externa-e-generalizacao.md) |
| 005 | Estratificacao incompleta e fatores ocultos | Respondida | [answered/005-estratificacao-incompleta-e-fatores-ocultos.md](answered/005-estratificacao-incompleta-e-fatores-ocultos.md) |
| 006 | Curva de estabilidade com menos dados | Respondida | [answered/006-curva-de-estabilidade-com-menos-dados.md](answered/006-curva-de-estabilidade-com-menos-dados.md) |
| 007 | Por que LatPhon nao reporta WER | Respondida | [answered/007-latphon-sem-wer.md](answered/007-latphon-sem-wer.md) |
| 008 | DA Loss: forcas e limites | Respondida | [answered/008-da-loss-forcas-limites.md](answered/008-da-loss-forcas-limites.md) |
| 009 | Regras para comparacoes empiricas justas | Respondida | [answered/009-comparacoes-empiricas-justas.md](answered/009-comparacoes-empiricas-justas.md) |
| 010 | Matriz consolidada de reivindicacoes | Respondida | [answered/010-matriz-reivindicacoes-consolidada.md](answered/010-matriz-reivindicacoes-consolidada.md) |
| 011 | Questionario de fechamento por etapas | Respondida | [answered/011-questionario-fechamento-por-etapas.md](answered/011-questionario-fechamento-por-etapas.md) |
| 012 | Clareza narrativa e nomenclatura | Respondida | [answered/012-clareza-nomenclatura-readme.md](answered/012-clareza-nomenclatura-readme.md) |
| 013 | Checklist final de submissao | Respondida | [answered/013-checklist-final-submissao.md](answered/013-checklist-final-submissao.md) |
| 014 | Auditoria de velocidade (Exp106) | Respondida | [answered/014-auditoria-velocidade-exp106.md](answered/014-auditoria-velocidade-exp106.md) |
| 015 | Consistencia: Exp104d vs Exp104b | Respondida | [answered/015-consistencia-exp104d-vs-exp104b.md](answered/015-consistencia-exp104d-vs-exp104b.md) |
| 016 | Performance cross-platform (train + inference) | Respondida | [answered/016-estudo-performance-cross-platform-train-inference.md](answered/016-estudo-performance-cross-platform-train-inference.md) |
| 017 | Benchmark latencia unitaria vs volume CPU/GPU | Respondida | [answered/017-benchmark-latencia-unitaria-vs-volume-cpu-gpu.md](answered/017-benchmark-latencia-unitaria-vs-volume-cpu-gpu.md) |
| 018 | Guia de inferencia: CPU, batch e metricas | Respondida | [answered/018-guia-inferencia-cpu-batch-metricas.md](answered/018-guia-inferencia-cpu-batch-metricas.md) |
| 019 | Real-World Use Case do G2P | Respondida | [answered/019-real-world-use-case-g2p-escopo-taxonomia-e-metricas.md](answered/019-real-world-use-case-g2p-escopo-taxonomia-e-metricas.md) |
| 020 | Proporcionalidade chars/s e pico por hardware | Respondida | [answered/020-proporcionalidade-chars-pico-hardware.md](answered/020-proporcionalidade-chars-pico-hardware.md) |
| 021 | Auditoria de publicacao v1.1 | Respondida | [answered/021-auditoria-publicacao-v1-1.md](answered/021-auditoria-publicacao-v1-1.md) |
| 022 | Meta-ticket: trabalhos futuros v2.0 | Open | [open/022_metrics_and_tf.md](open/022_metrics_and_tf.md) |
| 023 | Empacotamento PIP / distribuicao | Open | [open/023_pip_packaging.md](open/023_pip_packaging.md) |
| 024 | Pipeline fonotatico (4 fases) | Open | [open/024_fonotatica.md](open/024_fonotatica.md) |
| 025 | Espaco articulatorio continuo 7D | Open | [open/025_7d_space.md](open/025_7d_space.md) |
| 026 | Multilingue, Tupi e dialetos | Open | [open/026_multilingual_tupi.md](open/026_multilingual_tupi.md) |
| 027 | Estratificacao de batches | Open | [open/027_batch_stratification.md](open/027_batch_stratification.md) |
| 028 | Morfossintaxe e homografos heterofonos | Open | [open/028_morphosyntax.md](open/028_morphosyntax.md) |
| 029 | Normalizar pasta `evaluations` | Open | [open/029_normalize_folder.md](open/029_normalize_folder.md) |
| 030 | Auditoria: 'y' como glide / representação de 'j' | Open | [open/030_ipa_y_glide_and_j_representation.md](open/030_ipa_y_glide_and_j_representation.md) |
| 031 | Auditoria: ditongos nasais (`ỹ`, `ʊ̃`) | Open | [open/031_nasal_diphthongs_ỹ_ʊ̃_audit.md](open/031_nasal_diphthongs_ỹ_ʊ̃_audit.md) |
| 032 | Organizar fontes originais (`ipa-dict`) e scripts de mapeamento | Open | [open/032_dicts_sources_and_mapping_scripts.md](open/032_dicts_sources_and_mapping_scripts.md) |

## Uso rapido

1. Nova pergunta/TODO: criar arquivo em `open/` e adicionar uma linha no indice.
2. Pergunta resolvida: mover arquivo para `answered/` e atualizar status/arquivo no indice.
3. Detalhes sempre no arquivo do ticket, nunca no README.