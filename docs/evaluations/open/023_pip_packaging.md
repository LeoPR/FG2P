ID: 023
Title: Empacotamento PIP / Distribuição
Type: packaging
Priority: High
Status: Open

Descrição:
Planejar o empacotamento do repositório como pacote pip (`fg2p`) preservando a interface desejada:

Exemplo de uso esperado (API pública):
```py
from FG2P import G2P
predictor = G2P.load("best_per")
print(predictor.predict("computador"))
```

Escopo técnico (documentação do trabalho futuro, sem codificar agora):
- checklist de arquivos necessários (`pyproject.toml`, `README`, `LICENSE`, `MANIFEST.in`/incluir assets), políticas de versão (semver), publicação (TestPyPI/PyPI) e CI de release automático.
- instruções de instalação e uso (exemplos de `pip install fg2p` e uso importado).
- organizar onde colocar modelos pesados (p.ex., assets em `models/` e estratégia de download no primeiro uso) — documentar opções (PyPI bundle vs. separate hosting).

Critérios de aceite:
- Documento com checklist completo e instruções CLI/CI para empacotamento e publicação.
- Exemplo de uso (mostrado acima) incluído no ticket e no `README` do repo.

Próximos passos:
1. Escrever documento de empacotamento (este ticket).
2. Criar subtasks de implementação (pyproject, CI, release notes) quando autorizado.
