# dicts-workbench

Pasta de apoio para fontes brutas, separada de `dicts/`.

Uso atual (simples):
- `sources/ipa-dict/` -> cópia local do repositório ipa-dict
- `sources/ipa-dict/REFERENCE.tsv` -> `file`, `updated_at`, `last_change_reason`
- `sources/ipa-dict/sync_ipa_dict.ps1` -> atualiza essa pasta e regenera a referência

Comandos:
```powershell
./dicts-workbench/sources/ipa-dict/sync_ipa_dict.ps1
./dicts-workbench/sources/ipa-dict/sync_ipa_dict.ps1 -Pull
```
