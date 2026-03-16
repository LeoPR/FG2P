# Scripts README

## Scope
This folder contains operational scripts for metrics extraction, chart generation, report assembly, and cross-evaluation.

## Script Catalog
- `generate_comparative_visualizations.py`
  - Produces: `results/baseline_comparison.png`, `results/class_distribution_top5.png`
  - Uses policy file: `conf/visualization_policy.json`
- `generate_evolution_and_stability.py`
  - Produces: `results/evolution_per_wer.png`, `results/exp0_legacy_mystery.png`, `results/da_loss_gain.png`
- `extract_visualization_data.py` (in `src/`)
  - Data source layer for metrics/metadata consumed by plotting scripts
- `compile_article.py`
  - Article assembly/compilation workflow
- `generate_pdf.py`
  - PDF generation utility for report/document outputs
- `cross_eval.py`
  - Cross-evaluation utility
- `training_regime_analysis.py`
  - Training-regime analysis utility
- `_audit_panphon.py`
  - Internal audit utility

## Usage
Run from repository root:

```powershell
C:/Users/leona/OneDrive/Documents/Projects/Academicos/FG2P/.venv/Scripts/python.exe scripts/generate_comparative_visualizations.py
C:/Users/leona/OneDrive/Documents/Projects/Academicos/FG2P/.venv/Scripts/python.exe scripts/generate_evolution_and_stability.py
```

## Rules
- Keep this file objective. No roadmap narrative here.
- Execution checklists and step-by-step tasks belong in `TODO` files or `docs/evaluations`.
- Policy and selection logic for charts must stay explicit and reproducible.

## Current Maintenance Findings
- Script naming is heterogeneous (`generate_*`, `compile_*`, analysis/audit scripts).
- There are multiple entry points for related outputs (charts vs report), which increases operational drift risk.
- `generate_pdf.py` should be explicitly classified in workflow docs (report pipeline vs chart pipeline).
- Long descriptive script names should be avoided in future additions.

## Organization Plan (Incremental, No Deletions)
1. Define categories in this folder:
   - Chart generation
   - Report/document generation
   - Analysis/evaluation utilities
2. Standardize naming convention:
   - Prefer concise and stable names: `<domain>_<action>.py`
   - Avoid overly long descriptive filenames
3. Add one orchestration entry point later (optional):
   - Example: `scripts/run_visual_pipeline.py`
   - Only orchestrates existing scripts; no business-logic duplication
4. Keep deprecation conservative:
   - Do not delete scripts unless replacement is validated and documented

## Good Practice Baseline
- Single responsibility per script
- Explicit input/output artifacts
- Deterministic selection policy for published figures
- Minimal side effects outside `results/` and declared targets
