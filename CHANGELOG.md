# Changelog — FG2P: G2P para Português Brasileiro

All notable results and milestones are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-03-12 — GitHub Publication Release

### Changed (Publication Readiness)
- **README rewritten**: Improved abstract, three technical foundations (DA Loss, Dataset, Architecture), embedded convergence images
- **PanPhon explanation**: Clarified as genuine articulatory phonetics (place/manner/voicing/height/backness), not metaphorical
- **Project Status**: V1 completion table + Roadmap with chart improvements (broken y-axis, convergence grid, annotation layout)
- **Documentation cleaned**: 30 ephemeral files removed (archive/, analysis docs, AI artifacts, napkin guides)
  - Kept: core article (IMRaD), experiments log, phonological research, linguistic analysis
  - Removed: glossaries, proposal docs, temporary guides, validation planning artifacts
- **INTEGRATION.md**: Fixed API consistency (G2PCorpus constructor, .words attribute), example phoneme outputs synchronized
- **.gitignore**: Added `docs/*.pdf` to prevent tracking third-party papers


### GitHub Status
- ✅ Publication-ready for GitHub
- ✅ Only substantive scientific + user-facing documentation
- ✅ Clean docs/ structure: article/, linguistics/, presentation/, data/

---

## [0.9.0] — 2026-03-08 — Publication Prep

### Added
- `src/inference_minimal.py` — self-contained 150-line inference script, zero external deps beyond PyTorch
- `requirements_inference_only.txt` — single-line requirements for inference use (`torch>=2.0.0`)
- `Dockerfile` + `compose.yaml` — reproducible clean inference environment
- INTEGRATION.md — guide for integrating the model into other projects
- Manager CLI redesigned: `--run`, `--check`, `--clean`, `--compare` as human-friendly verbs
- Tier 2 validation: exp0_training_regime (batch=36, stratify=True) → PER=0.78% (vs legacy 0.38% unstratified)
  - **Conclusion**: Split stratification is dominant factor, not training regime. Legacy result was test set artifact.

### Scientific
- Tier 2 split bias research complete: confirmed stratification impact via control experiment
- Documented graded error distribution: DA Loss shifts errors toward Class B (imperceptible) from Class D (catastrophic)
  - Exp1 vs Exp9 comparison (identical output structure): Class D reduced −0.39pp, Class B increased +0.19pp

---

## [0.8.0] — 2026-03-03 — Tooling & Theory

### Added
- `src/benchmark_inference.py` — chunk-IQR throughput benchmark (robust to OS load)
- Manager `--missing` + `--index N` flags for targeted pipeline runs
- `conf/config_exp0_baseline_70split.json` updated to batch_size=96 (stratified sampling theory)

### Changed
- `model_report.html` — experiment filter now hides all artifacts of other experiments (tables, plots, error analysis)

---

## [0.7.0] — 2026-03-02 — Exp107: SOTA PER

### Results
- **Exp107**: PER **0.46%**, WER 5.56% — new SOTA PER (-6% vs Exp104b)
  - Split: 95% train (~91k), 4% val, 1% test (~960 words)
  - Conclusion: data volume is the primary driver of PER improvement

### Added
- PPTX presentation generator (`src/reporting/presentation_generator.py`)
- `docs/article/DA_LOSS_ANALYSIS.md` — full mathematical derivation with vector examples

---

## [0.6.0] — 2026-02-28 — Phase 6: Distance Override & Ablations

### Results
- **Exp104b**: PER **0.49%** SOTA — DA Loss + syllable separators + post-normalization distance override
- **Exp105**: PER 0.54% with 50% training data — robustness ablation
- **Exp106**: PER 0.58%, speed **30.2 words/s** (2.58× faster by removing hyphen separator)

### Changed
- Docs consolidated: 25 files → 8 files, reorganized into `article/`, `linguistics/`, `presentation/`
- Article `§2.3` added: ɣ/x complementary distribution validated empirically (0 exceptions in corpus)

---

## [0.5.0] — 2026-02-25 — Phase 5: Syllable Separators

### Results
- **Exp101**: baseline + syllable separators → PER −0.06pp, WER +0.83pp (clear trade-off)
- **Exp102**: intermediate + separators → PER −0.06pp, WER +0.83pp (consistent pattern)
- **Finding**: separators improve phoneme-level accuracy at cost of word-level accuracy.
  Not correctable by loss function alone — architectural trade-off.

---

## [0.4.0] — 2026-02-22 — Phase 4: SOTA WER

### Results
- **Exp9**: PER 0.58%, WER **4.96%** — **SOTA WER**
  - Architecture: Intermediate (9.7M params) + DA Loss λ=0.2, no separators
  - Surpasses LatPhon 2025 (WER ~10%+) by ~50%
- **Exp10**: 17.2M params + DA Loss → DA Loss does not scale to larger models (WER worse)

---

## [0.3.0] — 2026-02-19 — Phase 3: DA Loss Discovery

### Results
- **Exp6**: baseline + DA Loss λ=0.2 → PER −7% vs CE-only baseline
- **Exp7**: λ sweep (0.05, 0.20, 0.50) confirms λ=0.2 as optimal
- **Exp8**: PanPhon fixed embeddings + DA Loss → PanPhon ≈ learned embeddings at inference time

### Added
- `src/losses.py` — Distance-Aware Phonetic Loss (DA Loss) implementation
- `src/phoneme_embeddings.py` — PanPhon feature embedding module

---

## [0.2.0] — 2026-02-15 — Phase 2: Architecture Search

### Results
- **Exp3**: PanPhon trainable embeddings → PER 0.32% (competitive but not SOTA)
- **Exp4**: PanPhon fixed 24-dim embeddings → PER 0.36%
- **Exp5**: Intermediate architecture (9.7M) + 60/10/30 split → **PER 0.66% baseline**
- **Finding**: 9.7M parameter sweet spot. 17.2M saturates without quality gain.

---

## [0.1.0] — 2026-02-10 — Phase 1: Baseline

### Results
- **Exp0**: 70/10/20 stratified split — PER 0.59% (modern baseline)
- **Exp1**: 60/10/30 split — PER 0.66% (−41% PER per % of train data)
- **Exp2**: 17.2M params, 512 hidden — diminishing returns vs 9.7M

### Added
- Core architecture: BiLSTM encoder + Bahdanau attention + LSTM decoder
- `dicts/pt-br.tsv` — 95,937 word IPA dictionary (NFC normalized, ɡ U+0261 corrected)
- Stratified train/val/test split (χ²=0.95, Cramér V=0.0007)
- `src/inference_light.py` — full-featured inference API with neologism evaluation

---

## Two SOTA Frontiers

| Metric | Model | Value | Notes |
|--------|-------|-------|-------|
| **Best PER (fair)** | Exp104b | **0.49%** | 28.8k test, DA Loss + sep |
| **Best SOTA PER** | Exp107 | **0.46%** | 960 test (memorization risk) |
| **Best WER** | Exp9 | **4.96%** | DA Loss λ=0.2, no sep |
| **Fastest** | Exp106 | 30.2 w/s | No hyphen sep, 2.58× speedup |

**Publication recommendation**: Use Exp104b (0.49% PER, 28.8k test) as SOTA — larger test set provides
more reliable confidence intervals and avoids memorization risk of Exp107 (960 test words, 95% training).
Both Exp104b and Exp9 reflect the fundamental PER/WER trade-off introduced by syllable separators.
See `docs/article/ARTICLE.md §1.1` for detailed comparison methodology.

---

## Comparison with External Systems

| System | Language | PER | WER |
|--------|----------|-----|-----|
| **FG2P Exp104b** | PT-BR | 0.49% | 5.43% |
| **FG2P Exp9** | PT-BR | 0.58% | **4.96%** |
| LatPhon 2025 | PT-BR | 0.86% [0.56–1.16%] | ~10%+ |
| WFST (Phonetisaurus) | PT-BR | 2.7% (±0.50) | — |

**Exp104b** recommended for publication (larger test set, fair comparison). See `docs/article/ARTICLE.md §1.1` for methodology.
