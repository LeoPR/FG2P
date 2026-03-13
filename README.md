# FG2P — Phonetically-Aware Grapheme-to-Phoneme for Brazilian Portuguese

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)

> **FG2P** converts written Brazilian Portuguese to IPA phonemes with **PER = 0.48%** and **WER = 5.33%** on a stratified test set of 28,782 words (Exp104d). This repository emphasizes transparent evaluation: each claim is tied to explicit metrics, confidence intervals, and dataset context. FG2P uses a distance-aware training signal (PanPhon articulatory distance) to reduce severe phonetic confusions, while keeping comparisons with prior work bounded by clearly stated assumptions.

---

## Key Results

FG2P trained 22 model configurations in a systematic ablation study (§ Systematic Ablation Study below). The main reference in this README is Exp104d.

How to read these numbers:
- `Exp104d` is the main reference configuration for PER-centered reporting and external comparison.
- `Exp9` is the complementary reference configuration for WER-centered use cases.
- The comparison with LatPhon is anchored in PER because WER is not reported in that paper.
- `same source family (ipa-dict)` means the same lexical-resource lineage, not identical subsets or identical train/test splits.
- Residual micro-tradeoff: `Exp104b` is lighter and can be faster on CPU-only scenarios.

| Metric | FG2P (2026) | LatPhon (2025) | Context |
|--------|----------------|----------------|---------|
| **PER (Wilson 95% CI)** | **0.48% ± 0.03** | **0.86% ± 0.30** | PT-BR, same `ipa-dict` lineage, different subset sizes and splits; CIs do not overlap |
| **WER (Wilson 95% CI)** | **5.33% ± 0.26** | n/d | WER not reported for LatPhon |
| Inference speed | 28.4 w/s (RTX 3060) | 31.4 w/s (RTX 4090) | Hardware differs; reported throughput ratio is 1.11x, so read this as context, not a strict winner claim |
| Test set size | **28,782 words** | ~500 words (ipa-dict) | FG2P test is 57× larger |
| Evaluation design | Stratified train/val/test (χ² p=0.678) | Stratification not reported | FG2P reports split validation explicitly |
| Model | 17.2M BiLSTM (2014) | 7.5M Transformer (2017) | Architectural families differ |

**Comparison criteria**: primary metric PER, auxiliary metric WER, uncertainty via Wilson 95% CI, and interpretation bounded to the reported dataset/hardware conditions. External comparison is PER-anchored because that is the metric shared with LatPhon.

**Reading guide**: the PER comparison is favorable to FG2P in this setup (non-overlapping CIs). Numerically, FG2P upper CI bound (0.51%) is below LatPhon lower CI bound (0.56%), while speed and architecture remain contextual trade-offs.

**Caveat**: splits and subset composition are not identical across works, so conclusions should be read as evidence for PT-BR `ipa-dict` conditions, not as a universal ranking.

Detailed experimental evolution and per-experiment motivation are documented in `docs/article/EXPERIMENTS.md`.

![Evolution of PER/WER across experiments](results/evolution_per_wer.png)

### Comparison with Traditional and Modern Baselines

FG2P compared against both classical (WFST) and modern neural baselines on Portuguese:

| Method | Type | Test Set | PER | Hardware | Notes |
|--------|------|----------|-----|----------|-------|
| **FG2P (2026)** | BiLSTM + DA Loss | 28,782 words | **0.48% [0.46–0.51%]** | RTX 3060 12GB | Exp104d; largest test set, stratified split |
| **LatPhon (2025)** | Transformer (seq2seq) | ~500 words | 0.86% [0.56–1.16%] | RTX 4090 | Higher-tier GPU; use as hardware context only |
| **WFST (Phonetisaurus)** | Classical (5-gram) | ~500 words | 2.7% (±0.50) | CPU | Traditional n-gram baseline |
| **ByT5-Small** | Transformer (multilingual) | ~500 words | 9.1% | — | Multilingual model, task confusion |

**Key insight**: In this PER-anchored evaluation setup, FG2P reports lower PER than the reported LatPhon and WFST PT-BR baselines. The non-overlapping PER intervals (FG2P 0.47-0.51% vs LatPhon 0.56-1.16%) support this reading under the documented conditions.

![Baseline comparison: FG2P vs LatPhon, WFST, ByT5-Small](results/baseline_comparison.png)

---

## How It Works: Three Technical Foundations

### 1. Distance-Aware Loss — Real Articulatory Phonetics as Training Signal

Standard G2P uses **CrossEntropy loss**, which minimizes error *count* but treats all errors as equal:
- Predicting [s] instead of [t]: 1 error
- Predicting [u] instead of [t]: also 1 error

Both count the same. FG2P adds a penalty proportional to **articulatory distance** — how different the speech organs must be positioned to produce each sound:

```
L = L_CE + λ × d_panphon(predicted, target) × p(predicted)
```

The penalty `d_panphon` uses **PanPhon's 24-dimensional representation** of actual speech organ configurations. These features are not a metaphor or approximation — they directly encode:

| Feature Group | Examples |
|---------------|----------|
| **Place of articulation** | bilabial [p,b,m], alveolar [t,d,n,s,z], velar [k,g,ɡ], glottal [h] |
| **Manner of articulation** | stop [p,t,k], fricative [f,v,s,z,ʃ], nasal [m,n,ɲ], liquid [l,r] |
| **Voicing** | voiced [b,d,g,z,v] vs. voiceless [p,t,k,s,f] |
| **Vowel height** | high [i,u], mid [e,o], low [a] |
| **Vowel backness** | front [i,e,ɛ], central [a], back [u,o,ɔ] |
| **Nasality** | oral [a,e,o] vs. nasal [ã,ẽ,õ] |

**What this means for gradient descent**: when the model is uncertain between two candidates, it learns to "break ties toward the phonetically closer option" — because choosing a phonetically distant error carries a proportionally larger gradient penalty.

```
d_panphon(e, ɛ) ≈ 0.04   → small penalty   (both mid-front vowels, 1 feature difference)
d_panphon(t, s) ≈ 0.25   → medium penalty  (same alveolar place, different manner)
d_panphon(t, u) ≈ 0.70   → large penalty   (consonant vs. vowel — catastrophic)
```

In matched-output comparisons, the result is a redistribution of errors from **Class D** (catastrophic, phonetically distant) toward **Class B** (phonetically adjacent).

![DA Loss gain: error redistribution toward phonetically close classes](results/da_loss_gain.png)

**Note on PanPhon and non-phonetic tokens**: Syllable separator `.` and stress marker `ˈ` have zero vectors in PanPhon (they are not speech sounds). FG2P corrects for this with **custom structural distances** (consolidated in Exp104d), preventing the model from freely confusing `.` ↔ `ˈ` without penalty. The cleanest DA Loss isolation remains Exp1 vs Exp9 (same output structure).

### 2. Dataset: 95,937 Words, Phonologically Stratified

The training corpus consists of **95,937 (grapheme, IPA) pairs** from `dicts/pt-br.tsv`:

**Data cleaning**: 10,252 instances corrected — the grapheme "g" (U+0067) was mistakenly used where the IPA symbol "ɡ" (U+0261, voiced velar stop) was required. This distinction is critical for correct PanPhon feature lookup.

**Stratified split** by phonological features — each word is assigned to one of ~48 strata based on:
1. **Stress type** — oxytone, paroxytone, proparoxytone
2. **Syllable count bin** — monosyllabic, 2, 3, 4, 5+ syllables
3. **Word length bin** — ≤4, 5–7, 8–10, 11+ characters

Split quality validated with χ² test (χ²=0.95, p=0.678, Cramér V≈0.0007) — no statistically significant difference between train/val/test distributions. The stratification ensures the test set is a representative, unbiased sample of the phonological space.

| Subset | Words | % |
|--------|-------|---|
| Train | 57,561 | 60% |
| Val | 9,594 | 10% |
| Test | 28,782 | 30% |

**Why 60% training?** A larger test set (28,782 words) gives 10× tighter confidence intervals than the ~500-word tests used in comparable work. This sacrifices some training data for statistical rigor — a deliberate trade-off.

### 3. Architecture: BiLSTM Encoder-Decoder with Two Embedding Strategies

FG2P uses a **BiLSTM Encoder-Decoder with Bahdanau attention**:

```
Input: "c o m p u t a d o r"
         |
  [Character Embedding 128D]         ← learned or PanPhon-initialized
         |
  [BiLSTM Encoder 2×256D]            ← reads full grapheme sequence bidirectionally
         |
  [Bahdanau Attention]               ← aligns each output step to relevant input positions
         |
  [LSTM Decoder 2×256D]              ← generates IPA tokens autoregressively
         |
Output: "k õ p u . t a . ˈ d o x"
```

**Two embedding mechanisms** (independent, both explored):

| Strategy | How it works | Key property |
|----------|-------------|--------------|
| **Learned** (default, Exp0–2, 5–10, 101–107) | Random init, fully trained | Emergent structure from co-occurrence context |
| **PanPhon init** (Exp3, Exp4, Exp8) | Initialized from 24 articulatory features | Phonologically structured from epoch 1 — warm start |

**Important**: PanPhon init and DA Loss are **orthogonal mechanisms**. PanPhon init structures the *embedding space* at initialization. DA Loss structures the *gradient signal* during all of training. Ablation experiments (§ Systematic Ablation Study) show DA Loss is the dominant contributor — the learned embedding with DA Loss matches or exceeds PanPhon-initialized embeddings with CE loss.

---

## Training Convergence

22 models trained across systematic ablations. Training typically converges within 30–50 epochs with early stopping (patience=10):

<table>
<tr>
<td align="center"><strong>Exp104d</strong> — DA Loss + sep + structural correction (Main PER reference)</td>
<td align="center"><strong>Exp9</strong> — DA Loss, no sep (Lowest WER in current run set)</td>
</tr>
<tr>
<td><img src="results/exp104d_structural_tokens_correct/exp104d_structural_tokens_correct__20260312_142940_convergence.png" alt="Exp104d convergence"/></td>
<td><img src="results/exp9_intermediate_distance_aware/exp9_intermediate_distance_aware__20260222_064838_convergence.png" alt="Exp9 convergence"/></td>
</tr>
</table>

*Train loss (blue) and val loss (orange). Both models converge stably with early stopping (patience=10). Best checkpoint at minimum val loss.*

---

## Evidence: Output Structure and Fair Comparison

All FG2P models output at least stress markers (`ˈ`); later models additionally output syllable separators (`.`). To compare fairly, PER must be understood relative to what each model attempts:

| Exp | Output Structure | Official PER | Error Composition |
|-----|-----------------|:---:|---|
| **Exp1** (CE Baseline) | phonemes + ˈ | 0.64% | 89.7% phonetic · 10.3% stress |
| **Exp9** (DA Loss) | phonemes + ˈ | 0.61% | 91.6% phonetic · 8.4% stress |
| **Exp101** (CE + sep) | phonemes + ˈ + . | 0.53% | 72.0% phonetic · 3.8% stress · 24.2% sep |
| **Exp103** (DA + sep) | phonemes + ˈ + . | 0.53% | 71.1% phonetic · 4.2% stress · 24.7% sep |
| **Exp104d** (DA + sep + structural correction) | phonemes + ˈ + . | **0.48%** | 72.5% phonetic · 4.1% stress · **23.3% sep** |

**Reading this table**:
- Models with `.` output ~30% more tokens, distributing errors across three token types
- The "phonetic" share (72–92%) is the comparable core across groups
- Exp104d achieves the lowest error rate while outputting the most complex structure

### DA Loss Effect Within Same Output Group (Fairest Comparison)

Comparing only models with syllable separators (identical output structure):

| Exp | PER | vs Exp101 (CE baseline w/ sep) |
|-----|:---:|---|
| Exp101 (CE + sep) | 0.53% | — baseline |
| Exp103 (DA + sep) | 0.53% | =0.00pp |
| **Exp104d** (DA + sep + structural correction) | **0.48%** | **−0.05pp** |

**Within-group conclusion**: Exp104d reduces PER by 0.05pp over the CE baseline with the same output structure, but this gain reflects the combination of DA Loss and structural correction rather than DA Loss in isolation.

### DA Loss Effect on Error *Quality* (Class A–D Distribution)

Comparing Exp1 vs Exp9 — **identical output structure** (phonemes + ˈ only), isolating the DA Loss effect:

| Class | Articulatory Distance | Meaning | **Exp1 (CE)** | **Exp9 (DA Loss)** | **Change** |
|-------|----------------------|---------|:---:|:---:|---|
| A | 0 | Exact match | 94.52% | 94.80% | +0.28pp |
| B | ~1 feature | Phonetically adjacent | 3.53% | **3.72%** | **+0.19pp** ← more near misses |
| C | 2–3 features | Same phoneme family | 0.93% | 0.85% | −0.08pp |
| D | 4+ features | Catastrophic ✗ | 1.02% | **0.63%** | **−0.39pp** ← fewer catastrophic |

**What DA Loss achieves**: In the matched comparison Exp1 vs Exp9, catastrophic errors (Class D) fall by 0.39pp while phonetically adjacent errors (Class B) increase by 0.19pp. This supports the claim that DA Loss changes error quality; direct perceptual impact still requires listening-based validation.

![Error class distribution for top 5 models](results/class_distribution_top5.png)

---

### ⚠️ Fair Comparison: Accounting for Output Structure

**Important caveat**: Some models output **syllable separators** (`.`) and **stress markers** (`ˈ`), while others output phonemes + stress marker only.

- **Exp104d** (with separators): 12.32 tokens/word average
- **Exp1** (without separators): 9.48 tokens/word average
- **Difference**: +30% more output tokens in Exp104d

This means:
- Exp104d's 0.48% PER is achieved on a harder output setting with ~30% more tokens, so direct comparison with no-separator models requires caution.
- Fair comparisons: Only compare models with **identical output structure**
       - Compare Exp104d (0.48%) with Exp103 (0.53%) — both include syllable separators + stress markers
  - Compare Exp1 (0.64%) with Exp9 (0.61%) — both are phonemes + stress marker only
- Cross-paper PER comparisons remain conditional on the documented tokenization and evaluation design.

---

## Why These Results Support Rule Learning

FG2P has 17.2M parameters vs 95.9k vocabulary words. If the model only memorized, a compressed dictionary (≈3 MB) + lookup would be more efficient. Together with the held-out and extra-word evaluations, this pattern is more consistent with learning productive phonological regularities than with pure dictionary lookup, while still leaving room for additional external validation.

---

## Real-World Use Case: Phonetic Error Correction

Imagine a speaker who says "**cinto muito**" (grammatically wrong). A linguistic system needs to recognize this as a *phonetic error* for the intended word "**sinto**" (I feel).

FG2P learns that `C` before `I` → `/s/` (PT-BR soft-C rule), so:
- Predicted: /s/ ✓ (correct — learned the rule)
- Even if wrong, error would be near /s/, not random

**This is what makes FG2P suitable for downstream tasks**:
- TTS: Near-miss errors tend to be less salient; distant errors are more likely to break intelligibility
- NLP: Phonetically close predictions help error recovery
- Linguistics: Error patterns reflect natural phonological rules

---

## Quick Start

### Python API

```python
from src.inference_light import G2PPredictor

# Load recommended aliases from model registry
predictor = G2PPredictor.load("best_per")   # Lowest PER in current registry (TTS-oriented)
predictor = G2PPredictor.load("best_wer")   # Lowest WER in current registry (NLP/search-oriented)

# Predict
print(predictor.predict("computador"))  # k õ . p u t a . 'do x
print(predictor.predict("borboleta"))   # b o x . b o . l e t a
```

**IPA Character Display**:
If IPA characters don't render correctly in your editor, configure UTF-8 encoding and see [IPA_REFERENCE.md](IPA_REFERENCE.md) for complete symbol descriptions (e.g., `x` = rótico final, `ɣ` = rótico antes vozeado, `ə` = schwa).

### Command Line

```bash
# Install
pip install -r requirements_inference_only.txt  # only torch

# Predict
python src/inference_light.py --alias best_per --word "computador"
python src/inference_light.py --interactive

# Evaluate custom data
python src/inference_light.py --neologisms docs/data/generalization_test.tsv

# List available models
python src/inference_light.py --list
```

### Minimal Version (copy-paste, no dependencies beyond torch)

```python
from inference_minimal import G2PPredictor  # copy src/inference_minimal.py + model
predictor = G2PPredictor.load("best_per")
print(predictor.predict("computador"))
```

---

## Model Selection and Trade-offs

**Key principle**: Choose models based on **stability and generalization**, not single-run peak performance.

FG2P uses internal experiment IDs in the form `ExpN` or `ExpN[a-z]` to track each configuration, where `N` is a variable-length integer and the optional lowercase suffix marks a revision or branch of the same experiment family (for example, Exp104b, Exp104d). Aliases like `best_per` and `best_wer` point to the current recommended experiments in the model registry.

### Recommended Models

| Use Case | Model | PER | WER | Speed | Reason |
|----------|-------|-----|-----|-------|--------|
| **TTS / Publication** | `best_per` (Exp104d) | **0.48%** | 5.33% | 12.7 w/s | Outputs phonemes + stress + syllable structure; largest test set |
| **NLP / Search** | `best_wer` (Exp9) | 0.61% | **4.96%** | 34.5 w/s | Lowest word error rate in current registry; no separators = clean phoneme output |
| **Efficiency Ablation (not promoted)** | Exp106 | 0.58% | 6.12% | under speed audit | 50% train data, no hyphens — keep as exploratory until replicated benchmark closes |

**Note**: Exp106 remains useful as a linguistic ablation (hyphen removal with small PER/WER impact), but its speed claim is currently treated as preliminary.

### Train/Test Split: Why 60% Training?

| Train % | Test Words | Model | PER | Finding |
|---------|-----------|-------|-----|---------|
| **60%** | 28.8k (Exp104d) | **✓ RECOMMENDED** | 0.48% | **Most defensible reference setting**: large stratified test set, stable metrics |
| 50% | 38.4k (Exp105) | Ablation | 0.54% | Less training → forced learning (slightly worse) |
| 95% | 960 (Exp107) | ❌ Avoid | 0.46% | Tiny test set (960 words); high risk of **memorization** |

**Interpretation**: Exp107's 0.46% PER with 95% training and only 960 test words *looks* better but **risks overfitting**. Exp104d's 0.48% with 28.8k test words is more trustworthy and reproducible.

### On Metric Inflation: Dataset Bias in Exp0 and Exp1

**Warning**: Early experiments (Exp0, Exp1) report excellent metrics but suffer from **dataset bias** — lack of stratification can inflate results.

| Exp | PER | WER | Test Size | Design | Issue | Status |
|-----|-----|-----|-----------|--------|-------|--------|
| **Exp0** | **0.38%** | **3.41%** | 19.2k | Random split, no stratification | ⚠️ Confirmed bias: same regime + stratified split → 0.78% PER (Tier 2) | INFLATED |
| **Exp1** | **0.64%** | **5.48%** | 28.8k | Random split, no stratification | ⚠️ Potential bias | INFLATED |
| **Exp104d** | 0.48% | 5.33% | 28.8k | **Stratified split (χ² p=0.678)** | ✓ Validated | ROBUST |

**Empirical confirmation** (Tier 2 control experiment): Running the Exp0 training regime (batch=36, no early stopping) with `stratify=True` produced **0.78% PER** — 2× worse than the original 0.38%. The 0.38% was a test set sampling artifact, not algorithmic superiority. Details in [docs/article/EXPERIMENTS.md](docs/article/EXPERIMENTS.md).

### Systematic Ablation Study

22 models trained to isolate factors:

| Model | Config | PER | WER | Capacity | Sep | Loss | Purpose |
|-------|--------|-----|-----|----------|-----|------|---------|
| **Exp104d** | **Recommended** | **0.48%** | 5.33% | 17.2M | ✓ | DA | Main publication-quality reference model |
| Exp107 | High train % | 0.46% | 5.56% | 9.7M | ✓ | DA | Shows risk of memorization |
| Exp9 | No separators | 0.61% | **4.96%** | 9.7M | ✗ | DA | Lowest observed WER in current registry |
| Exp1 | CE baseline | 0.64% | 5.48% | 4.3M | ✗ | CE | Shows DA Loss helps |
| Exp5 | CE + capacity | 0.63% | 5.38% | 9.7M | ✗ | CE | Capacity alone insufficient |
| Exp102 | Sep only (CE) | 0.53% | 5.79% | 9.7M | ✓ | CE | Sep helps PER, hurts WER |
| Exp103 | Sep + DA | 0.53% | 5.73% | 9.7M | ✓ | DA | Without custom dist |
| Exp2 | Extended capacity | 0.60% | 4.98% | 17.2M | ✗ | CE | Diminishing returns at 17.2M |
| Exp3 | PanPhon embeddings | 0.66% | 5.45% | 4.3M | ✗ | CE | Neutral for PT-BR |

**Key findings**:
- **DA Loss effect**: Redistribution of Class D errors to Class B (0.39pp reduction in catastrophic errors, same output structure)
- **Dataset design matters**: Exp0 (0.38%, biased) vs Exp104d (0.48%, stratified) — unbiased metrics are higher but trustworthy
- **Capacity sweet spot**: 9.7M — further increase to 17.2M shows diminishing returns
- **Syllable separators**: PER ↓0.04pp, but WER ↑0.47pp (use for TTS, avoid for NLP)
- **Stability**: Exp104d shows stable behavior in long-word and compound-name stress tests
- **Generalization**: Model generalizes beyond training vocabulary to new word constructions

---

## Architecture Diagram

```
Input: "c o m p u t a d o r"
         |
  [Character Embedding 128D]
         |
  [BiLSTM Encoder 2x256D]
         |
  [Bahdanau Attention]
         |
  [LSTM Decoder 2x256D]          Loss = CE + lambda * d(pred, target) * p(pred)
         |                              |
Output: "k o~ p u t a . 'do x ."       [PanPhon: 24D articulatory features]
                                        [place, manner, voicing, height, backness, nasality...]
```

- **Encoder**: Bidirectional LSTM processes grapheme sequence
- **Attention**: Bahdanau additive attention aligns graphemes to phonemes
- **Decoder**: Autoregressive LSTM generates IPA phoneme sequence
- **DA Loss**: Penalizes errors proportionally to articulatory distance (PanPhon 24 features)
- **Training**: Adam optimizer, early stopping (patience=10), stratified train/val/test split

---

## Project Structure

```
fg2p/
  src/
    g2p.py                  # Core: CharVocab, PhonemeVocab, G2PLSTMModel, G2PCorpus
    train.py                # Training with early stopping + DA Loss
    inference_light.py      # User-facing prediction API (START HERE)
    inference_minimal.py    # 150-line self-contained predictor (copy-paste)
    inference.py            # Full evaluation pipeline (batch eval + metrics)
    losses.py               # Distance-Aware Loss + Soft Target CE
    phonetic_features.py    # PanPhon feature extraction + error classification
    phoneme_embeddings.py   # Learned / PanPhon embedding layers
    analyze_errors.py       # PER, WER, Wilson CI, graduated metrics
    manage_experiments.py   # Experiment pipeline CLI
    reporting/              # HTML report + PPTX generation

  dicts/pt-br.tsv           # 95,937 words with IPA transcriptions
  conf/                     # Training configs (JSON, one per experiment)
  models/                   # Checkpoints (.pt) + metadata (.json)
  results/                  # Evaluations, predictions, convergence plots
  docs/
    article/ARTICLE.md      # Scientific article (IMRaD format)
    article/EXPERIMENTS.md  # Detailed experiment log
    article/REFERENCES.bib  # Bibliography (BibTeX)
    data/                   # Test datasets (generalization, neologisms)
```

---

## Training Your Own Model

```bash
# Full dependencies
pip install -r requirements.txt

# Train baseline
python src/train.py --config conf/config_exp1_baseline_60split.json

# Train with Distance-Aware Loss
python src/train.py --config conf/config_exp9_intermediate_distance_aware.json

# Run evaluation pipeline
python src/manage_experiments.py --run N        # N = experiment index
python src/manage_experiments.py --missing      # check which experiments need processing
python src/manage_experiments.py --check        # verify consistency
```

---

## Performance & Generalization

### Inference Speed: GPU vs CPU Benchmark

Measured with `scripts/benchmark_inference.py` on **NVIDIA RTX 3060 12GB** (consumer GPU, 16-core CPU):

| Device | Model | Throughput | Avg Latency | P50 Latency | P95 Latency | Real-time* |
|--------|-------|-----------|-------------|-----------|-----------|-----------|
| **GPU** | **best_per** | **28.4 w/s** | 35.23 ms | 32.71 ms | 45.71 ms | ✓ 5.6× |
| **CPU** | **best_per** | **27.9 w/s** | 35.81 ms | 32.97 ms | 46.99 ms | ✓ 5.5× |
| **GPU** | **best_wer** | **34.5 w/s** | 28.97 ms | 27.45 ms | 36.55 ms | ✓ 6.9× |
| **CPU** | **best_wer** | **33.7 w/s** | 29.66 ms | 27.97 ms | 38.03 ms | ✓ 6.7× |

*Real-time factor: speedup relative to 5 w/s TTS threshold. >1.0 = faster than real-time.

**Key findings**:
- **GPU ≈ CPU**: Practically identical performance (1–2% difference) — achieved on consumer-grade RTX 3060
- **Hardware context**: LatPhon (2025) reported 31.4 w/s on RTX 4090, while FG2P reports 28.4 w/s on RTX 3060. Because hardware and evaluation conditions are not identical, this should be read as similar reported throughput under different setups, not as a direct speed ranking.
- **Reason**: Model is small (9.7M params); GPU transfer overhead ≈ computation latency
- **Implication**: CPU-friendly inference viable — no GPU required for production deployment
- **best_wer 20% faster** than best_per (no syllable separators = shorter output sequences)

### Full Multi-Model Benchmark (CPU + GPU, chars/s + CI95)

To enrich speed diagnostics beyond w/s, a full benchmark over all complete checkpoints was run with:

```bash
python scripts/benchmark_inference.py --all-models --warmup 8 --runs 40
```

Artifact with all models (19 checkpoints), devices, CI95, tokens/s, input chars/s and output chars/s:
- [results/benchmarks/benchmark_all_models_2026-03-13.txt](results/benchmarks/benchmark_all_models_2026-03-13.txt)

Interpretation note: this run reported contention/thermal warnings in multiple models, so values should be read as an audit snapshot (not a final publication table yet).

| Model (index) | GPU global w/s [CI95] | CPU global w/s [CI95] | GPU chars/s (in/out) | CPU chars/s (in/out) |
|---|---:|---:|---:|---:|
| Exp9 (index:6) | 14.9 [14.5, 15.4] | 28.7 [27.3, 30.2] | 164 / 164 | 316 / 316 |
| Exp104b (index:8) | 12.6 [12.2, 13.1] | 24.4 [23.3, 25.7] | 139 / 184 | 269 / 357 |
| Exp104d (index:18) | 12.7 [12.3, 13.2] | 16.2 [15.4, 17.1] | 140 / 186 | 178 / 236 |

### Generalization to Unseen Data

FG2P results are more consistent with rule learning than with pure memorized lookup. Evidence:

| Category | Test | Result |
|----------|------|--------|
| **In-vocabulary** (28.7k words) | Stratified test set (main evaluation) | 94.38% accuracy |
| **Extra: Constructed words** | 31 words (6 categories, outside dicts/pt-br.tsv) | 17/31 (55%) — limited by OOV chars k/w/y |

**What this means**: The model generalizes to some word patterns it did not see during training, which is consistent with learning productive phonological regularities (C→/s/ before I, r-coda assimilation, vowel neutralization in unstressed syllables, etc.) rather than acting only as a dictionary lookup. The extra-word set is still small, so this should be read as supportive evidence rather than universal proof.

---

## Documentation & Technical References

This project documents all metrics, formulas, and implementation details across multiple layers:

### High-Level Overview (You Are Here)
- **README.md**: Project motivation, key results, architecture, generalization analysis

### Low-Level Technical Details
- **[docs/article/FORMULAS.md](docs/article/FORMULAS.md)** — Complete mathematical reference for all metrics:
  - PER formula and implementation
  - WER formula and Wilson 95% CI
  - Graduated metrics (classes A/B/C/D, error distribution)
  - Wilson CI theory and numerical stability
  - Examples with real FG2P numbers

### Experimental Design & Results
- **[docs/article/EXPERIMENTS.md](docs/article/EXPERIMENTS.md)** — Full ablation study (22 models), methodological choices, design decisions
- **[docs/article/ARTICLE.md](docs/article/ARTICLE.md)** — Peer-review ready manuscript (IMRaD format)
- **[docs/evaluations/](docs/evaluations/)** — Ongoing evaluation notes & research robustness assessment

### How Metrics Flow Through Code
- **Formula** → [`docs/article/FORMULAS.md`](docs/article/FORMULAS.md) (theory + math)
- **Implementation** → [`src/analyze_errors.py`](src/analyze_errors.py) (lines 200–295)
- **Usage Example** → [`src/inference_light.py`](src/inference_light.py) (lines 630–750)
- **Model Eval** → [`src/inference.py`](src/inference.py) (batch evaluation with CI output)

**Example**: PER with 95% CI appears as `PER: 0.49% [0.47%, 0.51%]`
- Calculation: 1050 edit errors ÷ 181,000 reference phonemes = 0.58%
- Formula: Wilson CI at 𝛼=0.05 with 𝑧=1.96
- See: [FORMULAS.md §2.3](docs/article/FORMULAS.md#23-wilson-ci-for-per) for derivation

---

## Project Status

### V1 Complete ✅

| Area | Status |
|------|--------|
| Pipeline (22 models) | ✅ All experiments evaluated: eval + error_analysis + convergence plots |
| Distance-Aware Loss | ✅ Implemented, ablated (λ sweep: 0.05/0.10/0.20/0.50), optimal λ=0.20 |
| PanPhon audit | ✅ 100% PT-BR phoneme coverage, zero unresolved conflicts |
| Stratified splits | ✅ χ² p=0.678 balance validation, split bias confirmed (Tier 2) |
| Wilson CI | ✅ FG2P [0.47%, 0.51%] vs LatPhon [0.56%, 1.16%] — non-overlapping |
| Scientific article | ✅ docs/article/ARTICLE.md v1.2 complete (IMRaD, §1–§6) |
| Reproducibility | ✅ ±0.02pp PER variance between identical runs (D1 validation) |
| Exp104c | ✅ Rodado — ablação de capacidade válida (17M CE sem sep, WER 4.92% novo record); errata documentada em `conf/config_exp104c_structural_tokens.json` |

### Roadmap

| Priority | Item | Description |
|----------|------|-------------|
| High | **Exp104d** | Corrected structural token experiment: 17M + sep silábico + DA λ=0.2 (fixes exp104c config bugs) |
| High | **Revisão de λ com sep** | λ=0.2 foi otimizado em Exp7 (sem sep). Com sep silábico + custom dist, o ótimo pode ser diferente — sweep λ∈{0.1, 0.2, 0.3, 0.5} no regime sep+DA |
| Medium | **Class E errors** | Fifth error class for structural token confusions (post-publication) |
| Medium | **Chart: convergence grid** | Show Exp1 / Exp9 / Exp104b convergence side-by-side (currently only Exp104b in README) |
| Medium | **Chart: da_loss_gain layout** | Annotation boxes can overlap bars; move gain labels below chart area |
| Low | **PanPhon embedding analysis** | Spearman correlation: do DA Loss embeddings develop articulatory structure? |
| Low | **DA Loss failure at 17.2M** | Why over-regularization at Exp2 capacity? |
| Low | **Data ablation** | Fixed test set ablation for clean train-size effect curves |

---

## Citation

```bibtex
@article{fg2p2026,
  title={FG2P: Distance-Aware Loss for Phonetically Controlled Errors
         in Brazilian Portuguese Grapheme-to-Phoneme Conversion},
  author={Peixoto, Leonardo R.},
  year={2026},
  note={PER 0.49\%, 9.7M params, 28.8k stratified test words.
        Available at https://github.com/LeoPR/FG2P}
}
```

See [docs/article/REFERENCES.bib](docs/article/REFERENCES.bib) for the complete bibliography.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](QUICKSTART.md) | 2-minute Docker setup guide |
| [docs/INTEGRATION.md](docs/INTEGRATION.md) | Integration guide for using FG2P in your project |
| [docs/article/ARTICLE.md](docs/article/ARTICLE.md) | Scientific article (IMRaD, §1–§6) |
| [docs/article/EXPERIMENTS.md](docs/article/EXPERIMENTS.md) | Experiment log (Exp0–107) |
| [docs/article/REFERENCES.bib](docs/article/REFERENCES.bib) | Bibliography (BibTeX) |
| [IPA_REFERENCE.md](IPA_REFERENCE.md) | IPA symbol reference for PT-BR phonemes |
| [docs/linguistics/PHONOLOGICAL_ANALYSIS.md](docs/linguistics/PHONOLOGICAL_ANALYSIS.md) | PT-BR phonological rules and IPA validation |

---

## License

MIT License. See [LICENSE](LICENSE).

Pretrained models and datasets are provided for research and educational use.
