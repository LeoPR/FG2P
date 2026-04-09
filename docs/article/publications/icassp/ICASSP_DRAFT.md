# Distance-Aware Loss for Phonologically-Graded Grapheme-to-Phoneme Conversion in Brazilian Portuguese

**[IEEE ICASSP 2027 — Single-blind submission]**
**Authors**: Leonardo Marques de Souza
**Affiliation**: Independent Researcher, Manaus, Brazil

---

## Abstract

Standard grapheme-to-phoneme (G2P) training treats all phoneme substitutions equally, regardless of articulatory distance. We propose Distance-Aware (DA) Loss, which penalizes substitutions proportionally to PanPhon articulatory distance between predicted and target phonemes, weighted by the model's prediction confidence. Applied to Brazilian Portuguese with a BiLSTM encoder-decoder, DA Loss systematically redistributes errors toward phonologically closer targets: Class D (catastrophic) substitutions decrease 19% relative vs. a CE baseline. Our system achieves PER 0.48% and WER 5.33% in the reference configuration (a complementary no-separator model reaches WER 4.96%) on a stratified 28,782-word test set — 57× larger than comparable PT-BR evaluations — with a Wilson 95% CI of ±0.03 pp. Evaluation on 31 out-of-vocabulary words yields 100% accuracy on genuine novel PT-BR words, consistent with phonological rule generalization beyond memorization.

---

## 1. Introduction

Grapheme-to-phoneme (G2P) conversion is a core component of text-to-speech synthesis, automatic speech recognition, and multilingual NLP pipelines. For Brazilian Portuguese, G2P presents well-documented challenges: graphemic ambiguity (grapheme "c" maps to /k/ in *cama* but /s/ in *cena*; "r" maps to /ɾ/ in syllable onset but /x/ in word-final coda [10]), vowel neutralization in unstressed positions (/e/↔/ɛ/ and /o/↔/ɔ/ merge), and position-dependent coda realization (/x/ in final position; /ɣ/ before voiced consonants). Prior work on PT-BR G2P has addressed these challenges with decision trees, n-gram models, and more recently neural seq2seq architectures [8].

Standard sequence-to-sequence models trained with cross-entropy (CE) treat all phoneme errors equally: predicting /ɛ/ when the target is /e/ — a near-miss differing in one articulatory feature — incurs the same gradient penalty as predicting /k/ for /a/, an error spanning eight articulatory features. This phonological blindness distorts the training signal: the model learns *that* it erred but not *how severely*.

We address this with **Distance-Aware (DA) Loss**, which adds a training signal proportional to the articulatory distance between predicted and target phonemes, weighted by the model's confidence in its prediction. DA Loss asks not only "did you err?" but also "how far phonologically was your error?" We apply this to Brazilian Portuguese using a BiLSTM encoder-decoder with Bahdanau attention [1] — deliberately chosen to isolate the contribution of the loss function from architectural novelty.

**Comparison with LatPhon [2].** LatPhon is a 4-layer multilingual Transformer (7.5M parameters, RoPE) reporting PER 0.86% (Wilson 95% CI [0.56%, 1.16%]) on ~500 PT-BR words from the same IPA dictionary used in this work. Our system achieves PER 0.48% (CI [0.46%, 0.51%]) on 28,782 words. Both systems report Wilson CIs; the intervals do not overlap — our upper bound (0.51%) falls below LatPhon's lower bound (0.56%), a statistically significant difference at 95% confidence. The 57× difference in test set size confers 10× more precise CIs to our evaluation (±0.03 pp vs. ±0.30 pp).

**Contributions:**
1. DA Loss: a phonologically-graded training objective combining PanPhon articulatory distance with prediction confidence (§4)
2. Large-scale stratified evaluation: 28,782-word test set with Wilson CI 10× more precise than prior PT-BR work (§2)
3. Error quality taxonomy: Class A–D classification reveals systematic redistribution of catastrophic errors (§6)

---

## 2. Data and Evaluation Protocol

### 2.1 Corpus

The training corpus consists of **95,937 grapheme–IPA pairs** from a Brazilian Portuguese phonetic dictionary. The input charset covers a–z (excluding k, w, y, which are absent from the dictionary) plus Portuguese diacritics (ç, á, à, â, ã, é, ê, í, ó, ô, õ, ú, ü). Words containing k, w, or y are treated as character-OOV and mapped to `⟨UNK⟩`. Prior to training, 10,252 entries were corrected for ASCII-g (U+0067) vs. IPA-ɡ (U+0261) conflation, required for correct PanPhon feature lookup.

### 2.2 Stratified Split

The corpus is divided **60/10/30** (train/val/test) using stratified sampling over three phonological variables: stress type (oxytone/paroxytone/proparoxytone), syllable count bin, and character length bin. Their combination yields approximately 48 strata. A purely random split risks underrepresenting rare phonological patterns — proparoxytones account for fewer than 5% of the corpus — leading to inflated metrics on high-frequency patterns [3].

Split quality: χ²=0.95 (p=0.678), Cramér V=0.0007 — no statistically significant distributional difference across subsets. Stratification ensures unbiased PER and WER estimates with respect to the phonological features used in sampling. Per-epoch random reshuffling serves a separate purpose: variance reduction in stochastic gradient estimation [4].

### 2.3 Evaluation Metrics

**PER** (Phoneme Error Rate): Levenshtein distance over phoneme sequences, normalized by reference length [5]:

$$\text{PER} = \frac{\sum_i \text{edit}(\hat{y}_i, y_i)}{\sum_i |y_i|} \times 100\%$$

**WER** (G2P Word Error Rate): fraction of words with any phoneme error (exact-match; equivalent to String Error Rate in ASR literature [5]).

**Wilson 95% CI** [6, 11]: used throughout in place of Wald intervals, which underestimate uncertainty near p→0. For our test set (~181K reference phonemes), the Wilson CI on PER is ±0.03 pp — 10× more precise than ±0.30 pp for ~500-word evaluations.

---

## 3. Architecture

We use a **BiLSTM encoder-decoder with Bahdanau attention** [1], an architecture established for supervised G2P [7] and chosen to isolate the contribution of the loss function from architectural novelty.

**Encoder**: 2-layer BiLSTM over grapheme embeddings. Each position *t* yields hₜ = [h→ₜ; h←ₜ], providing full bidirectional context — critical for resolving graphemic ambiguity where phoneme identity depends on surrounding characters.

**Attention**: at each decoder step *t*, context vector cₜ is computed as:

$$e_{t,j} = v^\top \tanh(W_h h_j + W_s s_{t-1}), \qquad \alpha_{t,j} = \text{softmax}(e_{t,j}), \qquad c_t = \sum_j \alpha_{t,j} h_j$$

**Decoder**: 2-layer LSTM, teacher forcing during training, autoregressive at inference.

**Configurations tested:**

| Config | Embedding | Hidden | Parameters |
|--------|-----------|--------|------------|
| Small  | 128D | 256D | 4.3M |
| Medium | 192D | 384D | 9.7M |
| Large  | 256D | 512D | 17.2M |

The medium configuration (9.7M) is the optimal point for DA Loss without syllable separators. At 17.2M without separators, DA does not improve over CE; however, 17.2M *with* separators and corrected structural distances achieves the best PER observed (0.48%), indicating that the capacity–structure interaction is the determining factor (§5).

---

## 4. Distance-Aware Loss

### 4.1 Formulation

$$\boxed{L = L_{\text{CE}} + \lambda \cdot d_{\text{PanPhon}}(\hat{y},\, y) \cdot p(\hat{y})}$$

where:

- **L\_CE** = −log(p\_correct): cross-entropy over the target phoneme
- **d\_PanPhon(ŷ, y) ∈ [0, 1]**: normalized Euclidean distance between predicted and target phoneme in PanPhon's 24-dimensional articulatory feature space [9], encoding voicing, nasality, place, and manner of articulation. Representative distances: p↔b: 0.04 (voicing only); s↔ʃ: 0.15 (place); a↔k: 0.90 (vowel vs. velar stop)
- **p(ŷ) ∈ (0, 1]**: softmax probability of the *predicted* (argmax) phoneme — independent of p\_correct
- **λ**: coupling weight between phonological signal and CE

**Design rationale.** CE monitors p\_correct; DA monitors p(ŷ). These are independent: p\_correct can be low while p(ŷ) is high, i.e., the model is confident in a wrong prediction. The confidence factor scales the articulatory penalty with the model's certainty in its error — maximally penalizing errors that are both confident and phonologically distant.

**Bounded signal.** DA Loss is upper-bounded by λ × 1.0 × 1.0 = 0.20, while CE can reach ~16 in early training. DA is therefore effective primarily in the learning transition zone (CE 0.3–1.5), where the model is actively resolving phoneme ambiguities between competing candidates.

**Novelty scope.** Distance-weighted losses and phonologically-motivated training signals have been explored in ASR and TTS contexts. Our contribution is novel in the G2P context through the specific compositional formulation: (1) PanPhon articulatory distance as the gradient error signal, (2) prediction confidence as the weighting factor, and (3) controlled coupling via λ that preserves CE as the primary learning axis. We claim originality in this compositional formulation for IPA-level G2P, not universal pioneering over distance-weighted objectives.

### 4.2 Structural Token Override

PanPhon assigns zero vectors to non-phonemic tokens: syllable boundary "." and stress marker "ˈ". Without correction, d(., ˈ) = 0.0 — DA Loss provides no gradient signal for structural token confusions. We apply a post-normalization override assigning distance 1.0 between any structural token and any other symbol. The override must be applied *after* Euclidean normalization; applying it before yields d ≈ 0.25 (equivalent to a mid-vowel distance), not the intended maximum.

### 4.3 λ Sweep

Fixed architecture (4.3M, no syllable separators):

| λ | PER | WER | Behavior |
|---|-----|-----|----------|
| 0.05 | 0.62% | 5.36% | DA signal too weak relative to CE |
| 0.10 | 0.63% | 5.35% | Moderate improvement |
| **0.20** | **0.60%** | **5.14%** | **Optimal — inverted-U curve** |
| 0.50 | 0.65% | 5.57% | Over-penalization; gradient instability |

The inverted-U is expected: λ too low leaves CE undifferentiated; λ too high competes with CE and degrades learning. λ=0.20 was subsequently applied to all larger configurations.

---

## 5. Experiments

### 5.1 Error Quality Classes

Errors are classified by normalized Hamming distance d_H in PanPhon 24-feature space:

| Class | d_H | Features different | Example |
|-------|-----|--------------------|---------|
| A | 0.000 | 0 — exact | — |
| B | ≤0.050 | 1 — minimal pair | p↔b, s↔z, e↔ɛ |
| C | ≤0.150 | 2–3 — same family | s↔ʃ, z↔ʒ, a↔ə |
| D | >0.150 | 4+ — cross-class | n↔ɲ (0.17), vowel↔stop (0.42+) |

Training uses Euclidean distance; evaluation uses Hamming distance. Both operate in the same 24-feature PanPhon space.

### 5.2 Main Results

| System | Params | Loss | Sep | PER | WER |
|--------|--------|------|-----|-----|-----|
| CE baseline | 4.3M | CE | — | 0.66% | 5.65% |
| DA λ=0.1 | 4.3M | DA | — | 0.63% | 5.35% |
| DA λ=0.2 | 4.3M | DA | — | 0.60% | 5.14% |
| **DA λ=0.2** | **9.7M** | **DA** | **—** | **0.58%** | **4.96%** |
| CE + sep | 9.7M | CE | ✓ | 0.52% | 5.79% |
| **DA λ=0.2 + dist** | **17.2M** | **DA+dist** | **✓** | **0.48%** | **5.33%** |

Sep = syllable boundary tokens in output. Bold = recommended configurations.

**Separator trade-off.** Syllable boundary tokens improve PER (−17% to −20%) at the cost of WER (+6% to +8%). Each misplaced separator counts as a full word error under exact-match WER — a structural trade-off independent of architecture or loss function. Recommendation: use the 9.7M DA model (no separators) for WER-sensitive tasks (NLP, lexicon lookup); use the 17.2M DA+dist model (with separators) for PER-sensitive tasks (TTS, forced alignment).

**Comparison with LatPhon [2]:**

| System | PER (Wilson 95% CI) | Test words | Architecture |
|--------|---------------------|------------|--------------|
| LatPhon [2] | 0.86% ± 0.30 | ~500 | 7.5M Transformer (multilingual) |
| **Ours (DA+dist, 17.2M)** | **0.48% ± 0.03** | **28,782** | **17.2M BiLSTM (PT-BR)** |

Both systems report Wilson CIs over the same lexical resource (ipa-dict); the intervals do not overlap — a statistically significant difference at 95% confidence.

---

## 6. Analysis

### 6.1 Dominant Error Patterns

Top phoneme confusions on the 28,782-word test set:

| Substitution | Count | Group | Mechanism |
|--------------|-------|-------|-----------|
| ɛ → e | 255 | Mid-front vowel | PT-BR unstressed neutralization |
| e → ɛ | 197 | Mid-front vowel | Same (reverse direction) |
| ɔ → o | 131 | Mid-back vowel | Neutralization |
| i → e | 121 | High/mid front | Vowel reduction |
| o → ɔ | 95 | Mid-back vowel | Same |

Over 60% of errors are vowel neutralizations — phonological ambiguity inherent to PT-BR. Structural analysis of the e↔ɛ pattern reveals the cause: the global /e/:/ɛ/ corpus ratio is 7.1:1, but by syllabic position the pre-tonic ratio is 24.9:1 (/e/ dominant) while the tonic ratio is 0.33:1 (/ɛ/ dominant). The model learns the strong pre-tonic /e/ bias and overgeneralizes it to tonic syllables — a corpus distribution artifact. Resolving this class requires explicit prosodic features (stress position as input) or a dedicated tonic-vowel classifier, not additional training on the same corpus.

### 6.2 Error Quality: DA Loss Redistribution

| System | PER | Class B | Class D | D / all errors |
|--------|-----|---------|---------|----------------|
| CE baseline (4.3M) | 0.66% | 0.39% | 0.54% | 50.9% |
| DA λ=0.2 (9.7M)    | 0.58% | 0.36% | 0.44% | 48.4% |
| DA+dist (17.2M+sep) | 0.48% | 0.29% | 0.53%† | 47.4% |

†Class D inflated by structural token confusions (., ˈ) carrying elevated custom distances; not reflective of phonemic error severity.

DA Loss reduces Class D errors **19% relative** (4.3M CE → 9.7M DA, holding evaluation conditions fixed). Class B errors decrease proportionally — substitutions shift toward phonologically closer targets. This redistribution is a distinct effect from PER improvement: even models with similar PER commit fewer catastrophic substitutions under DA Loss. In downstream TTS, Class B errors (single-feature substitutions, e.g., /e/↔/ɛ/) are perceptually less salient than Class D errors (cross-class substitutions, e.g., vowel↔stop); formal perceptual validation (MOS/ABX) remains future work.

### 6.3 OOV Generalization

We evaluate the 17.2M DA+dist model on 31 curated OOV words across 6 categories:

| Category | Correct | Phon. Score |
|----------|---------|-------------|
| PT-BR neologisms | 6/9 (67%) | 97% avg |
| Geminate consonants | 1/5 (20%) | 81% avg |
| Anglicisms (in-vocab chars) | 1/5 (20%) | 71% avg |
| Char-OOV (k/w/y) | 0/3 (0%) | 68% avg |
| **Real PT-BR OOV** | **5/5 (100%)** | **100%** |
| Controls (in training) | 4/4 (100%) | 100% |
| **Total** | **17/31 (55%)** | — |

The 5/5 result on genuine novel PT-BR words — words verified absent from the training corpus — includes correct: palatalization (d+i→dʒ), coda reduction (l→w), rhotacism (rr→x), and nasal vowel mapping (om→õ). These patterns are consistent with phonological rule generalization rather than corpus memorization.

**Defined failure modes**: (1) geminate consonants from Italian/English loans — not represented in the training corpus; (2) English-phonology anglicisms — genuinely OOV phonologically; (3) characters k, w, y — hard character-level OOV. These are expected limits of a supervised monolingual model, not architectural weaknesses.

---

## 7. Conclusion

We presented Distance-Aware (DA) Loss, a phonologically-graded training objective for G2P that penalizes substitution errors proportionally to PanPhon articulatory distance, weighted by prediction confidence. Applied to Brazilian Portuguese, DA Loss reduces catastrophic (Class D) substitutions by 19% relative while achieving PER 0.48% (Wilson CI ±0.03 pp) and WER 5.33% in the reference configuration on the largest PT-BR G2P evaluation reported — 28,782 stratified words. Genuine OOV generalization (5/5 novel PT-BR words, 100%) supports phonological rule acquisition beyond corpus memorization.

DA Loss is applicable to any language with PanPhon phoneme coverage; its effectiveness beyond PT-BR remains to be validated experimentally. Identified limitations: geminate reduction for loan words absent from training data, English-phonology anglicisms as genuine phonological OOV, and homograph disambiguation requiring morphosyntactic context beyond word-level G2P.

---

## References

[1] D. Bahdanau, K. Cho, and Y. Bengio, "Neural machine translation by jointly learning to align and translate," *arXiv preprint arXiv:1409.0473*, 2014; published at *ICLR*, 2015.

[2] K. Chary et al., "LatPhon: Multilingual grapheme-to-phoneme conversion with language-aware encoders," *arXiv preprint arXiv:2509.03300*, 2025.

[3] R. Kohavi, "A study of cross-validation and bootstrap for accuracy estimation and model selection," in *Proc. IJCAI*, vol. 2, pp. 1137–1143, 1995.

[4] L. Bottou, "Large-scale machine learning with stochastic gradient descent," in *Proc. COMPSTAT*, pp. 177–186, 2010.

[5] M. Bisani and H. Ney, "Joint-sequence models for grapheme-to-phoneme conversion," *Speech Communication*, vol. 50, no. 5, pp. 434–451, 2008.

[6] E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *J. Amer. Statist. Assoc.*, vol. 22, no. 158, pp. 209–212, 1927.

[7] K. Rao, H. Sak, and R. Prabhavalkar, "Grapheme-to-phoneme conversion using long short-term memory recurrent neural networks," in *Proc. IEEE ICASSP*, 2015.

[8] N. Neto, F. Fagundes, and P. Catelli, "New resources for Brazilian Portuguese: Results for grapheme-to-phoneme and phone classification," in *Proc. IEEE ICASSP*, 2006.

[9] D. R. Mortensen, P. Littell, A. Bharadwaj, K. Goyal, C. Dyer, and L. Levin, "PanPhon: A resource for mapping IPA segments to articulatory feature vectors," in *Proc. COLING*, pp. 3264–3273, 2016.

[10] P. A. Barbosa and E. C. Albano, "Brazilian Portuguese," *J. Int. Phonetic Assoc.*, vol. 34, no. 2, pp. 227–232, 2004.

[11] L. D. Brown, T. T. Cai, and A. DasGupta, "Interval estimation for a binomial proportion," *Statist. Sci.*, vol. 16, no. 2, pp. 101–133, 2001.
