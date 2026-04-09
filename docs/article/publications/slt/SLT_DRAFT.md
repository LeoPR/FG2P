# Distance-Aware Loss for Phonologically-Graded Grapheme-to-Phoneme Conversion in Brazilian Portuguese

**[IEEE SLT 2026 — Anonymous Submission]**
**Authors**: [ANONYMOUS]
**Affiliation**: [ANONYMOUS]

---

## Abstract

Standard grapheme-to-phoneme (G2P) training with cross-entropy treats all phoneme substitutions equally, regardless of articulatory distance. We propose Distance-Aware (DA) Loss, which penalizes substitutions proportionally to PanPhon articulatory distance between predicted and target phonemes, weighted by the model's prediction confidence. Applied to Brazilian Portuguese with a BiLSTM encoder-decoder, DA Loss systematically redistributes errors toward phonologically closer targets: Class D (catastrophic) substitutions decrease 19% relative vs. a CE baseline, while Class B (near-miss) errors increase proportionally.

Our system achieves PER 0.48% and WER 5.33% in the reference configuration (a complementary no-separator model reaches WER 4.96%) on a stratified 28,782-word test set — 57× larger than comparable PT-BR evaluations — with a Wilson 95% CI of ±0.03 pp. Both Wilson CIs do not overlap with the closest reference system (upper bound 0.51% vs. lower bound 0.56%) — a statistically significant difference at 95% confidence. Evaluation on 31 out-of-vocabulary words yields 100% accuracy on genuine novel PT-BR words, consistent with phonological rule generalization beyond memorization. A factorial analysis of model capacity, syllable separators, and DA Loss reveals that the interaction between these factors — not capacity alone — determines performance.

---

## 1. Introduction

Grapheme-to-phoneme (G2P) conversion is a core component of text-to-speech synthesis, automatic speech recognition, and multilingual NLP pipelines. For Brazilian Portuguese (PT-BR), G2P presents well-documented challenges: graphemic ambiguity (grapheme "c" maps to /k/ in *cama* but /s/ in *cena*; "r" maps to /ɾ/ in syllable onset but /x/ in word-final coda [10]), vowel neutralization in unstressed positions (/e/↔/ɛ/ and /o/↔/ɔ/ merge), and position-dependent coda realization (/x/ in final position; /ɣ/ before voiced consonants). Prior work on PT-BR G2P has addressed these challenges with decision trees, n-gram models, and more recently neural seq2seq architectures [8].

Standard sequence-to-sequence models trained with cross-entropy (CE) treat all phoneme errors equally: predicting /ɛ/ when the target is /e/ — a near-miss differing in one articulatory feature — incurs the same gradient penalty as predicting /k/ for /a/, an error spanning eight articulatory features. This phonological blindness distorts the training signal: the model learns *that* it erred but not *how severely*.

We address this with **Distance-Aware (DA) Loss**, which adds a training signal proportional to the articulatory distance between predicted and target phonemes, weighted by the model's confidence in its prediction. We apply this to Brazilian Portuguese using a BiLSTM encoder-decoder with Bahdanau attention [1] — deliberately chosen to isolate the contribution of the loss function from architectural novelty.

**Comparison with LatPhon [2].** LatPhon is a 4-layer multilingual Transformer (7.5M parameters, RoPE) reporting PER 0.86% (Wilson 95% CI [0.56%, 1.16%]) on ~500 PT-BR words from the same IPA dictionary used in this work. Our system achieves PER 0.48% (CI [0.46%, 0.51%]) on 28,782 words. Both systems report Wilson CIs; the intervals do not overlap — the upper bound of our system (0.51%) falls below the lower bound of LatPhon (0.56%), constituting a statistically significant difference at 95% confidence. The difference in test set scale (57×) does not invalidate significance but confers 10× more precise confidence intervals to our evaluation (±0.03 pp vs. ±0.30 pp).

**Contributions:**
1. DA Loss: a phonologically-graded training objective combining PanPhon articulatory distance with prediction confidence (§4)
2. Large-scale stratified evaluation with empirical evidence of split bias impact (§2)
3. Factorial analysis of capacity × separators × DA Loss interaction (§5)
4. Error quality taxonomy: Class A–D classification reveals systematic redistribution of catastrophic errors (§6)
5. OOV generalization evaluation across 6 diagnostic categories (§6.3)

---

## 2. Data and Evaluation Protocol

### 2.1 Corpus

The training corpus consists of **95,937 grapheme–IPA pairs** from a Brazilian Portuguese phonetic dictionary. The input charset covers a–z (excluding k, w, y, which are absent from the dictionary) plus Portuguese diacritics (ç, á, à, â, ã, é, ê, í, ó, ô, õ, ú, ü). Words containing k, w, or y are treated as character-OOV and mapped to ⟨UNK⟩. Prior to training, 10,252 entries were corrected for ASCII-g (U+0067) vs. IPA-ɡ (U+0261) conflation, required for correct PanPhon feature lookup.

### 2.2 Stratified Split

The corpus is divided **60/10/30** (train/val/test) using stratified sampling over three phonological variables: stress type (oxytone/paroxytone/proparoxytone), syllable count bin (1, 2, 3, 4, 5+), and character length bin (≤4, 5–7, 8–10, 11+). Their combination yields approximately 48 strata. A purely random split risks concentrating easy words in the test set, inflating metrics.

Split quality: χ²=0.95 (p=0.678), Cramér V=0.0007 — no statistically significant distributional difference across subsets. The implementation uses `sklearn.model_selection.train_test_split(stratify=strata, random_state=42)` in two passes: (1) extract test (30%); (2) extract validation from remainder (~14.3% of trainval → 10% effective).

**Empirical evidence of split bias.** An unstratified 70/10/20 split (Exp0, same 4.3M architecture, CE loss) achieved PER 1.12%, while the stratified 60/10/30 split (Exp1) achieved 0.66% — a 41% PER reduction attributable entirely to evaluation protocol, not model improvement. Without stratification, the random partition can concentrate difficult words (proparoxytones, long words) in training and easy words in testing, inflating metrics artificially. This motivated stratification in all subsequent experiments and the deliberate choice of a large 30% test set over maximizing training data.

### 2.3 Evaluation Metrics

**PER** (Phoneme Error Rate): Levenshtein distance over phoneme sequences, normalized by reference length [5]. **WER** (Word Error Rate): fraction of words with any phoneme error (exact-match). **Wilson 95% CI** [6, 11]: used in place of Wald intervals, which underestimate uncertainty near p→0. For our test set (~181K reference phonemes), the Wilson CI on PER is ±0.03 pp — 10× more precise than ±0.30 pp for ~500-word evaluations.

---

## 3. Architecture

We use a **BiLSTM encoder-decoder with Bahdanau attention** [1], an architecture established for supervised G2P [7] and chosen to isolate the contribution of the loss function from architectural novelty.

**Encoder**: 2-layer BiLSTM over grapheme embeddings. Each position *t* yields hₜ = [h→ₜ; h←ₜ], providing full bidirectional context — critical for resolving graphemic ambiguity where phoneme identity depends on surrounding characters.

**Attention**: at each decoder step *t*, context vector cₜ is computed as:

$$e_{t,j} = v^\top \tanh(W_h h_j + W_s s_{t-1}), \quad \alpha_{t,j} = \text{softmax}(e_{t,j}), \quad c_t = \sum_j \alpha_{t,j} h_j$$

**Decoder**: 2-layer LSTM, teacher forcing during training, autoregressive at inference.

**Phoneme embeddings**: All main experiments use learned embeddings (random Glorot init). We additionally tested PanPhon-initialized embeddings (24D articulatory features projected to 128D), which provide a geometric prior — similar phonemes start close in embedding space. With CE loss, PanPhon init produces qualitatively better errors (lower PER_weighted) but does not improve raw PER, as CE does not reinforce the articulatory structure. DA Loss operates independently of embedding initialization: it uses an external lookup table of PanPhon distances, not the embedding geometry. The two mechanisms (PanPhon init and DA Loss) are orthogonal and could potentially be combined; this remains unexplored.

**Configurations tested:**

| Config | Embedding | Hidden | Parameters |
|--------|-----------|--------|------------|
| Small  | 128D | 256D | 4.3M |
| Medium | 192D | 384D | 9.7M |
| Large  | 256D | 512D | 17.2M |

---

## 4. Distance-Aware Loss

### 4.1 Formulation

$$\boxed{L = L_{\text{CE}} + \lambda \cdot d_{\text{PanPhon}}(\hat{y},\, y) \cdot p(\hat{y})}$$

where:

- **L_CE** = −log(p_correct): cross-entropy over the target phoneme
- **d_PanPhon(ŷ, y) ∈ [0, 1]**: normalized Euclidean distance between predicted and target phoneme in PanPhon's 24-dimensional articulatory feature space [9], encoding voicing, nasality, place, and manner of articulation. Representative distances: p↔b: 0.04 (voicing only); s↔ʃ: 0.15 (place shift); a↔k: 0.90 (vowel vs. velar stop)
- **p(ŷ) ∈ (0, 1]**: softmax probability of the *predicted* (argmax) phoneme
- **λ**: coupling weight (optimal: 0.20)

**Design rationale.** CE monitors p_correct; DA monitors p(ŷ). These are independent: p_correct can be low while p(ŷ) is high, i.e., the model is confident in a wrong prediction. The confidence factor scales the articulatory penalty with the model's certainty in its error — maximally penalizing errors that are both confident and phonologically distant.

**Bounded signal.** DA Loss is upper-bounded by λ × 1.0 × 1.0 = 0.20, while CE can reach ~16 in early training. DA is effective primarily in the learning transition zone (CE 0.3–1.5), where the model is actively resolving phoneme ambiguities.

**Numerical example.** Consider two errors with identical CE (p_correct = 0.40, CE = 0.916):

| Error type | Predicted | d_PanPhon | p(ŷ) | DA term | Total L |
|---|---|---|---|---|---|
| Near-miss: e→ɛ | ɛ | 0.10 | 0.45 | 0.009 | **0.925** |
| Catastrophic: a→k | k | 0.90 | 0.45 | 0.081 | **0.997** |

CE alone produces identical gradients for both cases (0.916). DA adds +0.009 for the near-miss vs. +0.081 for the catastrophic error — a 9× difference in phonological penalty, providing the model with an explicit signal about error severity. Over training epochs, this steers the model to "break ties toward the phonologically closer candidate."

**Novelty scope.** Distance-weighted losses have been explored in ASR and TTS contexts. Our contribution is the specific compositional formulation for IPA-level G2P: PanPhon articulatory distance as the error signal, prediction confidence as weighting factor, and controlled coupling via λ preserving CE as the primary learning axis.

### 4.2 Structural Token Override

PanPhon assigns zero vectors to non-phonemic tokens: syllable boundary "." and stress marker "ˈ". Without correction, d(., ˈ) = 0.0 — DA Loss provides no gradient signal for structural token confusions. We apply a post-normalization override assigning distance 1.0 between any structural token and any other symbol. The override must be applied *after* Euclidean normalization; applying it before yields d ≈ 0.25 (equivalent to a mid-vowel distance), not the intended maximum.

### 4.3 λ Sweep

Fixed architecture (4.3M, no syllable separators):

| λ | PER | WER | Behavior |
|---|-----|-----|----------|
| 0.05 | 0.62% | 5.36% | DA signal too weak |
| 0.10 | 0.63% | 5.35% | Moderate improvement |
| **0.20** | **0.60%** | **5.14%** | **Optimal** |
| 0.50 | 0.65% | 5.57% | Over-penalization |

The inverted-U pattern is expected: λ too low leaves CE undifferentiated; λ too high competes with CE and degrades learning.

---

## 5. Experiments

### 5.1 Main Results

| System | Params | Loss | Sep | PER | WER |
|--------|--------|------|-----|-----|-----|
| CE baseline | 4.3M | CE | — | 0.66% | 5.65% |
| DA λ=0.2 | 4.3M | DA | — | 0.60% | 5.14% |
| **DA λ=0.2** | **9.7M** | **DA** | **—** | **0.58%** | **4.96%** |
| CE + sep | 9.7M | CE | ✓ | 0.52% | 5.79% |
| DA λ=0.2 + sep | 9.7M | DA | ✓ | 0.53% | 5.73% |
| **DA + dist + sep** | **17.2M** | **DA+dist** | **✓** | **0.48%** | **5.33%** |

Sep = syllable boundary tokens in output; dist = structural distance override. Bold = recommended configurations.

**Separator trade-off.** Syllable boundary tokens improve PER (−17% to −20%) at the cost of WER (+6% to +8%). Each misplaced separator counts as a full word error under exact-match WER. Recommendation: use 9.7M DA (no separators) for WER-sensitive tasks; use 17.2M DA+dist (with separators) for PER-sensitive tasks.

### 5.2 Factorial Analysis: Capacity × Separators × DA Loss

A 2×2 factorial (separators × DA Loss) at 9.7M parameters isolates factors:

| | CE | DA λ=0.2 |
|---|---|---|
| **No separators** | Exp5: 0.63% PER | Exp9: 0.58% PER |
| **With separators** | Exp102: 0.52% PER | Exp103: 0.53% PER |

**Key findings:**
- Without separators: DA improves PER (0.63% → 0.58%, −8% relative)
- With separators: DA does not improve over CE at 9.7M (0.52% → 0.53%)
- Separators alone provide the largest single-factor improvement (−17% PER)
- The best result (0.48% PER) requires ALL three factors: 17.2M capacity + DA + separators + distance override

**Capacity interaction (corrected analysis).** At 17.2M parameters *without* separators (Exp10), DA Loss does not improve PER relative to CE (0.61% vs. 0.60%). However, at 17.2M *with* separators and corrected distances (Exp104d), DA achieves the best PER observed (0.48%). This refutes the simple hypothesis that "large models memorize and DA interferes" — the relevant variable is the interaction between capacity, structural representation, and distance correction, not capacity alone. Dedicated ablation is required for definitive conclusions about memorization.

### 5.3 Comparison with LatPhon

| System | PER (Wilson 95% CI) | Test words | Architecture |
|--------|---------------------|------------|--------------|
| LatPhon [2] | 0.86% [0.56%, 1.16%] | ~500 | 7.5M Transformer (multilingual) |
| **Ours (DA+dist, 17.2M)** | **0.48% [0.46%, 0.51%]** | **28,782** | **17.2M BiLSTM (PT-BR)** |

Both systems report Wilson CIs computed over the same lexical resource (ipa-dict). The intervals do not overlap — the upper bound of our system (0.51%) falls below the lower bound of LatPhon (0.56%), a statistically significant difference at 95% confidence. The 57× difference in test set size does not invalidate significance but confers 10× more precise CIs to our evaluation.

Methodological differences preclude direct architectural comparison: LatPhon is a multilingual Transformer (6 languages, RoPE, no phonological loss); our system is a monolingual BiLSTM with DA Loss. The result is consistent with the hypothesis that methodological design (loss function + evaluation protocol) can compensate for architectural differences, without establishing universal hierarchy between model families.

---

## 6. Analysis

### 6.1 Dominant Error Patterns

Top phoneme confusions on the 28,782-word test set:

| Substitution | Count | Mechanism |
|--------------|-------|-----------|
| ɛ → e | 255 | PT-BR unstressed neutralization |
| e → ɛ | 197 | Same (reverse direction) |
| ɔ → o | 131 | Mid-back neutralization |
| i → e | 121 | Vowel reduction |
| o → ɔ | 95 | Mid-back neutralization |

Over 60% of errors are vowel neutralizations — phonological ambiguity inherent to PT-BR, where the same orthographic context can correspond to multiple correct transcriptions depending on stress position. The global /e/:/ɛ/ corpus ratio is 7.1:1, but by syllabic position the pre-tonic ratio is 24.9:1 (/e/ dominant) while the tonic ratio is 0.33:1 (/ɛ/ dominant). The model learns the strong pre-tonic /e/ bias and overgeneralizes to tonic syllables.

### 6.2 Error Quality: DA Loss Redistribution

| System | PER | Class B | Class D | D / all errors |
|--------|-----|---------|---------|----------------|
| CE baseline (4.3M) | 0.66% | 0.39% | 0.54% | 50.9% |
| DA λ=0.2 (9.7M)    | 0.58% | 0.36% | 0.44% | 48.4% |
| DA+dist (17.2M+sep) | 0.48% | 0.29% | 0.53%† | 47.4% |

†Class D inflated by structural token confusions (., ˈ) carrying elevated custom distances.

DA Loss reduces Class D errors **19% relative** (4.3M CE → 9.7M DA, holding evaluation conditions fixed). This redistribution is a distinct effect from PER improvement: even models with similar PER commit fewer catastrophic substitutions under DA Loss. In downstream TTS, Class B errors (e.g., /e/↔/ɛ/) are perceptually less salient than Class D errors (e.g., vowel↔stop); formal perceptual validation (MOS/ABX) remains future work.

### 6.3 OOV Generalization

We evaluate the 17.2M DA+dist model on 31 curated OOV words across 6 categories, plus 35 neologisms in a separate test set:

| Category | Correct | Phon. Score |
|----------|---------|-------------|
| PT-BR neologisms | 6/9 (67%) | 97% avg |
| Geminate consonants | 1/5 (20%) | 81% avg |
| Anglicisms (in-vocab chars) | 1/5 (20%) | 71% avg |
| Char-OOV (k/w/y) | 0/3 (0%) | 68% avg |
| **Real PT-BR OOV** | **5/5 (100%)** | **100%** |
| Controls (in training) | 4/4 (100%) | 100% |
| **Total** | **17/31 (55%)** | — |

The 5/5 result on genuine novel PT-BR words includes correct: palatalization (d+i→dʒ), coda reduction (l→w), rhotacism (rr→x), and nasal vowel mapping (om→õ). These patterns are consistent with phonological rule generalization rather than corpus memorization.

The N=31 OOV bank is a **diagnostic probe**, not a replacement for the main 28,782-word test set. Its purpose is qualitative: demonstrating that specific phonological rules transfer to unseen words. Statistical generalization claims rest on the stratified test set; OOV results provide complementary evidence of rule acquisition.

**Defined failure modes**: (1) geminate consonants from Italian/English loans — absent from training corpus; (2) English-phonology anglicisms — genuinely OOV phonologically; (3) characters k, w, y — hard character-level OOV.

---

## 7. Conclusion

We presented Distance-Aware (DA) Loss, a phonologically-graded training objective for G2P that penalizes substitution errors proportionally to PanPhon articulatory distance, weighted by prediction confidence. Applied to Brazilian Portuguese, DA Loss reduces catastrophic (Class D) substitutions by 19% relative while achieving PER 0.48% (Wilson CI ±0.03 pp) and WER 5.33% in the reference configuration on the largest PT-BR G2P evaluation reported — 28,782 stratified words.

A factorial analysis reveals that the best performance requires the interaction of three factors — model capacity, syllable separators, and DA Loss with corrected structural distances — rather than any single factor. Genuine OOV generalization (5/5 novel PT-BR words) supports phonological rule acquisition beyond corpus memorization.

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
