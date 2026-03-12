# FG2P Integration Guide

Complete guide to integrating FG2P into your applications and workflows.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Python API Reference](#python-api-reference)
4. [Command-Line Interface](#command-line-interface)
5. [Integration Examples](#integration-examples)
6. [Fine-Tuning & Training](#fine-tuning--training)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Option 1: From Source (Development)

```bash
# Clone repository
git clone https://github.com/LeoPR/FG2P.git
cd fg2p

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Verify installation
python -c "from src.inference_light import G2PPredictor; print('✓ FG2P installed')"
```

### Option 2: As a Package (Future)

```bash
# Install from PyPI (when available)
pip install fg2p-br

# Or from GitHub
pip install git+https://github.com/LeoPR/FG2P.git
```

### Requirements

- Python 3.10 or higher
- PyTorch 2.0+ (CPU or CUDA)
- NumPy, Pandas, SciPy
- PanPhon (for phonetic features)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Basic Usage

### Quickest Start (3 lines)

```python
from src.inference_light import G2PPredictor

predictor = G2PPredictor.load(index=18)  # Load SOTA model
phonemes = predictor.predict("computador")
print(phonemes)  # Output: k õ . p u . t a . ˈ d o x
```

### With Error Handling

```python
from src.inference_light import G2PPredictor
from pathlib import Path

try:
    predictor = G2PPredictor.load(index=18)

    # Batch prediction
    words = ["python", "computador", "inteligência"]
    results = []
    for word in words:
        try:
            phonemes = predictor.predict(word)
            results.append((word, phonemes))
        except ValueError as e:
            print(f"Error predicting '{word}': {e}")
            results.append((word, None))

    # Print results
    for word, phonemes in results:
        if phonemes:
            print(f"{word:20} → {phonemes}")
        else:
            print(f"{word:20} → [ERROR]")

except Exception as e:
    print(f"Failed to load model: {e}")
```

---

## Python API Reference

### G2PPredictor Class

The main user-facing class for predictions.

#### Loading Models

```python
from src.inference_light import G2PPredictor

# Load by index (recommended)
predictor = G2PPredictor.load(index=18)  # Exp104b - SOTA PER

# Load by experiment name
predictor = G2PPredictor.load(name="exp104b")

# Load latest model
predictor = G2PPredictor.load()  # Loads most recent

# List available models
models = G2PPredictor.list_available()
for model in models:
    print(f"{model['index']:2d}: {model['name']:30s} PER={model.get('per', 'N/A')}")
```

#### Core Methods

##### `predict(word: str) → str`
Predict phonemes for a single word.

```python
phonemes = predictor.predict("computador")
# Returns: k õ . p u . t a . ˈ d o x

# Optional: get confidence scores
phonemes, scores = predictor.predict("casa", return_scores=True)
```

**Parameters:**
- `word` (str): Input word in Portuguese (lowercase recommended)

**Returns:**
- `str`: Space-separated IPA phonemes
- Optional: `(str, list)` of confidence scores per phoneme if `return_scores=True`

**Raises:**
- `ValueError`: If word is empty or contains only spaces

---

##### `predict_batch(words: list[str]) → list[str]`
Predict phonemes for multiple words efficiently.

```python
words = ["computador", "python", "inteligência"]
results = predictor.predict_batch(words)
# Returns: ["k õ . p u . t a . ˈ d o x", "p y . t ɔ̃", ...]
```

**Parameters:**
- `words` (list[str]): List of words to predict

**Returns:**
- `list[str]`: List of phoneme strings, same length as input

---

##### `evaluate_tsv(filepath: str, cache_tag: str = None) → dict`
Evaluate model on TSV file with phoneme references.

```python
# TSV format: word \t phonemes
# Example:
# computador	k õ . p u . t a . ˈ d o x
# casa	ˈ k a . z ə


results = predictor.evaluate_tsv("test.tsv")
print(results)
# Output: {
#   'per': 0.0123,       # Phoneme Error Rate
#   'wer': 0.05,         # Word Error Rate
#   'accuracy': 0.95,    # Word-level accuracy
#   'n_test': 1000,      # Number of test words
#   'bootstrap_ci_per': (0.0100, 0.0150),  # 95% CI
# }
```

**Parameters:**
- `filepath` (str): Path to TSV file (word \t phonemes_reference)
- `cache_tag` (str, optional): Cache tag for repeated evaluations

**Returns:**
- `dict`: Dictionary with metrics (per, wer, accuracy, bootstrap confidence intervals)

---

##### `find_similar(word: str, n: int = 5, metric: str = 'phonetic') → list`
Find words with similar pronunciation.

```python
similar = predictor.find_similar("computador", n=10)
for word, distance in similar:
    print(f"{word:20s} (distance: {distance:.3f})")
```

**Parameters:**
- `word` (str): Query word
- `n` (int): Number of similar words to return (default: 5)
- `metric` (str): 'phonetic' or 'graphemic' (default: 'phonetic')

**Returns:**
- `list[(str, float)]`: List of (word, distance) tuples, sorted by distance

---

##### `evaluate_neologisms(filepath: str = None) → dict`
Evaluate on neologisms and out-of-vocabulary words.

```python
# Use default test set
results = predictor.evaluate_neologisms()

# Or use custom TSV
results = predictor.evaluate_neologisms("docs/data/neologisms_test.tsv")

# Print analysis
print(f"Correct: {results['correct']}/{results['total']}")
print(f"Categories:")
for cat, accuracy in results['by_category'].items():
    print(f"  {cat}: {accuracy:.0%}")
```

**Parameters:**
- `filepath` (str, optional): Path to TSV file (default: `docs/data/neologisms_test.tsv`)

**Returns:**
- `dict`: Analysis by category, with correct/total counts

---

### ModelMetadata Class

Access model configuration and statistics.

```python
from src.inference_light import G2PPredictor

predictor = G2PPredictor.load(index=18)
metadata = predictor.metadata

# Access properties
print(f"Model: {metadata.get('experiment_name')}")
print(f"Params: {metadata.get('total_params'):,}")
print(f"PER: {metadata.get('per', 'N/A')}")
print(f"WER: {metadata.get('wer', 'N/A')}")
print(f"Training time: {metadata.get('total_time_seconds')/3600:.1f}h")
```

---

## Command-Line Interface

### Basic Commands

#### 1. Single Word Prediction
```bash
python src/inference_light.py --word "computador"
python src/inference_light.py --index 18 --word "python"
```

#### 2. Interactive Mode
```bash
python src/inference_light.py --interactive
# Prompts for words, type 'exit' or 'quit' to quit
```

#### 3. List Available Models
```bash
python src/inference_light.py --list
```

#### 4. Evaluate on Custom Data
```bash
# TSV format: word \t phonemes
python src/inference_light.py --index 18 --tsv "my_test.tsv"
python src/inference_light.py --index 18 --evaluate "my_test.tsv"
```

#### 5. Evaluate Neologisms
```bash
python src/inference_light.py --index 18 --neologisms
python src/inference_light.py --index 18 --neologisms "docs/data/generalization_test.tsv"
```

#### 6. Find Similar Words
```bash
python src/inference_light.py --word "laeta" --similar
python src/inference_light.py --word "computador" --similar --similar-count 10
```

---

## Integration Examples

### Example 1: Text-to-Speech Pipeline

```python
from src.inference_light import G2PPredictor
import re

class TTSPreprocessor:
    def __init__(self, g2p_index=18):
        self.g2p = G2PPredictor.load(index=g2p_index)

    def text_to_phonemes(self, text: str) -> list:
        # Lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())

        # Get phonemes for each word
        phoneme_sequences = []
        for word in words:
            try:
                phonemes = self.g2p.predict(word)
                phoneme_sequences.append(phonemes.split())
            except ValueError:
                print(f"Warning: Could not process '{word}'")
                continue

        return phoneme_sequences

    def get_syllable_boundaries(self, phonemes: list) -> list:
        """Extract syllable structure from phonemes with separators"""
        syllables = []
        current_syllable = []

        for phoneme in phonemes:
            if phoneme == '.':  # Syllable separator
                if current_syllable:
                    syllables.append(current_syllable)
                    current_syllable = []
            else:
                current_syllable.append(phoneme)

        if current_syllable:
            syllables.append(current_syllable)

        return syllables

# Usage
preprocessor = TTSPreprocessor()
text = "O computador é uma máquina incrível"
phonemes = preprocessor.text_to_phonemes(text)
syllables = preprocessor.get_syllable_boundaries(
    preprocessor.g2p.predict("computador").split()
)

for i, syllable in enumerate(syllables, 1):
    print(f"Syllable {i}: {' '.join(syllable)}")
```

### Example 2: Pronunciation Dictionary Builder

```python
from src.inference_light import G2PPredictor
from pathlib import Path
import csv

def build_pronunciation_dictionary(words_file: str, output_file: str = "pronunciations.csv"):
    """Create pronunciation dictionary from word list"""

    g2p = G2PPredictor.load(index=18)

    with open(words_file, 'r', encoding='utf-8') as f_in:
        words = [word.strip().lower() for word in f_in.readlines()]

    with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['Word', 'IPA', 'Syllables'])

        for word in words:
            try:
                ipa = g2p.predict(word)
                syllables = ipa.replace(' .', '|').replace(' ˈ', '[stress]')
                writer.writerow([word, ipa, syllables])
            except ValueError:
                print(f"Warning: Skipping '{word}'")

# Usage
build_pronunciation_dictionary("word_list.txt", "pronunciations.csv")
```

### Example 3: Speech Recognition Post-Processing

```python
from src.inference_light import G2PPredictor
from difflib import SequenceMatcher

class SpeechRecognitionValidator:
    """Check ASR output against G2P predictions"""

    def __init__(self, g2p_index=18):
        self.g2p = G2PPredictor.load(index=g2p_index)

    def compare_pronunciations(self, word: str, asr_phonemes: str) -> dict:
        """Compare ASR output with G2P predictions"""

        g2p_phonemes = self.g2p.predict(word)

        # Normalize for comparison
        g2p_phones = g2p_phonemes.split()
        asr_phones = asr_phonemes.split()

        # Calculate similarity
        matcher = SequenceMatcher(None, g2p_phones, asr_phones)
        ratio = matcher.ratio()

        return {
            'word': word,
            'g2p': g2p_phonemes,
            'asr': asr_phonemes,
            'similarity': ratio,
            'is_correct': ratio > 0.85,  # Threshold
        }

# Usage
validator = SpeechRecognitionValidator()
result = validator.compare_pronunciations(
    "computador",
    "k õ p u t a d o x"  # Hypothetical ASR output
)
print(f"Match: {result['is_correct']} (similarity: {result['similarity']:.2%})")
```

---

## Fine-Tuning & Training

### Fine-Tune on New Data

```python
from src.g2p import G2PCorpus, G2PLSTMModel
from src.train import train_model
from pathlib import Path
import json

# 1. Prepare your data
# Format: TSV file with word \t phonemes columns

# 2. Load base model
from src.inference_light import G2PPredictor
predictor = G2PPredictor.load(index=18)  # Start from SOTA

# 3. Create corpus with your data
corpus = G2PCorpus("path/to/your/data.tsv")
train_split, val_split, test_split = corpus.create_split(
    train_ratio=0.7,
    val_ratio=0.15,
    stratify=True
)

# 4. Configure fine-tuning
config = {
    "training": {
        "batch_size": 32,
        "learning_rate": 0.0001,  # Lower for fine-tuning
        "epochs": 20,
        "early_stopping_patience": 3,
    },
    "model": {
        # Use same architecture as base model
        "emb_dim": 128,
        "hidden_dim": 256,
        "num_layers": 2,
    }
}

# 5. Train
# (Implementation depends on your training loop)
# See src/train.py for details
```

### Train from Scratch

See [docs/article/EXPERIMENTS.md](docs/article/EXPERIMENTS.md) for the full training procedure and hyperparameter choices for each experiment.

```bash
# Example training command
python src/train.py --config conf/config_custom.json
```

---

## Performance Tuning

### CPU vs GPU

```python
from src.inference_light import G2PPredictor

# GPU (faster)
predictor = G2PPredictor.load(index=18, device='cuda')
# Speed: ~400 words/second on A100

# CPU (slower but no GPU memory)
predictor = G2PPredictor.load(index=18, device='cpu')
# Speed: ~10 words/second on i7-12700K
```

### Batch Processing

For large-scale prediction, use `predict_batch`:

```python
words = ["word1", "word2", ..., "wordN"]

# Efficient batch processing
phonemes = predictor.predict_batch(words)

# NOT: [predictor.predict(w) for w in words]  # Much slower
```

### Caching

```python
# Cache predictions to avoid recomputation
cache = {}

def predict_with_cache(word: str) -> str:
    if word in cache:
        return cache[word]

    result = predictor.predict(word)
    cache[word] = result
    return result
```

---

## Troubleshooting

### Issue: Model fails to load

```python
# Check available models first
from src.inference_light import G2PPredictor
models = G2PPredictor.list_available()
print(f"Found {len(models)} models")

# Load with error handling
try:
    predictor = G2PPredictor.load(index=18)
except FileNotFoundError as e:
    print(f"Model not found: {e}")
    print("Download models from GitHub releases")
```

### Issue: Out of memory on GPU

```python
# Use CPU instead
predictor = G2PPredictor.load(index=18, device='cpu')

# Or reduce batch size
phonemes = predictor.predict_batch(words, batch_size=16)  # Default: 32
```

### Issue: Unexpected characters in output

```python
# Ensure input is lowercase and normalized
import unicodedata

def normalize_word(word: str) -> str:
    # Remove accents (optional)
    # nfd = unicodedata.normalize('NFD', word)
    # return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')

    return word.lower().strip()

word = normalize_word("ComputadoR")
phonemes = predictor.predict(word)
```

### Issue: Poor performance on specific words

```python
# Check if word is in vocabulary
from src.g2p import G2PCorpus

corpus = G2PCorpus("dicts/pt-br.tsv")
print(f"In vocab: {'computador' in corpus.words}")

# Evaluate generalization on similar words
similar = predictor.find_similar("computador", n=10)
print(f"Similar words: {[w for w, _ in similar]}")
```

---

## FAQ

**Q: Which model should I use for my application?**
A:
- TTS/Phonetic alignment → Exp104b (PER 0.49%)
- NLP/Search → Exp9 (WER 4.96%)
- Real-time/Embedded → Exp106 (2.58× faster, WER 6.12%)
- Balanced → Exp9

**Q: Can I use FG2P with other languages?**
A: Models are trained for Brazilian Portuguese. For other languages, you'd need to train new models on language-specific data.

**Q: What's the difference between PER and WER?**
A:
- **PER** = % of individual phonemes predicted incorrectly
- **WER** = % of words with at least one phoneme error

**Q: How do I handle abbreviations and acronyms?**
A: Expand them first. Example: "USP" → "universidade de são paulo"

**Q: Can I use this in production?**
A: Yes! The models are production-ready. Consider caching predictions and error handling.

---

## API Stability

The core API (`G2PPredictor.predict()` and `.load()`) is considered stable and will maintain backward compatibility.

For development/advanced use, see `src/g2p.py` for lower-level API details.

---

**Last Updated**: March 2026
**API Version**: 1.0.0 (stable)
