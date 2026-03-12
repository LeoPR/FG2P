# FG2P Quick Start

Get FG2P running in 2 minutes.

## Option 1: Docker (Recommended)

```bash
# Build image
docker build -t fg2p .

# Run inference on test words
docker run --rm fg2p python -m src.inference_minimal
```

Output:
```
computador           → k õ . p u . t a . ˈ d o x
português            → p o x . t u . ˈ ɡ e s
inteligência         → ĩ . t e . l i . ˈ ɡ ẽ . s i . ə
```

### Docker Compose (interactive)

```bash
docker compose build
docker compose run --rm inference /bin/bash
python -m src.inference_minimal
```

## Option 2: Local Python

```bash
# Install dependencies
pip install -r requirements_inference_only.txt

# Run
python src/inference_minimal.py
```

## Next Steps

- **Integration**: See [docs/INTEGRATION.md](docs/INTEGRATION.md) for Python API and advanced usage
- **Models**: `G2PPredictor.load(alias='best_per')` or `'best_wer'`
- **Documentation**: [README.md](README.md) for full project overview

---

**Image size**: ~1.5GB (includes 2 models + dictionary)
**No volumes needed** — everything self-contained in container
