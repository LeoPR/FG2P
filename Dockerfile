# FG2P — Inference Docker Image
#
# Image name: fg2p (standard, always this)
# Container name: fg2p (use docker run --name fg2p or compose.yaml)
#
# Build:   docker build -t fg2p .
# Run:     docker run --rm fg2p python -m src.inference_minimal
# Compose: docker compose up

FROM python:3.13-slim

LABEL description="FG2P — G2P for Brazilian Portuguese (inference only)"
LABEL url="https://github.com/LeoPR/FG2P"

WORKDIR /app

# Install dependencies
COPY requirements_inference_only.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements_inference_only.txt

# Copy source code (minimal inference stack)
COPY src/inference_minimal.py src/
COPY src/inference_light.py src/
COPY src/g2p.py src/
COPY src/utils.py src/
COPY src/inference.py src/
COPY src/analyze_errors.py src/
COPY src/phoneme_embeddings.py src/
COPY src/file_registry.py src/
COPY src/phonetic_features.py src/

# Copy data
COPY dicts/pt-br.tsv dicts/

# Copy models (best_per → exp104d, best_wer → exp9)
COPY models/model_registry.json models/
COPY models/exp104d_structural_tokens_correct/exp104d_structural_tokens_correct__20260312_142940.pt models/exp104d_structural_tokens_correct/
COPY models/exp104d_structural_tokens_correct/exp104d_structural_tokens_correct__20260312_142940_metadata.json models/exp104d_structural_tokens_correct/
COPY models/exp9_intermediate_distance_aware/exp9_intermediate_distance_aware__20260310_193733.pt models/exp9_intermediate_distance_aware/
COPY models/exp9_intermediate_distance_aware/exp9_intermediate_distance_aware__20260310_193733_metadata.json models/exp9_intermediate_distance_aware/

# Test
RUN python -c "from src.inference_minimal import G2PPredictor; print('✓ Ready')"

CMD ["python", "-m", "src.inference_minimal"]
