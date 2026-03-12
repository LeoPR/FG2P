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

# Copy models (best_per, best_wer)
COPY models/model_registry.json models/
COPY models/exp104b_intermediate_sep_da_custom_dist_fixed/exp104b_intermediate_sep_da_custom_dist_fixed__20260225_045333.pt models/exp104b_intermediate_sep_da_custom_dist_fixed/
COPY models/exp104b_intermediate_sep_da_custom_dist_fixed/exp104b_intermediate_sep_da_custom_dist_fixed__20260225_045333_metadata.json models/exp104b_intermediate_sep_da_custom_dist_fixed/
COPY models/exp9_intermediate_distance_aware/exp9_intermediate_distance_aware__20260222_064838.pt models/exp9_intermediate_distance_aware/
COPY models/exp9_intermediate_distance_aware/exp9_intermediate_distance_aware__20260222_064838_metadata.json models/exp9_intermediate_distance_aware/

# Test
RUN python -c "from src.inference_minimal import G2PPredictor; print('✓ Ready')"

CMD ["python", "-m", "src.inference_minimal"]
