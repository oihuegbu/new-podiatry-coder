FROM python:3.11-slim AS app

# System dependencies: poppler for pdf2image, curl for healthchecks
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies as a cached layer.
# torch CPU-only (~220 MB vs 532 MB for the default CUDA wheel) is installed
# first from the PyTorch index; remaining deps come from PyPI.
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

# Pre-download FastEmbed models so the first pipeline run doesn't need internet
RUN python -c "\
from fastembed import TextEmbedding, SparseTextEmbedding; \
print('Downloading BAAI/bge-base-en-v1.5 ...'); TextEmbedding('BAAI/bge-base-en-v1.5'); \
print('Downloading Qdrant/bm25 ...'); SparseTextEmbedding('Qdrant/bm25'); \
print('Models cached')"

# Copy application code and bundled data (code JSON files, global tables)
COPY . .

# Output and log dirs will typically be bind-mounted from the host
RUN mkdir -p output/results logs data/result_cache

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]


# --- test stage ---------------------------------------------------------------
# The deployed image is the `app` stage above; it deliberately carries only
# production deps (requirements.txt) and no test framework. This stage extends
# it with the dev-only test deps (requirements-dev.txt -> pytest) so the suite
# runs from a CLEAN, reproducible build instead of a hand-`pip install`ed pytest
# inside a running container (which lives only in that container's writable layer
# and vanishes on recreate). Built explicitly and never deployed:
#     docker build --target test -t podiatry-coder-test .
#     docker run --rm podiatry-coder-test            # -> python -m pytest -q
# Every deploy path uses `docker compose build app` (compose pins target: app),
# so this stage never reaches production even though it is the last stage here.
FROM app AS test
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements-dev.txt
CMD ["python", "-m", "pytest", "-q"]
