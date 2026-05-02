FROM python:3.11-slim

# System dependencies: poppler for pdf2image, curl for healthchecks
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies as a cached layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
