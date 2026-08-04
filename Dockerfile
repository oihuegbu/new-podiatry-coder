FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY medical_coder ./medical_coder
RUN pip install --no-cache-dir .
COPY source-packs ./source-packs
COPY data ./data
ENTRYPOINT ["medical-coder"]
CMD ["--help"]

