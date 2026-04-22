# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install all Python dependencies in an isolated layer so the final image
# is as small as possible — we only copy the installed packages across.
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build tools needed by some Python packages (scipy, scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first — Docker caches this layer unless requirements change
COPY requirements.txt .

# Install all packages into a custom prefix so we can copy them cleanly
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Install FastAPI extras (not in requirements.txt by default)
RUN pip install --no-cache-dir --prefix=/install \
    fastapi \
    uvicorn[standard] \
    python-multipart


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy the entire project
COPY . .

# Create all required runtime directories
RUN mkdir -p \
    uploads \
    templates \
    artifacts/raw_data \
    artifacts/processed_data \
    artifacts/models \
    artifacts/reports \
    logs \
    mlruns

# ── Hugging Face Spaces requirement ───────────────────────────────────────────
# HF Spaces runs containers as a non-root user (UID 1000) on port 7860.
# We create the user, fix ownership, and expose the right port.
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

# HF Spaces always uses port 7860 — do not change this
EXPOSE 7860

# Environment variables
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=mlruns

# Health check — HF Spaces polls this to know the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start the FastAPI server
CMD ["python", "app.py"]
