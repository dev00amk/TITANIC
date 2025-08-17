# Titanic Production CatBoost Pipeline Docker Image
# Multi-stage build for production-ready container

# Build stage
FROM python:3.9-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .

# Production stage
FROM python:3.9-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH"

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY Makefile ./
COPY pyproject.toml README.md ./

# Create necessary directories
RUN mkdir -p data artifacts logs reports

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import pandas, numpy, catboost, sklearn; print('Health check passed')" || exit 1

# Default command
CMD ["python", "-m", "src.modeling.train_cv"]

# Development stage (for development with volume mounts)
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN pip install jupyter notebook ipython pytest black isort ruff mypy

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Expose ports for development
EXPOSE 8888 5000 6006

# Switch back to appuser
USER appuser

# Development command
CMD ["bash"]

# Labels for metadata
LABEL maintainer="ML Team" \
      version="1.0" \
      description="Titanic Survival Prediction with CatBoost" \
      org.opencontainers.image.title="Titanic CatBoost Pipeline" \
      org.opencontainers.image.description="Production-ready machine learning pipeline for Titanic survival prediction" \
      org.opencontainers.image.vendor="ML Engineering Team" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/username/titanic-pro-catboost"