# Multi-stage Dockerfile for study package
# Supports both CPU and GPU environments with optimized builds

# Build stage
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency resolution
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY README.md LICENSE ./

# Build the package
RUN uv build

# Runtime stage
FROM python:3.12-slim as runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Install system dependencies for runtime
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy built package
COPY --from=builder /app/dist/*.whl /tmp/

# Install the package
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Copy source code (for development/debugging)
COPY --from=builder /app/src ./src

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose port for Jupyter/API services
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import study; print('Package loaded successfully')" || exit 1

# Default command
CMD ["python", "-c", "import study; print('study package is ready!')"]

# GPU variant
FROM runtime as gpu

# Switch back to root for CUDA installation
USER root

# Install CUDA runtime (if needed for GPU support)
# This assumes the base image doesn't have CUDA
RUN apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Install GPU-specific Python packages
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Switch back to app user
USER app

# CPU variant (default)
FROM runtime as cpu

# Install CPU-specific Python packages
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Development variant
FROM runtime as dev

# Switch back to root for development tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    git \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
COPY --from=builder /app/.venv /app/.venv-dev
RUN pip install jupyter jupyterlab ipywidgets

# Copy development files
COPY . .

# Install package in development mode
RUN pip install -e .

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose additional ports for development
EXPOSE 8888 8889

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
