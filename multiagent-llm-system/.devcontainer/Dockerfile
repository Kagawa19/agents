# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    software-properties-common \
    ca-certificates \
    gnupg \
    postgresql-client \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

# Install Python development tools and linters
RUN pip install --no-cache-dir \
    black \
    pylint \
    flake8 \
    mypy \
    pytest \
    pre-commit \
    poetry \
    docker-compose

# Create a non-root user
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid vscode -m vscode

# Set up permissions
RUN mkdir -p /workspace \
    && chown -R vscode:vscode /workspace

# Switch to non-root user
USER vscode

# Default command
CMD ["/bin/bash"]