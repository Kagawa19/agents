FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-dev \
    libpq-dev \
    curl \
    iputils-ping \
    net-tools \
    dnsutils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Expose ports
EXPOSE 8000
EXPOSE 5678

# Install pip requirements
COPY requirements.txt .

# Upgrade pip and install requirements with additional options
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       --upgrade \
       --use-pep517 \
       --prefer-binary \
       "docarray==0.21.0" \
       -r requirements.txt \
    && pip install --no-cache-dir debugpy

# Copy the rest of the application
COPY . /app

# Create a health check file
RUN echo "#!/bin/sh\npython -c 'import socket; socket.socket().connect((\"redis\", 6379))'" > /app/healthcheck.sh \
    && chmod +x /app/healthcheck.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD /app/healthcheck.sh

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Start application
CMD ["python", "-m", "debugpy", "--wait-for-client", "--listen", "0.0.0.0:5678", "-m", "uvicorn", "multiagent.app.main:app", "--host", "0.0.0.0", "--port", "8000"]