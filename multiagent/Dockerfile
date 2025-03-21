FROM python:3.11-slim

EXPOSE 8000
EXPOSE 5678  

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip requirements
WORKDIR /app
COPY requirements.txt .

# Upgrade pip and install requirements with additional options
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       --upgrade \
       --use-pep517 \
       --prefer-binary \
       "docarray==0.21.0" \
       -r requirements.txt

# Install debugpy
RUN pip install --no-cache-dir debugpy

# Copy the rest of the application
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Start application
CMD ["python", "-m", "debugpy", "--wait-for-client", "--listen", "0.0.0.0:5678", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]