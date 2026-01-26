# Hugging Face Spaces compatible Dockerfile
FROM python:3.11-slim

# Set up a non-root user (required for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install system dependencies as root first
USER root
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER user

# Copy requirements first for caching
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy application code
COPY --chown=user . .

# Create cache directory
RUN mkdir -p .cache data models

# Expose HF Spaces required port
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Run with gunicorn on port 7860 (HF Spaces requirement)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120"]
