FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (optional): add build-essential if compiling libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# Copy application code
COPY src /app/src

# Expose port
EXPOSE 5000

# Default command to run the API (production server)
WORKDIR /app/src/api
ENV PORT=5000
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "--workers", "2", "flask_api:app"]
