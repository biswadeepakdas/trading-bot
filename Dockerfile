FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copy project files
COPY config.py features.py lstm_model.py run_prediction.py ./
COPY models/ models/

# Cloud Run uses PORT env var
ENV PORT=8080

# Entry point
COPY cloud_run.py .
CMD ["python", "cloud_run.py"]
