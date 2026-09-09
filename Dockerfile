# Production Dockerfile for Integrated Plant Suite
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for compiling packages / image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Healthcheck to monitor app status
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit bound to 0.0.0.0 for external & mobile access
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
