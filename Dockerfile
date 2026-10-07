FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (curl + required C libraries for OpenCV/ONNX image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and model
COPY . .

# Render defaults to port 10000
EXPOSE 10000

# Run using shell execution so Render's dynamic ${PORT} is properly evaluated
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]