FROM python:3.11-slim

WORKDIR /app

# System dependencies:
#   libmupdf-dev   — PyMuPDF (native PDF parsing)
#   poppler-utils  — pdftoppm (renders PDF pages to images for OCR)
#   tesseract-ocr  — OCR engine
#   tesseract-ocr-eng — English language pack (add more as needed e.g. tesseract-ocr-fra)
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Optional: uncomment to enable Google Cloud Vision OCR fallback
# RUN pip install --no-cache-dir google-cloud-vision==3.7.2

COPY requirements.txt .
RUN pip install --no-cache-dir pandas==2.2.3
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories
RUN mkdir -p data/uploads data/indexes

EXPOSE 8080

# Cloud Run uses PORT env variable
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}