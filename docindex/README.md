# DocIndex — Open-Source PageIndex Alternative

Reasoning-based document intelligence using Gemini Flash. No vector databases, no chunking, no expensive infra.

## How It Works

Instead of embedding your document into vectors and doing similarity search (which loses context and structure), DocIndex:

1. **Parses** your document into a hierarchical tree index (like a table of contents), stored as plain JSON
2. **Navigates** — when you ask a question, Gemini reads the outline and reasons about which sections are relevant
3. **Answers** — Gemini reads those specific sections and generates a precise, cited answer

This is exactly how PageIndex works, at a fraction of the cost.

## Cost Comparison

| Service | Cost per 1M tokens |
|---|---|
| PageIndex API | ~$20–$50 (estimated) |
| **Gemini 2.0 Flash (this)** | **$0.075 input / $0.30 output** |
| Gemini 1.5 Flash | $0.075 input / $0.30 output |

**For a 100-page financial report, a full Q&A session costs ~$0.01–$0.05.**
Your $1000 GCP credits = ~3–10 million document queries.

---

## Quick Start

### 1. Install system dependencies (for OCR)

**Ubuntu / Debian (including GCP VMs):**
```bash
sudo apt install poppler-utils tesseract-ocr tesseract-ocr-eng
```

**macOS:**
```bash
brew install poppler tesseract
```

**Windows:**
- Poppler: https://github.com/oschwartz10612/poppler-windows/releases
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Add both to your PATH

**Need other languages?** (e.g. French, Arabic, Chinese)
```bash
# Ubuntu — install extra language packs
sudo apt install tesseract-ocr-fra tesseract-ocr-ara tesseract-ocr-chi-sim

# Then pass lang="fra" or lang="ara" to _ocr_page() in parser.py
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Gemini API key

Get a free key at https://aistudio.google.com/app/apikey

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 4. Run the API server

```bash
cd docindex
uvicorn api.main:app --reload --port 8000
```

### 5. Verify OCR is working

```bash
curl http://localhost:8000/ocr-status
# {"poppler": true, "tesseract": true, "ocr_available": true, "advice": []}
```

### 6. Open the Web UI

Open `ui/index.html` in your browser. Make sure the API URL says `http://localhost:8000`.

---

## API Usage

### Upload a document
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@report.pdf"
```

### Query a document
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "abc12345",
    "query": "What are the main revenue drivers?",
    "chat_history": []
  }'
```

### List documents
```bash
curl http://localhost:8000/documents
```

---

## Deploy to GCP Cloud Run (uses your credits)

```bash
# 1. Build and push container
gcloud builds submit --tag gcr.io/YOUR_PROJECT/docindex

# 2. Deploy to Cloud Run
gcloud run deploy docindex \
  --image gcr.io/YOUR_PROJECT/docindex \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_key_here \
  --memory 512Mi \
  --min-instances 0 \
  --max-instances 10
```

Cloud Run scales to zero when not in use — you only pay for actual requests.

---

## Project Structure

```
docindex/
├── core/
│   ├── parser.py        # Document → hierarchical tree index
│   └── retriever.py     # Reasoning-based retrieval with Gemini
├── api/
│   └── main.py          # FastAPI REST API
├── ui/
│   └── index.html       # Web chat interface (single file)
├── data/                # Created automatically
│   ├── uploads/         # Original documents
│   └── indexes/         # JSON tree indexes
├── requirements.txt
├── Dockerfile
└── .env.example
```

## Supported Document Types

- **PDF** — with TOC extraction for best results
- **DOCX/DOC** — uses heading styles for structure
- **TXT** — flat text, page-grouped

## Extending

- **Add more LLMs**: Edit `RETRIEVAL_MODEL` / `ANSWER_MODEL` in `core/retriever.py`
- **Use Vertex AI**: Replace `google-generativeai` with `vertexai` SDK for direct GCP billing
- **Add authentication**: Add FastAPI middleware or use GCP Identity-Aware Proxy
- **Persistent storage**: Replace local file storage with GCP Cloud Storage
- **Database**: Replace `documents.json` with Cloud Firestore or PostgreSQL