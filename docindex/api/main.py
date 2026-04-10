"""
DocIndex API - FastAPI backend
Endpoints: upload document, query document, list documents
"""

import asyncio
import concurrent.futures
import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.indexer import build_index
from core.parser import load_index, ocr_status, parse_document, save_index, tree_to_outline
from core.retriever import retrieve, retrieve_streaming

app = FastAPI(title="DocIndex API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage paths
UPLOAD_DIR = Path("data/uploads")
INDEX_DIR  = Path("data/indexes")
META_FILE  = Path("data/documents.json")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# FIX 5: lock for meta.json to prevent concurrent-upload race condition
_meta_lock = threading.Lock()

# Thread pool for CPU/IO-bound indexing work (keeps FastAPI event loop free)
_index_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def load_meta() -> dict:
    if META_FILE.exists():
        with open(META_FILE) as f:
            return json.load(f)
    return {}


def save_meta(meta: dict):
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)


def _sections_count(tree: dict) -> int:
    """FIX 2: count top-level sections supporting both 'children' and 'nodes' keys."""
    return len(tree.get("nodes") or tree.get("children") or [])


# ─── Models ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    document_id: str
    query: str
    chat_history: Optional[list] = []
    stream: Optional[bool] = False
    show_trace: Optional[bool] = False


class QueryResponse(BaseModel):
    answer: str
    sources: list
    retrieval_trace: Optional[list] = []


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "DocIndex API"}


@app.get("/ocr-status")
def get_ocr_status():
    """Check OCR dependencies are installed."""
    status = ocr_status()
    advice = []
    if not status["poppler"]:
        advice.append("Install Poppler: sudo apt install poppler-utils")
    if not status["tesseract"]:
        advice.append("Install Tesseract: sudo apt install tesseract-ocr tesseract-ocr-eng")
    return {**status, "advice": advice}


@app.get("/documents")
def list_documents():
    meta = load_meta()
    return {"documents": list(meta.values())}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a document.
    FIX 2: Indexing runs in a thread executor so it never blocks the event loop.
    FIX 4: BackgroundTasks removed — we return immediately after saving the file
           and the index is built synchronously in the thread (client waits, but
           the FastAPI event loop stays free for other requests).
    FIX 5: meta.json write is protected by a threading.Lock.
    """
    allowed = {".pdf", ".docx", ".doc", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {allowed}")

    doc_id      = str(uuid.uuid4())[:8]
    upload_path = UPLOAD_DIR / f"{doc_id}{ext}"
    index_path  = INDEX_DIR  / f"{doc_id}.json"

    # Save the uploaded bytes
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # FIX 2: run blocking indexer in thread pool — safe inside FastAPI async endpoint
    def _do_index():
        if ext == ".pdf":
            return build_index(str(upload_path), add_summary=True)
        return parse_document(str(upload_path))

    loop = asyncio.get_running_loop()
    try:
        tree = await loop.run_in_executor(_index_executor, _do_index)
        save_index(tree, str(index_path))
    except Exception as e:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Failed to index document: {e}")

    # FIX 5: lock meta write to prevent clobbering concurrent uploads
    with _meta_lock:
        meta = load_meta()
        meta[doc_id] = {
            "id":          doc_id,
            "filename":    file.filename,
            "type":        ext.lstrip("."),
            "title":       tree.get("title", file.filename),
            "total_pages": tree.get("total_pages"),
            "sections":    _sections_count(tree),   # FIX 2
            "index_path":  str(index_path),
            "upload_path": str(upload_path),
        }
        save_meta(meta)

    return {
        "document_id": doc_id,
        "title":       tree.get("title"),
        "sections":    _sections_count(tree),
        "total_pages": tree.get("total_pages"),
        "message":     "Document indexed successfully",
    }


@app.get("/documents/{document_id}")
def get_document(document_id: str):
    meta = load_meta()
    if document_id not in meta:
        raise HTTPException(404, "Document not found")
    doc_meta = meta[document_id]
    tree     = load_index(doc_meta["index_path"])
    outline  = tree_to_outline(tree)
    return {**doc_meta, "outline": outline}


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    with _meta_lock:
        meta = load_meta()
        if document_id not in meta:
            raise HTTPException(404, "Document not found")
        doc_meta = meta.pop(document_id)
        Path(doc_meta["index_path"]).unlink(missing_ok=True)
        Path(doc_meta["upload_path"]).unlink(missing_ok=True)
        save_meta(meta)
    return {"message": "Document deleted"}


@app.post("/query")
async def query_document(req: QueryRequest):
    """Query a document using reasoning-based retrieval."""
    meta = load_meta()
    if req.document_id not in meta:
        raise HTTPException(404, "Document not found")

    doc_meta    = meta[req.document_id]
    tree        = load_index(doc_meta["index_path"])
    source_file = doc_meta.get("upload_path")

    if req.stream:
        def stream_generator():
            for chunk in retrieve_streaming(
                req.query, tree, req.chat_history, source_file=source_file
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    try:
        result = retrieve(req.query, tree, req.chat_history, source_file=source_file)
        if not req.show_trace:
            result["retrieval_trace"] = []
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")


@app.post("/query/multi")
async def query_multiple(
    document_ids: list[str],
    query: str,
    chat_history: Optional[list] = None,
):
    """Query across multiple documents."""
    meta    = load_meta()
    results = []
    for doc_id in document_ids:
        if doc_id not in meta:
            continue
        doc_meta    = meta[doc_id]
        tree        = load_index(doc_meta["index_path"])
        source_file = doc_meta.get("upload_path")   # FIX 6: was missing source_file
        result = retrieve(query, tree, chat_history, source_file=source_file)
        results.append({
            "document_id":    doc_id,
            "document_title": doc_meta["title"],
            **result,
        })
    return {"results": results, "query": query}