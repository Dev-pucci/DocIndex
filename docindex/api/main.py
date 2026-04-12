"""
DocIndex API - FastAPI backend
Endpoints: upload document, query document, list documents
"""

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

class MultiQueryRequest(BaseModel):
    document_ids: list[str]
    query: str
    chat_history: Optional[list] = []


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
    Upload a document and start indexing in the background.
    Returns immediately with document_id and status='indexing'.
    Poll GET /documents/{id}/status to check progress.
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

    # Register document immediately with status=indexing
    with _meta_lock:
        meta = load_meta()
        meta[doc_id] = {
            "id":          doc_id,
            "filename":    file.filename,
            "type":        ext.lstrip("."),
            "title":       file.filename,
            "total_pages": None,
            "sections":    None,
            "status":      "indexing",
            "index_path":  str(index_path),
            "upload_path": str(upload_path),
        }
        save_meta(meta)

    # Index in background thread — does not block the response
    def _do_index():
        try:
            if ext == ".pdf":
                tree = build_index(str(upload_path), add_summary=True)
            else:
                tree = parse_document(str(upload_path))
            save_index(tree, str(index_path))
            with _meta_lock:
                m = load_meta()
                if doc_id in m:
                    m[doc_id].update({
                        "title":       tree.get("title", file.filename),
                        "total_pages": tree.get("total_pages"),
                        "sections":    _sections_count(tree),
                        "status":      "ready",
                    })
                    save_meta(m)
        except Exception as e:
            print(f"[indexer] Background indexing failed for {doc_id}: {e}")
            with _meta_lock:
                m = load_meta()
                if doc_id in m:
                    m[doc_id]["status"] = f"error: {e}"
                    save_meta(m)

    _index_executor.submit(_do_index)

    return {
        "document_id": doc_id,
        "filename":    file.filename,
        "status":      "indexing",
        "message":     "Upload received. Indexing in background — poll /documents/{id}/status",
    }


@app.get("/documents/{document_id}/status")
def get_document_status(document_id: str):
    """Poll this endpoint to check indexing progress."""
    meta = load_meta()
    if document_id not in meta:
        raise HTTPException(404, "Document not found")
    doc = meta[document_id]
    return {
        "document_id": document_id,
        "status":      doc.get("status", "ready"),
        "title":       doc.get("title"),
        "sections":    doc.get("sections"),
        "total_pages": doc.get("total_pages"),
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
def query_multiple(req: MultiQueryRequest):
    """
    Query across multiple documents and synthesize a single answer.
    Retrieves relevant passages from each doc, then calls the LLM once
    with all passages together to produce a unified response.
    """
    from core.retriever import _get_client, ANSWER_MODEL
    from google.genai import types as gtypes

    meta = load_meta()
    all_passages = []
    all_sources  = []

    for doc_id in req.document_ids:
        if doc_id not in meta:
            continue
        doc_meta = meta[doc_id]
        if doc_meta.get("status", "ready") != "ready":
            continue
        tree        = load_index(doc_meta["index_path"])
        source_file = doc_meta.get("upload_path")
        result      = retrieve(req.query, tree, req.chat_history, source_file=source_file)
        doc_title   = doc_meta.get("title", doc_id)
        parties     = tree.get("parties") or []

        if result.get("answer") and "couldn't find" not in result["answer"].lower():
            all_passages.append(
                f"=== {doc_title} ===\n"
                + (f"Parties: {', '.join(parties)}\n" if parties else "")
                + result["answer"]
            )
        for src in result.get("sources", []):
            src["document_title"] = doc_title
            src["document_id"]    = doc_id
            all_sources.append(src)

    if not all_passages:
        return {"answer": "I couldn't find relevant content across the selected documents.", "sources": []}

    # Synthesize a single answer across all documents
    history_ctx = ""
    if req.chat_history:
        turns = req.chat_history[-4:]
        history_ctx = "Conversation so far:\n" + "\n".join(
            f"{t['role'].capitalize()}: {t['content']}" for t in turns
        ) + "\n\n"

    prompt = (
        f"You are an expert analyst. The user is querying across multiple documents.\n"
        f"{history_ctx}"
        f"Question: {req.query}\n\n"
        f"Below are relevant findings retrieved from each document:\n\n"
        + "\n\n".join(all_passages)
        + "\n\nProvide a single unified answer that:\n"
        f"- Compares and contrasts across documents where relevant\n"
        f"- Cites which document each fact comes from\n"
        f"- Uses structured formatting (tables, bullet points) where helpful\n"
        f"- Is precise and comprehensive"
    )

    client = _get_client()
    cfg    = gtypes.GenerateContentConfig(temperature=0.15, max_output_tokens=3000)
    resp   = client.models.generate_content(model=ANSWER_MODEL, contents=prompt, config=cfg)
    answer = (resp.text or "").strip()

    return {"answer": answer, "sources": all_sources}