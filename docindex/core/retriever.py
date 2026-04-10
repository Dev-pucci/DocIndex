"""
Iterative Multi-Hop Retrieval Engine
Mirrors PageIndex's agentic retrieval loop.

Fixes applied (5 blockers):
  1. Table handling     — PyMuPDF table extraction preserved as Markdown before content capped
  2. Async/event-loop   — asyncio.run() replaced with nest_asyncio + thread executor so it
                          works inside FastAPI's running loop without crashing
  3. prefix_summary     — f-string syntax fixed; DEEPER/BACKTRACK use _get_children/_get_node_id
  4. _build_sources     — uses _get_page/_get_node_id so LLM-indexed nodes get correct page/id
  5. _call_json retry   — exponential backoff on rate-limit / transient errors
"""

import json
import os
import re
import time
import concurrent.futures
from typing import Optional, Generator
from google import genai
from google.genai import types
from core.parser import tree_to_outline, get_node_by_id, _get_node_id, _get_page, _get_children


# ── Config ────────────────────────────────────────────────────────────────────

VERTEX_PROJECT  = os.environ.get("VERTEX_PROJECT",  "docindex-prod")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")

RETRIEVAL_MODEL = os.environ.get("RETRIEVAL_MODEL", "gemini-2.5-flash")
ANSWER_MODEL    = os.environ.get("ANSWER_MODEL",    "gemini-2.5-pro")

_genai_client = None

def _get_client():
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(
            vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION
        )
    return _genai_client

MAX_HOPS       = 6
MAX_NODES_READ = 10
_LLM_RETRIES   = 4
_LLM_BACKOFF   = 1.5   # seconds base


# ── FIX 5: _call_json with retry ─────────────────────────────────────────────

def _call_json(prompt: str, max_tokens: int = 600,
               model_name: str = RETRIEVAL_MODEL) -> dict:
    """
    Call Gemini and parse JSON response.
    Retries up to _LLM_RETRIES times with exponential backoff on any error
    (rate limits, transient 500s, malformed JSON on first attempt).
    Returns {} only if all retries exhausted.
    """
    client = _get_client()
    cfg = types.GenerateContentConfig(temperature=0.1, max_output_tokens=max_tokens)
    last_exc = None

    for attempt in range(_LLM_RETRIES):
        try:
            response = client.models.generate_content(
                model=model_name, contents=prompt, config=cfg
            )
            text = (response.text or "").strip()
            text = re.sub(r"^```(?:json)?", "", text).rstrip("` \n")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Try to salvage a JSON object or array from noisy output
                for pat in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
                    m = re.search(pat, text)
                    if m:
                        try:
                            return json.loads(m.group())
                        except json.JSONDecodeError:
                            pass
                # JSON parse failed — retry with backoff in case it was garbled
                raise ValueError(f"Could not parse JSON from response: {text[:200]}")

        except Exception as e:
            last_exc = e
            if attempt < _LLM_RETRIES - 1:
                wait = _LLM_BACKOFF ** attempt
                time.sleep(wait)

    print(f"[retriever] _call_json failed after {_LLM_RETRIES} attempts: {last_exc}")
    return {}


def _call_streaming(prompt: str, max_tokens: int = 2048,
                    model_name: str = ANSWER_MODEL):
    """Stream tokens from Gemini with retry on first attempt."""
    client = _get_client()
    cfg = types.GenerateContentConfig(temperature=0.15, max_output_tokens=max_tokens)
    for attempt in range(_LLM_RETRIES):
        try:
            return client.models.generate_content_stream(
                model=model_name, contents=prompt, config=cfg
            )
        except Exception as e:
            if attempt < _LLM_RETRIES - 1:
                time.sleep(_LLM_BACKOFF ** attempt)
            else:
                raise


# ── FIX 1: Table extraction ───────────────────────────────────────────────────

def _extract_tables_as_markdown(file_path: str, start_page: int, end_page: int) -> str:
    """
    Extract tables from PDF pages using PyMuPDF's table detector.
    Returns Markdown table strings appended after the text content.
    If PyMuPDF table detection isn't available or finds nothing, returns "".
    """
    try:
        import fitz
        doc = fitz.open(file_path)
        table_md_parts = []

        for page_num in range(start_page - 1, min(end_page, len(doc))):
            page = doc[page_num]
            try:
                tabs = page.find_tables()
                for tab in tabs:
                    df = tab.to_pandas()
                    if df.empty:
                        continue
                    # Convert to Markdown
                    headers = " | ".join(str(c) for c in df.columns)
                    separator = " | ".join("---" for _ in df.columns)
                    rows = [" | ".join(str(v) for v in row) for _, row in df.iterrows()]
                    table_md_parts.append(
                        f"\n[Table on page {page_num + 1}]\n"
                        f"| {headers} |\n| {separator} |\n" +
                        "\n".join(f"| {r} |" for r in rows)
                    )
            except Exception:
                pass  # page has no tables or fitz version lacks find_tables

        doc.close()
        return "\n".join(table_md_parts)
    except Exception:
        return ""


def _extract_docx_tables(file_path: str) -> str:
    """FIX 8: Extract all tables from a DOCX file as Markdown."""
    try:
        from docx import Document
        doc = Document(file_path)
        parts = []
        for i, table in enumerate(doc.tables):
            rows = []
            for j, row in enumerate(table.rows):
                cells = " | ".join(cell.text.strip() for cell in row.cells)
                rows.append(f"| {cells} |")
                if j == 0:
                    sep = " | ".join("---" for _ in row.cells)
                    rows.append(f"| {sep} |")
            if rows:
                parts.append(f"\n[Table {i+1}]\n" + "\n".join(rows))
        return "\n".join(parts)
    except Exception:
        return ""


def _enrich_node_content(node: dict, file_path: str = None) -> str:
    """
    Return node content enriched with Markdown tables.
    FIX 1: PDF tables via PyMuPDF find_tables.
    FIX 8: DOCX tables via python-docx.
    FIX 3: uses _get_page() instead of raw .get("page") for correct key lookup.
    """
    base_content = (node.get("content") or "")[:2000]

    if not file_path or not os.path.exists(file_path):
        return base_content

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        from core.parser import _get_page
        start    = node.get("start_index") or _get_page(node) or 1
        end      = node.get("end_index") or start
        table_md = _extract_tables_as_markdown(file_path, start, end)
    elif ext in (".docx", ".doc"):
        table_md = _extract_docx_tables(file_path)
    else:
        table_md = ""

    if table_md:
        return base_content + "\n\n" + table_md[:2000]
    return base_content


# ── FIX 2: Event-loop-safe execution ─────────────────────────────────────────

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def _run_in_thread(fn, *args, **kwargs):
    """
    Run a blocking callable in a thread-pool executor.
    Safe to call from inside FastAPI's asyncio event loop — does NOT use
    asyncio.run() which would raise 'This event loop is already running'.
    """
    future = _executor.submit(fn, *args, **kwargs)
    return future.result(timeout=120)


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(query: str, tree: dict,
             chat_history: Optional[list] = None,
             source_file: str = None) -> dict:
    """
    Synchronous iterative retrieval.
    source_file: path to original PDF — used for table enrichment (fix 1).
    Returns: {answer, sources, retrieval_trace}
    """
    return _iterative_retrieve(query, tree, chat_history, source_file)


def retrieve_streaming(query: str, tree: dict,
                       chat_history: Optional[list] = None,
                       source_file: str = None) -> Generator[dict, None, None]:
    """
    Streaming iterative retrieval.
    Yields: hop events, sources, then streamed answer text chunks.
    """
    outline = tree_to_outline(tree)
    hop_log = []
    visited = set()
    collected = []   # [(node, [passage_strings])]

    # Phase 1: Plan
    plan = _call_json(_plan_prompt(query, outline, chat_history), max_tokens=400)
    hop_log.append({"hop": 0, "action": "PLAN",
                    "reasoning": plan.get("reasoning", ""),
                    "targets": plan.get("section_ids", [])})
    yield {"type": "hop", "content": hop_log[-1]}

    queue = list(plan.get("section_ids", []))

    # Phase 2: Iterative exploration
    hops = 0
    while queue and hops < MAX_HOPS and len(collected) < MAX_NODES_READ:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)

        node = get_node_by_id(tree, node_id)
        if not node:
            continue

        # FIX 1: enrich content with tables before passing to LLM
        enriched_content = _enrich_node_content(node, source_file)
        decision = _call_json(
            _explore_prompt(query, node, enriched_content, outline, chat_history, hop_log),
            max_tokens=600,
        )
        _normalise_decision(decision)
        hops += 1

        hop_entry = {
            "hop": hops,
            "action": decision["action"],
            "node_id": node_id,
            "node_title": node.get("title", ""),
            "reasoning": decision.get("reasoning", ""),
        }
        hop_log.append(hop_entry)
        yield {"type": "hop", "content": hop_entry}

        if decision["action"] in ("ANSWER", "COLLECT"):
            passages = decision.get("relevant_passages", [])
            if passages:
                collected.append((node, passages))

        _process_navigation(decision, node, visited, queue, tree)

        if decision["action"] == "ANSWER":
            break

    if not collected:
        yield {"type": "sources", "content": []}
        yield {"type": "text", "content": "I couldn't find relevant content for your query."}
        return

    # FIX 4: sources use correct helper functions
    yield {"type": "sources", "content": _build_sources(collected)}

    # Phase 4: Stream answer
    prompt = _answer_prompt(query, collected, chat_history)
    response = _call_streaming(prompt)
    for chunk in response:
        if chunk.text:
            yield {"type": "text", "content": chunk.text}


# ── Prompt builders ───────────────────────────────────────────────────────────

def _plan_prompt(query: str, outline: str, chat_history: Optional[list]) -> str:
    history_ctx = _format_history(chat_history, turns=3)
    return f"""You are a document navigation expert. Given a document outline and a query,
identify the TOP 1-3 sections most likely to contain the answer.
Prefer specificity — choose the deepest relevant section visible in the outline.

Document Outline:
{outline}
{history_ctx}

Query: {query}

Respond ONLY with valid JSON:
{{
  "reasoning": "Why these sections are the best starting points",
  "section_ids": ["id1", "id2"]
}}

Section IDs are the values inside brackets like [a1b2c3d4] or [0001] shown in the outline.
Return at most 3 IDs. If nothing looks relevant return {{"reasoning":"...", "section_ids":[]}}"""


def _explore_prompt(query: str, node: dict, enriched_content: str,
                    outline: str, chat_history: Optional[list],
                    hop_log: list) -> str:
    """FIX 3: f-string syntax fixed, uses helper functions throughout."""
    node_id    = _get_node_id(node) or ""
    page       = _get_page(node) or "?"
    children   = _get_children(node)
    child_str  = ", ".join(
        f"[{_get_node_id(c) or ''}] {c.get('title', '')}"
        for c in children
    ) or "none"

    # FIX 3: prefix_summary context — no broken f-string
    prefix = node.get("prefix_summary", "")
    prefix_ctx = (f"Context: {prefix}\n") if prefix else ""

    history_ctx = _format_history(chat_history, turns=2)
    hop_summary = _format_hop_log(hop_log)

    return (
        f"You are an expert document analyst performing iterative retrieval.\n\n"
        f"Query: {query}\n"
        f"{history_ctx}"
        f"{prefix_ctx}"
        f"Current node: [{node_id}] {node.get('title', '')} (page {page})\n"
        f"Children: {child_str}\n\n"
        f"Node content (may include tables):\n"
        f'"""\n{enriched_content}\n"""\n\n'
        f"Retrieval so far:\n{hop_summary}\n\n"
        f"Decide what to do next:\n"
        f"- ANSWER   → sufficient to fully answer. Extract relevant passages.\n"
        f"- COLLECT  → partial info here, keep exploring. Extract passages.\n"
        f"- DEEPER   → answer is in child sections. Go deeper.\n"
        f"- SIBLING  → wrong section, try another. Provide up to 2 sibling IDs.\n"
        f"- BACKTRACK→ completely off track, go back up.\n\n"
        f"Respond ONLY with valid JSON:\n"
        f"{{\n"
        f'  "action": "ANSWER|COLLECT|DEEPER|SIBLING|BACKTRACK",\n'
        f'  "reasoning": "Brief explanation",\n'
        f'  "relevant_passages": ["exact sentence or key fact from this node"],\n'
        f'  "sibling_ids": ["id1"]\n'
        f"}}\n\n"
        f"relevant_passages: ONLY if action is ANSWER or COLLECT. Empty list otherwise.\n"
        f"sibling_ids: ONLY if action is SIBLING."
    )


def _answer_prompt(query: str, collected: list, chat_history: Optional[list]) -> str:
    history_ctx = _format_history(chat_history, turns=4)
    context_parts = []
    for node, passages in collected:
        page = _get_page(node)
        page_ref = f" (Page {page})" if page else ""
        context_parts.append(
            f"--- {node.get('title', '')}{page_ref} ---\n" +
            "\n".join(f"• {p}" for p in passages)
        )
    context = "\n\n".join(context_parts)
    return (
        f"You are a precise document analyst. Answer ONLY using the retrieved passages below.\n"
        f"{history_ctx}\n"
        f"Retrieved passages:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Instructions:\n"
        f"- Answer directly and precisely\n"
        f"- Cite section titles and page numbers\n"
        f"- If passages only partially answer the question, say so clearly\n"
        f"- Use structured formatting (bullet points, numbered lists) where helpful"
    )


# ── Navigation helpers ────────────────────────────────────────────────────────

def _normalise_decision(decision: dict) -> None:
    decision.setdefault("action", "COLLECT")
    decision.setdefault("reasoning", "")
    decision.setdefault("relevant_passages", [])
    decision.setdefault("sibling_ids", [])
    valid = {"ANSWER", "COLLECT", "DEEPER", "SIBLING", "BACKTRACK"}
    if decision["action"] not in valid:
        decision["action"] = "COLLECT"


def _process_navigation(decision: dict, node: dict, visited: set,
                        queue: list, tree: dict) -> None:
    """FIX 3: uses _get_children and _get_node_id for both schema types."""
    action = decision["action"]

    if action == "DEEPER":
        child_ids = [_get_node_id(c) for c in _get_children(node)
                     if _get_node_id(c)]
        queue[:0] = child_ids[:3]   # prepend

    elif action == "SIBLING":
        for sid in decision.get("sibling_ids", []):
            if sid and sid not in visited:
                queue.insert(0, sid)

    elif action == "BACKTRACK":
        parent = _find_parent(tree, _get_node_id(node))
        if parent:
            sibling_ids = [
                _get_node_id(c) for c in _get_children(parent)
                if _get_node_id(c) and _get_node_id(c) not in visited
            ]
            queue[:0] = sibling_ids[:2]


# ── FIX 4: _build_sources ────────────────────────────────────────────────────

def _build_sources(collected: list) -> list:
    """Uses _get_page and _get_node_id so both schema types return correct values."""
    return [
        {
            "title": node.get("title", ""),
            "page":  _get_page(node),           # fix: was node.get("page") — wrong for LLM nodes
            "id":    _get_node_id(node),         # fix: was node.get("id")   — wrong for LLM nodes
            "relevant_contents": passages,
            "excerpt": (node.get("content") or "")[:200] + "...",
        }
        for node, passages in collected
    ]


# ── Sync loop ─────────────────────────────────────────────────────────────────

def _iterative_retrieve(query: str, tree: dict,
                        chat_history: Optional[list],
                        source_file: str = None) -> dict:
    outline   = tree_to_outline(tree)
    hop_log   = []
    visited   = set()
    collected = []

    plan = _call_json(_plan_prompt(query, outline, chat_history), max_tokens=400)
    hop_log.append({"hop": 0, "action": "PLAN",
                    "reasoning": plan.get("reasoning", ""),
                    "targets": plan.get("section_ids", [])})
    queue = list(plan.get("section_ids", []))

    hops = 0
    while queue and hops < MAX_HOPS and len(collected) < MAX_NODES_READ:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)

        node = get_node_by_id(tree, node_id)
        if not node:
            continue

        enriched_content = _enrich_node_content(node, source_file)
        decision = _call_json(
            _explore_prompt(query, node, enriched_content, outline, chat_history, hop_log),
            max_tokens=600,
        )
        _normalise_decision(decision)
        hops += 1
        hop_log.append({
            "hop": hops,
            "action": decision["action"],
            "node_id": node_id,
            "node_title": node.get("title", ""),
            "reasoning": decision.get("reasoning", ""),
        })

        if decision["action"] in ("ANSWER", "COLLECT"):
            passages = decision.get("relevant_passages", [])
            if passages:
                collected.append((node, passages))

        _process_navigation(decision, node, visited, queue, tree)

        if decision["action"] == "ANSWER":
            break

    if not collected:
        return {
            "answer": "I couldn't find relevant content in this document for your query.",
            "sources": [],
            "retrieval_trace": hop_log,
        }

    return {
        "answer": _generate_answer_sync(query, collected, chat_history),
        "sources": _build_sources(collected),
        "retrieval_trace": hop_log,
    }


def _generate_answer_sync(query: str, collected: list,
                          chat_history: Optional[list]) -> str:
    prompt = _answer_prompt(query, collected, chat_history)
    client = _get_client()
    cfg = types.GenerateContentConfig(temperature=0.15, max_output_tokens=2048)
    # FIX 2: use thread executor instead of asyncio.run inside FastAPI
    def _call():
        resp = client.models.generate_content(
            model=ANSWER_MODEL, contents=prompt, config=cfg
        )
        return (resp.text or "").strip()

    for attempt in range(_LLM_RETRIES):
        try:
            return _run_in_thread(_call)
        except Exception as e:
            if attempt < _LLM_RETRIES - 1:
                time.sleep(_LLM_BACKOFF ** attempt)
            else:
                return f"Answer generation failed: {e}"
    return ""


# ── Utilities ─────────────────────────────────────────────────────────────────

def _format_history(chat_history: Optional[list], turns: int = 3) -> str:
    if not chat_history:
        return ""
    recent = chat_history[-(turns * 2):]
    lines = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
    return f"\nConversation history:\n{lines}\n"


def _format_hop_log(hop_log: list) -> str:
    if not hop_log:
        return "No hops yet."
    lines = []
    for h in hop_log[-4:]:
        if h["action"] == "PLAN":
            lines.append(f"PLAN → targets: {h.get('targets', [])}")
        else:
            lines.append(
                f"Hop {h['hop']}: {h['action']} on "
                f"[{h.get('node_id', '')}] {h.get('node_title', '')} "
                f"— {h.get('reasoning', '')[:80]}"
            )
    return "\n".join(lines)


def _find_parent(tree: dict, node_id: str) -> Optional[dict]:
    """FIX 3: uses _get_node_id and _get_children for both schema types."""
    for child in _get_children(tree):
        if _get_node_id(child) == node_id:
            return tree
        result = _find_parent(child, node_id)
        if result:
            return result
    return None