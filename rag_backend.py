"""rag_backend.py - Basic feature: syllabus lookup RAG (member A's role).

Pipeline:
    knowledge_base/*.pdf
        -> pypdf text extraction (page by page)
        -> paragraph-aware chunking with overlap
        -> sentence-transformers embeddings (all-MiniLM-L6-v2 by default)
        -> Chroma persistent vector store (chroma_db/)
        -> retrieve the top-k chunks OF THE ASKED COURSE
        -> local Ollama model writes a grounded answer
        -> {"answer": ..., "source_files": [...]}

Public API - the only function the chatbot / advisor should call:
    rag_course_query(course_code) -> {"answer": str, "source_files": list[str]}
It NEVER raises: every failure (no PDFs, missing library, Ollama down, unknown course)
returns {"answer": "", "source_files": []}, so the Streamlit app cannot crash on it.

Environment variables (all optional):
    RAG_KB_DIR         knowledge_base/                     folder with the syllabus PDFs
    RAG_DB_DIR         chroma_db/                          persistent Chroma store
    RAG_EMBED_MODEL    sentence-transformers/all-MiniLM-L6-v2
    RAG_TOP_K          4                                   chunks given to the LLM
    RAG_CHUNK_SIZE     900                                 characters per chunk
    RAG_CHUNK_OVERLAP  150
    RAG_LLM            1                                   0 = skip the LLM, quote the syllabus text
    RAG_OLLAMA_URL     http://localhost:11434/api/generate
    RAG_OLLAMA_MODEL   auto                                first installed model from MODEL_FALLBACKS

Command line:
    python rag_backend.py --build            # (re)build the vector store
    python rag_backend.py COMP4431           # one course lookup
    python rag_backend.py --ask "What is the assessment of COMP4431?"

Dependencies (see requirements.txt):
    pip install chromadb sentence-transformers pypdf        # Python 3.10-3.12 recommended
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

KB_DIR = Path(os.getenv("RAG_KB_DIR", HERE / "knowledge_base"))
DB_DIR = Path(os.getenv("RAG_DB_DIR", HERE / "chroma_db"))
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "comp_syllabus")
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
MIN_CHUNK = 40
USE_LLM = os.getenv("RAG_LLM", "1") == "1"

OLLAMA_URL = os.getenv("RAG_OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_TAGS_URL = os.getenv("RAG_OLLAMA_TAGS_URL", OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/tags")
OLLAMA_MODEL = os.getenv("RAG_OLLAMA_MODEL", "")
MODEL_FALLBACKS = ["qwen2.5:3b", "qwen2.5:7b", "qwen3.5:4b", "qwen3.5:9b"]

MANIFEST_PATH = DB_DIR / "manifest.json"
# COMP4431 / COMP 4431 / COMP-4431 / DSAI2201 / AMA1007
CODE_RE = re.compile(r"\b([A-Z]{3,4})[\s\-]?(\d[A-Z0-9]{2,3})\b")
SUBJECT_CODE_RE = re.compile(
    r"subject\s*code\s*[:：\-]?\s*([A-Z]{3,4}[\s\-]?\d[A-Z0-9]{2,3})", re.IGNORECASE)
# "Jul 2025" / "JULY2025" in the form header looks exactly like a course code to CODE_RE.
_MONTH_PREFIXES = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

_EMBEDDER = None
_CLIENT = None
_COLLECTION = None
_BUILT_FP = None
_MODEL_NAME = None
_LOCK = threading.RLock()


# ------------------------------------------------------------------ small helpers
def _empty():
    """The 'could not retrieve anything' answer. A fresh dict each call."""
    return {"answer": "", "source_files": []}


def _log(msg):
    print(f"[rag] {msg}", file=sys.stderr)


def extract_course_code(text):
    """'Tell me about comp 4431' -> 'COMP4431'. '' when there is no code in the text."""
    for match in CODE_RE.finditer(str(text or "").upper()):
        prefix, digits = match.group(1), match.group(2)
        if prefix[:3] in _MONTH_PREFIXES and digits.isdigit() and len(digits) == 4:
            continue  # "Jul 2025" in the form header is a date, not a course
        return f"{prefix}{digits}"
    return ""


def _is_bare_code(text):
    return bool(re.fullmatch(r"[\s\-]*[A-Za-z]{3,4}[\s\-]?\d[A-Za-z0-9]{2,3}[\s.!?]*", str(text or "")))


def declared_subject_code(page_text):
    """The code the DOCUMENT declares ('Subject Code: COMP4431'), else ''.

    Only an explicit declaration is trusted per page: later pages often mention other
    courses (prerequisites, exclusions), so "first code on the page" is not reliable.
    """
    match = SUBJECT_CODE_RE.search(page_text)
    return extract_course_code(match.group(1)) if match else ""


def _code_from_filename(path):
    return extract_course_code(path.stem) or re.sub(r"[^A-Z0-9]+", "", path.stem.upper())[:12]


def _clean_text(text):
    text = str(text or "").replace("\u00a0", " ")
    text = text.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _squash(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


# ------------------------------------------------------------------ PDF -> chunks
def pdf_files():
    """All syllabus PDFs in knowledge_base/, sorted (empty list if the folder does not exist)."""
    if not KB_DIR.is_dir():
        return []
    return sorted(p for p in KB_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def pdf_pages(path):
    """[(page_number, text), ...] - pages without extractable text (scanned) are skipped."""
    from pypdf import PdfReader  # imported here so importing this module needs no dependencies

    pages = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        try:
            text = _clean_text(page.extract_text() or "")
        except Exception:
            text = ""
        if text:
            pages.append((number, text))
    return pages


def chunk_text(text):
    """Split one page into chunks of about CHUNK_SIZE characters, keeping paragraphs together."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    blocks, buffer = [], ""
    for paragraph in paragraphs:
        buffer = f"{buffer}\n{paragraph}".strip() if buffer else paragraph
        if len(buffer) >= CHUNK_SIZE:
            blocks.append(buffer)
            buffer = ""
    if buffer:
        blocks.append(buffer)

    chunks = []
    for block in blocks:
        if len(block) <= CHUNK_SIZE * 1.5:  # a normal page/paragraph - keep it whole
            chunks.append(block)
            continue
        step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)  # very long block: sliding window
        for start in range(0, len(block), step):
            chunks.append(block[start:start + CHUNK_SIZE])
    return [c for c in chunks if len(c) >= MIN_CHUNK]


# ------------------------------------------------------------------ embeddings + store
def _embed(texts):
    """Sentence-transformer embeddings, normalised so cosine and L2 rank identically."""
    global _EMBEDDER
    with _LOCK:
        if _EMBEDDER is None:
            from sentence_transformers import SentenceTransformer

            _EMBEDDER = SentenceTransformer(EMBED_MODEL)
    vectors = _EMBEDDER.encode(list(texts), batch_size=32, show_progress_bar=False,
                               normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def _open_collection():
    """Open (creating if needed) the persistent Chroma collection. Cached per process."""
    global _CLIENT, _COLLECTION
    import chromadb

    with _LOCK:
        if _CLIENT is None:
            DB_DIR.mkdir(parents=True, exist_ok=True)
            try:
                from chromadb.config import Settings

                _CLIENT = chromadb.PersistentClient(path=str(DB_DIR),
                                                    settings=Settings(anonymized_telemetry=False))
            except Exception:
                _CLIENT = chromadb.PersistentClient(path=str(DB_DIR))
        if _COLLECTION is None:
            _COLLECTION = _CLIENT.get_or_create_collection(name=COLLECTION_NAME)
        return _COLLECTION


def _fingerprint(files):
    """Changes whenever a PDF is added/removed/edited or the chunking/embedding settings change."""
    digest = hashlib.sha256()
    digest.update(f"{EMBED_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{COLLECTION_NAME}".encode())
    for path in files:
        stat = path.stat()
        digest.update(f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}".encode())
    return digest.hexdigest()


def _read_manifest():
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_manifest(payload):
    DB_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_index(force=False, verbose=True):
    """(Re)build the vector store from knowledge_base/*.pdf. Returns the number of chunks.

    Rebuild is skipped when the manifest fingerprint still matches the PDFs and the
    collection is non-empty, so the Streamlit process pays for embedding only once.
    """
    global _COLLECTION, _BUILT_FP
    say = _log if verbose else (lambda *_: None)

    files = pdf_files()
    if not files:
        say(f"no PDFs in {KB_DIR} - nothing to index")
        return 0

    fingerprint = _fingerprint(files)
    with _LOCK:
        collection = _open_collection()
        if not force and collection.count() > 0 and _read_manifest().get("fingerprint") == fingerprint:
            _BUILT_FP = fingerprint
            say(f"index already up to date ({collection.count()} chunks)")
            return collection.count()

        say(f"building index from {len(files)} PDF(s) ...")
        try:
            _CLIENT.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        _COLLECTION = _CLIENT.create_collection(name=COLLECTION_NAME)
        collection = _COLLECTION

        pending = {"ids": [], "documents": [], "metadatas": []}
        total = 0

        def flush():
            nonlocal total
            if not pending["ids"]:
                return
            collection.add(ids=pending["ids"], embeddings=_embed(pending["documents"]),
                           documents=pending["documents"], metadatas=pending["metadatas"])
            total += len(pending["ids"])
            pending["ids"], pending["documents"], pending["metadatas"] = [], [], []

        for path in files:
            pages = pdf_pages(path)
            file_code = _code_from_filename(path)
            if not pages:
                say(f"  {path.name}: no extractable text (scanned PDF?) - skipped")
                continue
            for page_number, text in pages:
                # The filename is authoritative: every page of COMP4431.pdf belongs to COMP4431.
                # Only an explicit "Subject Code" line overrides it (PDFs holding several subjects).
                code = declared_subject_code(text) or file_code
                for i, chunk in enumerate(chunk_text(text)):
                    pending["ids"].append(f"{path.stem}:p{page_number}:c{i}")
                    pending["documents"].append(chunk)
                    pending["metadatas"].append({"course_code": code, "source": path.name,
                                                 "page": page_number})
                    if len(pending["ids"]) >= 64:
                        flush()
            say(f"  {path.name}: {len(pages)} page(s), code {file_code}")

        flush()
        _write_manifest({"fingerprint": fingerprint, "embedding_model": EMBED_MODEL,
                         "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
                         "files": [p.name for p in files], "chunks": total,
                         "built_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        _BUILT_FP = fingerprint
        say(f"index built: {total} chunks from {len(files)} PDF(s)")
        return total


def _ensure_index():
    """Collection, building or rebuilding it if the PDFs changed. None if nothing is indexed."""
    global _COLLECTION
    with _LOCK:
        files = pdf_files()
        if not files:
            return None
        if _COLLECTION is not None and _BUILT_FP == _fingerprint(files) and _COLLECTION.count() > 0:
            return _COLLECTION
        try:
            build_index()
        except Exception as exc:  # missing library, model download failure, corrupt store ...
            _log(f"index build failed: {type(exc).__name__}: {exc}")
            return None
        collection = _COLLECTION
        return collection if collection is not None and collection.count() > 0 else None


def _has_course(collection, code):
    try:
        return bool(collection.get(where={"course_code": code}, limit=1).get("ids"))
    except Exception:
        return False


# ------------------------------------------------------------------ LLM
def resolve_model():
    """First installed Ollama model from MODEL_FALLBACKS (machines differ), else the first pulled."""
    global _MODEL_NAME
    if OLLAMA_MODEL:
        return OLLAMA_MODEL
    if _MODEL_NAME is None:
        names = []
        try:
            with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=5) as response:
                names = [m.get("name", "") for m in json.loads(response.read().decode()).get("models", [])]
        except Exception:
            names = []
        _MODEL_NAME = next((m for m in MODEL_FALLBACKS if m in names), names[0] if names else "")
    return _MODEL_NAME


def _generate(payload, timeout):
    request = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8")).get("response", "").strip()


def _ollama(prompt, timeout=90):
    payload = {"model": resolve_model(), "prompt": prompt, "stream": False,
               "options": {"temperature": 0.1, "num_predict": 350}}
    try:
        # qwen3.x otherwise returns an empty "response" (its text goes to "thinking")
        return _generate({**payload, "think": False}, timeout)
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            raise
        return _generate(payload, timeout)  # older Ollama / non-thinking model rejects "think"


def _llm_answer(question, code, blocks):
    if not USE_LLM:
        return ""
    prompt = (
        "You are a course information assistant for PolyU COMP undergraduate students.\n"
        "Answer the QUESTION using ONLY the SYLLABUS EXCERPTS below.\n"
        f"QUESTION: {question}\n"
        f"COURSE: {code or 'unknown'}\n\n"
        "SYLLABUS EXCERPTS:\n" + "\n\n".join(blocks) + "\n\n"
        "Rules: 3-6 short sentences. Quote credit, prerequisite and assessment values exactly as "
        "written. For an assessment weighting report the top-level items only (Continuous "
        "Assessment and Examination), not their sub-components, so the numbers do not look like "
        "they add up to more than 100%. If the excerpts do not contain the answer, say the syllabus "
        "does not state it. Do not invent facts, do not recommend courses, mention the source file once."
    )
    try:
        answer = _ollama(prompt)
    except Exception as exc:
        _log(f"LLM unavailable ({type(exc).__name__}), using syllabus text: {exc}")
        return ""
    return answer if len(answer) >= 30 else ""


def _extractive_answer(code, blocks):
    """Deterministic grounded answer used when the LLM is off or unreachable."""
    header = f"**{code} - syllabus excerpt**" if code else "**Syllabus excerpt**"
    # drop the "[n] (file, p.X)" prefix that was added for the LLM prompt
    cleaned = [re.sub(r"^\[\d+\]\s*\([^)]*\)\s*", "", block).strip() for block in blocks[:2]]
    body = "\n\n".join(f"- {_squash(block)[:600]}" for block in cleaned if block)
    return f"{header}\n\n{body}\n\n_Showing the most relevant syllabus text; the source file is cited below._"


# ------------------------------------------------------------------ public API
def rag_course_query(course_code):
    """Look up one course's syllabus and answer from it.

    Input : course code ("COMP4431") or a question containing one
            ("What is the assessment of COMP4431?").
    Output: {"answer": str, "source_files": list[str]} - both empty when nothing could be
            retrieved (unknown course, empty knowledge base, missing dependency, LLM error).
            This function never raises.
    """
    try:
        raw = str(course_code or "").strip()
        if not raw:
            return _empty()

        code = extract_course_code(raw)
        collection = _ensure_index()
        if collection is None:
            return _empty()

        if code:
            if not _has_course(collection, code):
                _log(f"{code} is not in the knowledge base")
                return _empty()
            where = {"course_code": code}
            query_text = (raw if not _is_bare_code(raw)
                          else f"{code} course syllabus description learning outcomes assessment")
        else:
            where = None  # no code in the question: semantic search over the whole syllabus set
            query_text = raw

        result = collection.query(query_embeddings=_embed([query_text]), n_results=TOP_K,
                                  where=where, include=["documents", "metadatas", "distances"])
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        if not documents:
            return _empty()

        blocks = [f"[{i + 1}] ({m.get('source')}, p.{m.get('page')})\n{d}"
                  for i, (d, m) in enumerate(zip(documents, metadatas))]
        answer = _llm_answer(raw, code, blocks) or _extractive_answer(code, blocks)
        sources = list(dict.fromkeys(m.get("source") for m in metadatas if m.get("source")))
        return {"answer": answer, "source_files": sources}
    except Exception as exc:  # the contract is "never raise"
        _log(f"query failed: {type(exc).__name__}: {exc}")
        return _empty()


# ------------------------------------------------------------------ CLI
def main(argv=None):
    parser = argparse.ArgumentParser(description="COMP syllabus lookup RAG backend")
    parser.add_argument("code", nargs="?", help="course code, e.g. COMP4431")
    parser.add_argument("--ask", help="free-text question")
    parser.add_argument("--build", action="store_true", help="(re)build the vector store")
    parser.add_argument("--force", action="store_true", help="rebuild even if up to date")
    parser.add_argument("--no-llm", action="store_true", help="quote the syllabus text only")
    args = parser.parse_args(argv)

    if args.no_llm:
        global USE_LLM
        USE_LLM = False

    if args.build or (not args.code and not args.ask):
        chunks = build_index(force=args.force)
        print(json.dumps({"chunks": chunks, "kb_dir": str(KB_DIR), "db_dir": str(DB_DIR)},
                         indent=2, ensure_ascii=False))
        if not args.code and not args.ask:
            return 0

    query = args.ask or args.code
    result = rag_course_query(query)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
