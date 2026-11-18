"""Connection layer between the Streamlit front-end (app.py) and the two backends.

Front-end  (app.py, Streamlit UI)
    |
    v
backend_bridge.py          <-- this file: imports, wires and guards the backends
    |            \\
    |             \\--> Backend B: advisor.academic_advisor_recommend
    |                            (+ rule_engine.py + courses.json)
    \\--> Backend A: rag_backend.rag_course_query
                     (knowledge_base/*.pdf -> Chroma -> Ollama)

Why an extra module instead of importing the backends straight into app.py:

* app.py stays presentation-only, so the UI can be redesigned without touching the glue;
* the glue is importable (and testable) without starting a Streamlit runtime;
* every backend call is wrapped, so a missing pip package, a dead Ollama server or an
  empty vector index degrades into a readable message instead of a Streamlit traceback.

Public API used by the UI (both return {"answer": str, "source_files": list[str]}):
    rag_course_query(user_query)                          -> basic feature, course lookup
    academic_advisor_recommend(finished, credits, pref, semester=None)
                                                          -> additional feature, advisor
    backend_status()                                      -> dict for the UI status panel
"""

from __future__ import annotations

import os
import re

# ------------------------------------------------------------------ Backend A: RAG lookup
RAG_IMPORT_ERROR = ""
try:  # rag_backend imports chromadb / sentence-transformers lazily, so this stays cheap
    from rag_backend import rag_course_query as _rag_query
except Exception as exc:  # noqa: BLE001 - any import problem must not kill the UI
    _rag_query = None
    RAG_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# ------------------------------------------------------------------ Backend B: advisor
ADVISOR_IMPORT_ERROR = ""
try:
    from advisor import academic_advisor_recommend as _advisor_query
except Exception as exc:  # noqa: BLE001
    _advisor_query = None
    ADVISOR_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# Feed syllabus excerpts into the advisor prompt. On by default; USE_RAG_SNIPPETS=0 disables.
USE_RAG_SNIPPETS = os.getenv("USE_RAG_SNIPPETS", "1") == "1"

CODE_RE = re.compile(r"\b([A-Z]{3,4})[\s\-]?(\d[A-Z0-9]{2,3})\b")
_STOPWORDS = {
    "A", "AN", "THE", "ABOUT", "TELL", "ME", "WHAT", "IS", "ARE", "OF", "FOR", "IN",
    "ON", "TO", "COURSE", "SUBJECT", "SYLLABUS", "DESCRIPTION", "ASSESSMENT", "CREDIT",
    "CREDITS", "PREREQUISITE", "PREREQUISITES", "OFFERED", "PLEASE", "SHOW", "GIVE",
    "CAN", "YOU", "I", "AND", "WITH", "DOES", "IT", "HAVE", "MUCH", "MANY", "HOW",
}


def _empty():
    return {"answer": "", "source_files": []}


def _norm_code(text: str) -> str:
    match = CODE_RE.search(str(text or "").upper())
    return f"{match.group(1)}{match.group(2)}" if match else ""


def _load_catalog() -> dict:
    """{course_code: course_dict} from courses.json, reloaded on every call (cheap, 81 rows)."""
    try:
        from rule_engine import load_courses
        return load_courses()
    except Exception:  # noqa: BLE001
        return {}


def _blank(value) -> bool:
    """True for the many spellings of "nothing" used across courses.json."""
    text = str(value or "").strip()
    return re.fullmatch(r"[/\-]?\s*(\()?none(\))?|nil|n/?a", text, re.IGNORECASE) is not None or not text


def _assessment_text(course: dict) -> str:
    assessment = course.get("assessment") or {}
    parts = []
    if assessment.get("exam_pct") is not None:
        parts.append(f"Exam {assessment['exam_pct']}%")
    if assessment.get("continuous_pct") is not None:
        parts.append(f"Continuous {assessment['continuous_pct']}%")
    text = " · ".join(parts)
    detail = (assessment.get("continuous_detail") or "").strip()
    if detail and "no split given" not in detail.lower():
        text = f"{text} ({detail})" if text else detail
    return text or "not listed in the Subject Description Form"


def _offered_text(course: dict) -> str:
    offered = course.get("offered_2026_27")
    if isinstance(offered, dict):
        semesters = [sem for sem in ("S1", "S2") if offered.get(sem)]
        if semesters:
            return ", ".join(semesters)
        return "not offered in 2026/27"
    if offered is None:
        return "not offered in 2026/27" if (course.get("offered_note") or "").strip() else "not listed"
    return str(offered)


def _fact_sheet(course: dict) -> dict:
    """Turn one courses.json row into the answer markdown shown in the chat."""
    code = course["course_code"]
    title = course.get("title") or ""
    lines = [f"**{code} – {title}**", ""]

    meta = [f"Credits: **{course.get('credit')}**", f"Level: **{course.get('level')}**"]
    if course.get("category"):
        meta.append(f"Area: **{course['category']}**")
    if course.get("coding_load"):
        meta.append(f"Coding load: **{course['coding_load']}**")
    lines.append("- " + " · ".join(meta))

    if course.get("credit_type") == "training":
        lines.append("- Training credit – this course is not counted as a taught credit")
    prereq = "" if _blank(course.get("prerequisite_text")) else str(course["prerequisite_text"]).strip()
    lines.append(f"- Prerequisite: {prereq if prereq else 'none listed'}")
    if not _blank(course.get("exclusion")):
        lines.append(f"- Excluded: {str(course['exclusion']).strip()}")
    lines.append(f"- Assessment: {_assessment_text(course)}")
    lines.append(f"- Offered 2026/27: {_offered_text(course)}")
    lines.append(f"- Elective: {'yes' if course.get('elective_status') == 'elective' else 'not on the elective list'}")
    if (course.get("summary") or "").strip():
        lines.append(f"- Summary: {course['summary']}")
    if (course.get("note") or "").strip():
        lines.append(f"- Note: {course['note']}")

    source = (course.get("source") or "").strip()
    source_line = f"_[{code} Subject Description Form]({source})_" if source else ""
    if course.get("form_date"):
        source_line += f" · form date {course['form_date']}"
    if source_line:
        lines += ["", source_line]
    return {"answer": "\n".join(lines), "source_files": [f"{code} Subject Description Form"] if source else []}


def lookup_from_catalog(user_query: str) -> dict:
    """Answer a course-lookup question from courses.json alone.

    This is the fallback behind the RAG backend: knowledge_base/ only holds 10 of the
    81 courses, so without it the chat would answer "not found" for most of the syllabus.
    """
    courses = _load_catalog()
    if not courses:
        return _empty()

    code = _norm_code(user_query)
    if code:
        # DSAI4203 is stored as an equivalent of COMP4432 etc. - resolve through the aliases too.
        if code not in courses:
            for row in courses.values():
                if code in (row.get("equivalent_codes") or []):
                    code = row["course_code"]
                    break
        return _fact_sheet(courses[code]) if code in courses else _empty()

    # No code in the question: match the course title, longest title first.
    query_upper = str(user_query or "").upper()
    words = [w for w in re.findall(r"[A-Za-z0-9]+", query_upper) if w not in _STOPWORDS and len(w) > 1]
    best = None
    for row in courses.values():
        title = (row.get("title") or "").upper()
        if not title:
            continue
        if title in query_upper:
            best = row if best is None or len(title) > len(best.get("title") or "") else best
            continue
        title_words = [w for w in re.findall(r"[A-Za-z0-9]+", title) if len(w) > 2]
        if title_words and all(w in words for w in title_words):
            best = row if best is None or len(title) > len(best.get("title") or "") else best
    return _fact_sheet(best) if best else _empty()


# ------------------------------------------------------------------ public API
def rag_course_query(user_query: str) -> dict:
    """Basic feature – course lookup.

    Order: real RAG backend (syllabus PDFs) -> courses.json fact sheet -> help message.
    The extra "engine" key is informational; the UI may ignore it.
    """
    if _rag_query is not None:
        try:
            result = _rag_query(user_query) or {}
        except Exception as exc:  # noqa: BLE001 - contract says rag_backend never raises
            result = {"answer": "", "source_files": [], "error": f"{type(exc).__name__}: {exc}"}
        if (result.get("answer") or "").strip():
            return {"answer": result["answer"], "source_files": result.get("source_files") or [],
                    "engine": "RAG (syllabus PDF)"}

    fallback = lookup_from_catalog(user_query)
    if fallback["answer"]:
        return {"answer": fallback["answer"], "source_files": fallback["source_files"],
                "engine": "courses.json"}

    hint = "I could not match that to a COMP course."
    if RAG_IMPORT_ERROR:
        hint += f" (RAG backend unavailable: {RAG_IMPORT_ERROR})"
    hint += ("\n\nTry a course code such as **COMP4431**, or a course title such as "
             "*Artificial Intelligence*. RAG covers the 10 Subject Description Forms in "
             "`knowledge_base/`; the other courses are answered from `courses.json`.")
    return {"answer": hint, "source_files": [], "engine": "none"}


def academic_advisor_recommend(finished_courses: list[str], remaining_credit: int,
                               preference_text: str, semester=None) -> dict:
    """Additional feature – rule engine + optional RAG snippets + LLM wording."""
    if _advisor_query is None:
        return {"answer": (f"The advisor backend could not be imported: {ADVISOR_IMPORT_ERROR}\n\n"
                           "Run `pip install -r requirements.txt` and keep advisor.py, "
                           "rule_engine.py and courses.json in the same folder as app.py."),
                "source_files": []}
    rag_fn = _rag_query if (USE_RAG_SNIPPETS and _rag_query is not None) else None
    try:
        return _advisor_query(finished_courses, remaining_credit, preference_text,
                              semester=semester, rag_fn=rag_fn)
    except Exception as exc:  # noqa: BLE001
        return {"answer": f"Advisor failed: {type(exc).__name__}: {exc}", "source_files": []}


def backend_status() -> dict:
    """Small dict the UI renders in the info panel - proves which backends are live."""
    try:
        rag_rows = len(_load_catalog())
    except Exception:  # noqa: BLE001
        rag_rows = 0
    return {
        "rag": "connected" if _rag_query is not None else "unavailable",
        "rag_error": RAG_IMPORT_ERROR,
        "advisor": "connected" if _advisor_query is not None else "unavailable",
        "advisor_error": ADVISOR_IMPORT_ERROR,
        "rag_snippets": USE_RAG_SNIPPETS and _rag_query is not None,
        "catalog_rows": rag_rows,
    }


# ------------------------------------------------------------------ CLI smoke test
if __name__ == "__main__":
    import json
    print(json.dumps(backend_status(), indent=2, ensure_ascii=False))

    for query in ("What is the assessment of COMP4431?", "COMP3131", "Artificial Intelligence"):
        out = rag_course_query(query)
        print(f"\n=== {query} -> [{out['engine']}]")
        print(out["answer"][:400])
        print("sources:", out["source_files"])

    advisor = academic_advisor_recommend(
        ["COMP1011", "COMP2011", "COMP2411", "COMP2421"], 12,
        "I prefer AI-focused courses and want to avoid heavy coding coursework.", semester="S2")
    print("\n=== advisor ===")
    print(advisor["answer"][:800])
    print("sources:", advisor["source_files"], "| LLM used:", advisor.get("used_llm"))