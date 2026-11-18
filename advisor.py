"""advisor.py - Backend B: Academic Advisor pipeline used by app.py.

Pipeline:  preferences -> rule engine (hard rules) -> optional RAG snippets -> LLM wording -> guardrail.

* The RULE ENGINE decides which courses may be recommended (prerequisites, semester, credits ...).
* The LLM (local Ollama model, no API key) only words the explanation.
* GUARDRAIL: if the LLM mentions a course the engine did not allow, forgets a picked course, or is
  unreachable, we fall back to a deterministic template answer. The demo therefore never breaks and
  the chatbot can never recommend a course the student is not allowed to take.

Settings (environment variables, all optional):
    USE_LLM=0|1            default 1 (set 0 to always use the template answer)
    OLLAMA_URL             default http://localhost:11434/api/generate
    OLLAMA_MODEL           default: auto (first installed model from MODEL_FALLBACKS)
"""
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from rule_engine import CODE_RE, DISCLAIMER, build_llm_prompt, load_courses, norm, recommend

USE_LLM = os.getenv("USE_LLM", "1") == "1"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_TAGS_URL = os.getenv("OLLAMA_TAGS_URL", OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/tags")
# Leave OLLAMA_MODEL empty to use whichever of these is installed (the models differ per machine).
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")
MODEL_FALLBACKS = ["qwen2.5:3b", "qwen2.5:7b", "qwen3.5:4b", "qwen3.5:9b"]
COURSES_PATH = Path(__file__).with_name("courses.json")

# ---------------------------------------------------------------- preference parsing
AREA_KEYWORDS = {
    "AI": [r"\bai\b", r"artificial intelligence", r"machine learning", r"\bml\b", r"deep learning",
           r"computer vision", r"\bnlp\b", r"language processing"],
    "Data": [r"\bdata\b", r"database", r"big data", r"data mining", r"analytics", r"information retrieval"],
    "Security": [r"security", r"cryptograph", r"privacy"],
    "System": [r"\bsystems?\b", r"cloud", r"network", r"operating", r"embedded", r"mobile"],
    "Application": [r"\bweb\b", r"\bgames?\b", r"graphics", r"software", r"\bhci\b", r"interaction",
                    r"e-?commerce", r"\bxr\b", r"extended reality"],
    "Theory": [r"theory", r"algorithm", r"mathemat"],
    "Business": [r"business", r"management", r"entrepreneur"],
    "FinTech": [r"fintech", r"finance", r"financial"],
}
AVOID_CODING_RE = re.compile(
    r"\b(avoid|less|no|without|light|minimal|not\s+too\s+much|not\s+heavy)\b.{0,25}\b(coding|programming|code)\b")
SEMESTER_RE = re.compile(r"\b(?:s|sem|semester)\s*([12])\b")


def parse_preferences(text):
    """Free text -> {'categories': [...], 'avoid_heavy_coding': bool, 'semester': 'S1'|'S2'|None}."""
    t = (text or "").lower()
    cats = [cat for cat, pats in AREA_KEYWORDS.items() if any(re.search(p, t) for p in pats)]
    sem = None
    m = SEMESTER_RE.search(t)
    if m:
        sem = "S" + m.group(1)
    elif re.search(r"\bfirst semester\b", t):
        sem = "S1"
    elif re.search(r"\bsecond semester\b", t):
        sem = "S2"
    return {"categories": cats, "avoid_heavy_coding": bool(AVOID_CODING_RE.search(t)), "semester": sem}


def parse_finished(finished):
    """['COMP 2411', 'comp2421'] or 'COMP2411 COMP2421' -> ['COMP2411', 'COMP2421'] (no duplicates)."""
    text = " ".join(finished) if isinstance(finished, (list, tuple)) else str(finished or "")
    text = re.sub(r"([A-Za-z]{3,4})\s+(\d)", r"\1\2", text.upper())
    out = []
    for code in CODE_RE.findall(text):
        if code not in out:
            out.append(code)
    return out


# ---------------------------------------------------------------- LLM
def installed_models(timeout=5):
    """Names of the models pulled in the local Ollama, [] if Ollama is not reachable."""
    try:
        with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=timeout) as r:
            return [m.get("name", "") for m in json.loads(r.read().decode("utf-8")).get("models", [])]
    except Exception:
        return []


_RESOLVED_MODEL = None


def resolve_model():
    """Model to use: OLLAMA_MODEL if set, else the first installed MODEL_FALLBACKS entry.

    Different teammates have different models pulled (qwen2.5:3b vs qwen3.5:9b ...), so a hard-coded
    name 404s on another machine and the advisor would silently drop to the template answer.
    """
    global _RESOLVED_MODEL
    if OLLAMA_MODEL:
        return OLLAMA_MODEL
    if _RESOLVED_MODEL is None:
        models = installed_models()
        _RESOLVED_MODEL = next((m for m in MODEL_FALLBACKS if m in models), models[0] if models else "")
    return _RESOLVED_MODEL


def _generate(payload, timeout):
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")).get("response", "").strip()


def call_ollama(prompt, timeout=90):
    payload = {"model": resolve_model(), "prompt": prompt, "stream": False,
               "options": {"temperature": 0.2, "num_predict": 300}}
    try:
        # qwen3.x "thinks" by default and then returns an EMPTY response (all text is in the
        # separate "thinking" field), so the guardrail would always reject it. Ask for no thinking.
        return _generate({**payload, "think": False}, timeout)
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        # older Ollama / non-thinking models may reject the "think" option -> retry without it
        return _generate(payload, timeout)


def llm_answer_is_safe(answer, result, courses):
    """Guardrail: only allowed course codes, and every picked course must be mentioned."""
    if len(answer) < 40:
        return False
    allowed = {c["course_code"] for c in result["candidates"]} | set(result["profile"]["completed"])
    for code in result["picked"]:  # prerequisite alternatives written in the form, e.g. COMP2011/COMP2013
        allowed |= set(CODE_RE.findall(courses[code].get("prerequisite_text", "").upper()))
    mentioned = set(CODE_RE.findall(answer.upper()))
    return not (mentioned - allowed) and set(result["picked"]) <= mentioned


# ---------------------------------------------------------------- answer text
def _offered_text(off):
    if off is None:
        return "offering not checked"
    sems = [s for s in ("S1", "S2") if off.get(s)]
    return " and ".join(sems) if sems else "not offered"


def _why(course, prefs, result):
    parts = []
    if prefs["categories"] and course["category"] in prefs["categories"]:
        parts.append(f"matches your preferred area ({course['category']})")
    pt = course.get("prerequisite_text", "")
    parts.append("no prerequisite needed" if not course.get("prerequisite") else f"prerequisite satisfied ({pt})")
    if prefs["avoid_heavy_coding"] and course["coding_load"] != "high":
        parts.append("not a heavy-coding course")
    return "; ".join(parts)


def template_answer(result, courses, prefs, snippets):
    lines = [f"*{DISCLAIMER}*", ""]
    if result["picked"]:
        lines.append("### Recommended courses for you")
        for i, code in enumerate(result["picked"], 1):
            c = courses[code]
            a = c["assessment"]
            lines += [
                f"{i}. **{code} {c['title']}** ({c['credit']} credits)",
                f"- Area: {c['category']} | Coding load: {c['coding_load']} | Offered 2026/27: {_offered_text(c.get('offered_2026_27'))}",
                f"- Why: {_why(c, prefs, result)}",
                f"- Assessment: exam {a['exam_pct']}% / coursework {a['continuous_pct']}%",
                f"- About: {c['summary']}",
            ]
            if code in snippets:
                lines.append(f"- Syllabus excerpt: {snippets[code]}")
            lines.append("")
    else:
        lines += ["### No suitable course found",
                  "No elective matches your finished courses and preferences. Try a different area, "
                  "another semester, or remove the 'avoid heavy coding' wish.", ""]
    return lines


def _footer(result, courses):
    lines = [f"**Total suggested credit: {result['total_credit']} / Your remaining credit: "
             f"{result['profile']['remaining_credit']}**"]
    others = [c["course_code"] for c in result["candidates"] if c["course_code"] not in result["picked"]]
    if others:
        lines.append(f"Also eligible but not picked (credit limit): {', '.join(others)}")
    cats = result["profile"]["categories"]  # only explain courses in the areas the student asked for
    useful = [(k, v) for k, v in result["excluded"].items()
              if not v.startswith(("not offered", "category ", "level "))
              and (not cats or courses[k]["category"] in cats)][:5]
    if useful:
        lines.append("")
        lines.append("Left out because:")
        lines += [f"- {k} {courses[k]['title']}: {v}" for k, v in useful]
    for w in result["warnings"]:
        lines.append(f"\n*Note: {w}*")
    lines.append("\n> Course details come from the official PolyU COMP Subject Description Forms.")
    return lines


# ---------------------------------------------------------------- main entry for app.py
def academic_advisor_recommend(finished_courses, remaining_credit, preference_text, semester=None, rag_fn=None):
    """Returns {'answer': str, 'source_files': list[str], ...extra keys are ignored by the UI...}.

    semester: 'S1' / 'S2' / None. If None, we try to read it from the preference text.
    rag_fn:   optional callable(query:str)->{'answer':str,'source_files':list[str]} from teammate A's RAG.
    """
    courses = load_courses(COURSES_PATH)  # reloaded each call, so edits to courses.json take effect
    prefs = parse_preferences(preference_text)
    if semester:
        prefs["semester"] = semester.upper()
    done = parse_finished(finished_courses)
    result = recommend(courses, done, int(remaining_credit), prefs["categories"],
                       prefs["avoid_heavy_coding"], prefs["semester"])

    snippets, rag_sources = {}, []
    if rag_fn:
        for code in result["picked"]:
            try:
                r = rag_fn(f"Tell me about {code}")
                if r.get("source_files"):  # skip "not found" answers
                    snippets[code] = " ".join(r["answer"].split())[:400]
                    rag_sources += r["source_files"]
            except Exception:
                pass

    answer, used_llm = None, False
    if USE_LLM and result["picked"]:
        try:
            raw = call_ollama(build_llm_prompt(result, preference_text, snippets))
            if llm_answer_is_safe(raw, result, courses):
                head = raw if raw.lstrip().lower().startswith("this is only a suggestion") else f"*{DISCLAIMER}*\n\n{raw}"
                answer, used_llm = "\n".join([head, ""] + _footer(result, courses)), True
        except Exception:
            pass  # Ollama not running / timeout -> template
    if answer is None:
        answer = "\n".join(template_answer(result, courses, prefs, snippets) + _footer(result, courses))

    if rag_sources:
        sources = list(dict.fromkeys(rag_sources))
    else:
        sources = [f"[{c} Subject Description Form]({courses[c]['source']})" for c in result["picked"]]
    return {"answer": answer, "source_files": sources, "used_llm": used_llm,
            "picked": result["picked"], "excluded": result["excluded"], "preferences": prefs}


if __name__ == "__main__":
    import sys
    fin = sys.argv[1] if len(sys.argv) > 1 else "COMP1011, COMP2011, COMP2411, COMP2421"
    cr = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    pref = sys.argv[3] if len(sys.argv) > 3 else "I prefer AI-focused courses and want to avoid heavy coding coursework."
    sem = sys.argv[4] if len(sys.argv) > 4 else None
    out = academic_advisor_recommend(fin, cr, pref, sem)
    print(out["answer"])
    print("\nSources:", out["source_files"], "| LLM used:", out["used_llm"])
