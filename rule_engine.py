"""Rule engine for the COMP course-advisor chatbot (COMP4431 group project).

The engine applies the HARD rules (prerequisites, exclusions, offering semester,
credit limit, preferences) in plain Python. The LLM only explains the result,
so it cannot recommend a course the student is not allowed to take.

Data comes from courses.json at run time, so updating the course list needs no code change.

Prerequisite format in courses.json: a list of groups. Every group must be satisfied (AND);
a group is satisfied if ANY code in it is completed (OR).
    [["COMP2011","COMP2013","DSAI2201"], ["COMP2411"]]  = (2011 or 2013 or DSAI2201) and 2411

Quick use:
    from rule_engine import load_courses, recommend
    courses = load_courses("courses.json")
    result = recommend(courses, completed=["COMP1011", "COMP2011"], remaining_credit=12,
                       categories=["AI"], avoid_heavy_coding=True, semester="S1")

Command line:
    python rule_engine.py --completed COMP1011 COMP2011 --credit 12 --categories AI --avoid-heavy-coding --semester S1
"""
import argparse
import json
import re
from pathlib import Path

CODE_RE = re.compile(r"[A-Z]{3,4}\d[A-Z0-9]{3,4}")
LOAD_RANK = {"low": 0, "medium": 1, "high": 2}

DISCLAIMER = "This is only a suggestion. Please refer to official academic advising."


def norm(code):
    return str(code).strip().upper().replace(" ", "")


def load_courses(path=None):
    """Return {course_code: course_dict} from courses.json."""
    path = Path(path) if path else Path(__file__).with_name("courses.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {c["course_code"]: c for c in data}


def expand_completed(courses, completed):
    """Normalise codes and add equivalent codes (e.g. DSAI4203 <-> COMP4432)."""
    done = {norm(c) for c in completed}
    for code, c in courses.items():
        for eq in c.get("equivalent_codes", []):
            if code in done:
                done.add(eq)
            if eq in done:
                done.add(code)
    return done


def missing_prereq_groups(course, done):
    """Prerequisite groups that are NOT yet satisfied, as 'A/B/C' strings."""
    return ["/".join(g) for g in course.get("prerequisite", []) if not any(p in done for p in g)]


def exclusion_hits(course, done):
    """Completed codes that the course lists as exclusions."""
    codes = set(CODE_RE.findall(str(course.get("exclusion", "")).upper()))
    codes.discard(course["course_code"])
    return sorted(codes & done)


def _all_prereq_codes(courses):
    return {p for c in courses.values() for g in c.get("prerequisite", []) for p in g}


def recommend(courses, completed, remaining_credit, categories=None, avoid_heavy_coding=False,
              semester=None, electives_only=True, include_not_offered=False,
              min_level=None, max_level=None):
    """Apply the hard rules. Returns a dict with candidates, picked plan, and exclusion reasons.

    semester: "S1", "S2" or None. With None, subjects offered in neither semester are still dropped
              (unless include_not_offered=True) but a subject offered in only one semester is kept.
    """
    sem = semester.upper() if semester else None
    if sem not in (None, "S1", "S2"):
        raise ValueError("semester must be 'S1', 'S2' or None")
    cats = [categories] if isinstance(categories, str) else list(categories or [])
    cats_l = [c.lower() for c in cats]

    done = expand_completed(courses, completed)
    warnings = []
    known = set(courses) | _all_prereq_codes(courses)
    for code in sorted(done):
        if code not in known and not any(code in c.get("equivalent_codes", []) for c in courses.values()):
            warnings.append(f"Unknown course code '{code}' (not in the dataset); used as-is for prerequisite checks.")
    if sem is None:
        warnings.append("No semester given: a subject offered in only one semester may not be available in yours.")

    excluded, candidates = {}, []
    for code, c in courses.items():
        if code in done:
            continue  # already completed (silent)
        if electives_only and c.get("elective_status") != "elective":
            continue  # not an elective (silent)
        if c.get("non_elective_candidate"):
            excluded[code] = "not a regular elective (capstone/internship/service-learning)"
            continue
        if c.get("credit_type") == "training":
            excluded[code] = "training credits, not normal credits"
            continue
        if min_level is not None and c["level"] < min_level:
            excluded[code] = f"level {c['level']} is below the minimum level {min_level}"
            continue
        if max_level is not None and c["level"] > max_level:
            excluded[code] = f"level {c['level']} is above the maximum level {max_level}"
            continue

        off = c.get("offered_2026_27")
        flags = []
        if off is None:
            flags.append("offering_unknown")
        elif sem and not off.get(sem):
            excluded[code] = f"not offered in {sem} (2026/27)"
            continue
        elif sem is None and not (off.get("S1") or off.get("S2")) and not include_not_offered:
            excluded[code] = "not offered in 2026/27"
            continue

        hits = exclusion_hits(c, done)
        if hits:
            excluded[code] = "excluded because you completed " + ", ".join(hits)
            continue
        missing = missing_prereq_groups(c, done)
        if missing:
            excluded[code] = "missing prerequisite: " + " AND ".join(missing)
            continue
        if cats_l and c["category"].lower() not in cats_l:
            excluded[code] = f"category {c['category']} is not in your preferred areas"
            continue
        if avoid_heavy_coding and c.get("coding_load") == "high":
            excluded[code] = "heavy coding load (you asked to avoid it)"
            continue
        if not c.get("prerequisite") and c.get("prerequisite_text", "/ (none)").strip() not in ("/ (none)", "Nil", ""):
            flags.append("check_prerequisite_text")  # text-only prerequisite the engine cannot check
        candidates.append((c, flags))

    def sort_key(item):
        c, _ = item
        pref = cats_l.index(c["category"].lower()) if cats_l else 0
        load = LOAD_RANK.get(c.get("coding_load"), 1) if avoid_heavy_coding else 0
        return (pref, load, c["course_code"])

    candidates.sort(key=sort_key)

    picked, total = [], 0
    for c, _ in candidates:
        if total + c["credit"] <= remaining_credit:
            picked.append(c["course_code"])
            total += c["credit"]

    return {
        "profile": {"completed": sorted(done), "remaining_credit": remaining_credit, "categories": cats,
                    "avoid_heavy_coding": avoid_heavy_coding, "semester": sem},
        "candidates": [{
            "course_code": c["course_code"], "title": c["title"], "credit": c["credit"], "level": c["level"],
            "category": c["category"], "coding_load": c["coding_load"], "offered_2026_27": c.get("offered_2026_27"),
            "flags": flags, "source": c.get("source"),
        } for c, flags in candidates],
        "picked": picked,
        "total_credit": total,
        "excluded": excluded,
        "excluded_category": {k: courses[k]["category"] for k in excluded},
        "warnings": warnings,
    }


def candidate_codes(courses, completed, remaining_credit, **prefs):
    """Public contract of the rule engine: the filtered plan as course codes ONLY.

    No titles, no explanations, no LLM output - so other modules (advisor, UI, tests) can depend on
    a stable, testable interface. Everything else in recommend() is diagnostic detail assembled
    AFTER the decision has been made; the decision itself is made only here.
    """
    return recommend(courses, completed, remaining_credit, **prefs)["picked"]


def build_llm_prompt(result, user_text="", syllabus_snippets=None):
    """Prompt for the local LLM. It may only talk about the engine's picks.

    syllabus_snippets: optional {course_code: text} from the RAG module (teammate A).
    """
    snippets = syllabus_snippets or {}
    by_code = {c["course_code"]: c for c in result["candidates"]}
    lines = []
    for code in result["picked"]:
        c = by_code[code]
        lines.append(f"- {code} {c['title']} ({c['credit']} credits, area: {c['category']}, "
                     f"coding load: {c['coding_load']}, source: {c['source']})")
        if code in snippets:
            lines.append(f"  Syllabus excerpt: {snippets[code]}")
    other = [code for code in by_code if code not in result["picked"]]
    p = result["profile"]
    # Keep the prompt short for a small local model: drop the noisy reasons (not offered / other category)
    # and cap the rest. The full list stays available in result["excluded"].
    cats = p["categories"]
    useful = [(k, v) for k, v in result["excluded"].items()
              if not v.startswith(("not offered", "category ", "level "))
              and (not cats or result.get("excluded_category", {}).get(k) in cats)][:8]
    head = (
        "You are a course advisor for PolyU COMP undergraduate students.\n"
        f"STUDENT: completed={p['completed']}, remaining credits={p['remaining_credit']}, "
        f"preferred areas={p['categories'] or 'any'}, avoid heavy coding={p['avoid_heavy_coding']}, "
        f"semester={p['semester'] or 'not given'}\n"
        f"STUDENT MESSAGE: {user_text}\n\n"
    )
    if result["picked"]:
        # A list of left-out courses and totals is appended by the program afterwards, so the model
        # only writes a short explanation of the recommended courses (small models copy long prompts).
        return (
            head
            + "RECOMMENDED (already checked by the rule engine):\n" + "\n".join(lines) + "\n\n"
            "TASK: Write 3 to 5 short sentences in plain English explaining why these recommended courses "
            "fit this student. Mention each recommended course code once and its source link. "
            "Mention ONLY the recommended course codes. Do not mention any other course code, do not copy "
            "the lists above, do not invent credits or prerequisites. "
            f"Start with exactly: \"{DISCLAIMER}\""
        )
    return (
        head
        + "RECOMMENDED: (none)\n\nEXCLUSION REASONS (selected):\n"
        + ("\n".join(f"- {k}: {v}" for k, v in useful) or "(none)") + "\n\n"
        "TASK: In 2 to 4 short sentences say no suitable course was found and explain using the exclusion "
        f"reasons only. Start with exactly: \"{DISCLAIMER}\""
    )


def main():
    ap = argparse.ArgumentParser(description="COMP course advisor rule engine")
    ap.add_argument("--courses", default=None, help="path to courses.json (default: next to this file)")
    ap.add_argument("--completed", nargs="*", default=[], help="completed course codes")
    ap.add_argument("--credit", type=int, required=True, help="remaining credits")
    ap.add_argument("--categories", nargs="*", default=[], help="e.g. AI Data System Security")
    ap.add_argument("--avoid-heavy-coding", action="store_true")
    ap.add_argument("--semester", choices=["S1", "S2"], default=None)
    ap.add_argument("--all-subjects", action="store_true", help="do not restrict to electives")
    args = ap.parse_args()
    res = recommend(load_courses(args.courses), args.completed, args.credit, args.categories,
                    args.avoid_heavy_coding, args.semester, electives_only=not args.all_subjects)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
