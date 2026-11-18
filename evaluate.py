"""Evaluation for the course-advisor rule engine.  Run:  python evaluate.py

Three parts:
  1. DATA CHECKS   - sanity checks on courses.json
  2. UNIT TESTS    - prerequisite/exclusion/equivalence/credit rules on a tiny synthetic dataset
  3. PROFILE TESTS - simulated students. The expected results were worked out BY HAND from the
                     course data (not copied from the engine's output), so a mismatch means a bug
                     in the engine, a wrong expectation, or an error in courses.json.

NOTE: the PROFILE TESTS with 'exact' sets depend on courses.json. If you change prerequisites,
offering semesters or the elective list, re-derive those expectations by hand.
Writes eval_results.json (for the slides: pass rate and the failing cases).
"""
import json
import sys
from pathlib import Path

from rule_engine import candidate_codes, load_courses, recommend

HERE = Path(__file__).parent
courses = load_courses(HERE / "courses.json")
results = {"data_checks": [], "unit_tests": [], "profile_tests": [], "advisor_tests": [], "llm_tests": []}


def report(section, name, ok, detail=""):
    results[section].append({"name": name, "passed": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail and not ok else ""))


# ---------------------------------------------------------------- 1. data checks
def data_checks():
    print("\n1. DATA CHECKS")
    cs = list(courses.values())
    report("data_checks", "course codes are unique", len(cs) == len({c['course_code'] for c in cs}))
    bad = [c["course_code"] for c in cs if c["assessment"]["exam_pct"] + c["assessment"]["continuous_pct"] != 100]
    report("data_checks", "assessment percentages sum to 100", not bad, str(bad))
    bad = [c["course_code"] for c in cs if not all(isinstance(g, list) and g for g in c["prerequisite"])]
    report("data_checks", "prerequisites are non-empty groups", not bad, str(bad))
    bad = [c["course_code"] for c in cs if c["elective_status"] not in ("elective", "not_in_elective_list", "unconfirmed")]
    report("data_checks", "elective_status has a valid value", not bad, str(bad))
    bad = [c["course_code"] for c in cs if c["coding_load"] not in ("low", "medium", "high")]
    report("data_checks", "coding_load is low/medium/high", not bad, str(bad))
    bad = [c["course_code"] for c in cs if c["offered_2026_27"] is not None
           and set(c["offered_2026_27"]) != {"S1", "S2"}]
    report("data_checks", "offered_2026_27 is null or {S1,S2}", not bad, str(bad))
    n_el = sum(c["elective_status"] == "elective" for c in cs)
    report("data_checks", "39 electives recorded (from the programme elective list)", n_el == 39, f"found {n_el}")


# ---------------------------------------------------------------- 2. unit tests
def mk(code, prereq=None, exclusion="", credit=3, ctype="credit", eq=None, off=None, cat="AI", load="low", level=3):
    return {"course_code": code, "title": code, "credit": credit, "credit_type": ctype, "level": level,
            "prerequisite": prereq or [], "prerequisite_text": "", "exclusion": exclusion, "category": cat,
            "coding_load": load, "elective_status": "elective", "non_elective_candidate": False,
            "offered_2026_27": off, "equivalent_codes": eq or [], "source": ""}


def cand(res):
    return {c["course_code"] for c in res["candidates"]}


def unit_tests():
    print("\n2. UNIT TESTS (synthetic data)")
    m = {c["course_code"]: c for c in [
        mk("T1", prereq=[["X", "Y"], ["Z"]]),
        mk("T2", exclusion="ZZZZ1001, ZZZZ1002"),  # realistic code shape, like COMP2013, DSAI2201
        mk("T3", prereq=[["P9"]]),
        mk("P9", eq=["D9"]),
        mk("T4", credit=3), mk("T5", credit=3), mk("T6", credit=3),
        mk("TR", credit=12, ctype="training"),
        mk("S1ONLY", off={"S1": True, "S2": False}),
        mk("NEVER", off={"S1": False, "S2": False}),
    ]}
    r = lambda done, **kw: recommend(m, done, kw.pop("credit", 99), **kw)
    report("unit_tests", "prereq AND/OR: (X or Y) and Z satisfied", "T1" in cand(r(["Y", "Z"])))
    report("unit_tests", "prereq AND/OR: missing group Z blocks", "T1" not in cand(r(["X"])))
    res = r(["Z"])
    report("unit_tests", "missing group is explained", res["excluded"].get("T1") == "missing prerequisite: X/Y",
           str(res["excluded"].get("T1")))
    report("unit_tests", "exclusion blocks the course", "T2" not in cand(r(["ZZZZ1002"])))
    report("unit_tests", "exclusion does not block when not completed", "T2" in cand(r([])))
    report("unit_tests", "equivalent code D9 satisfies prerequisite P9", "T3" in cand(r(["D9"])))
    report("unit_tests", "equivalent code D9 counts as completed P9", "P9" not in cand(r(["D9"])))
    report("unit_tests", "training credits are never recommended", "TR" not in cand(r([])))
    res = recommend(m, [], 6)
    report("unit_tests", "credit cap respected (cap 6 -> at most 6)", res["total_credit"] <= 6, str(res["total_credit"]))
    report("unit_tests", "semester filter drops S1-only course in S2", "S1ONLY" not in cand(r([], semester="S2")))
    report("unit_tests", "course offered in neither semester is dropped", "NEVER" not in cand(r([])))
    report("unit_tests", "unknown offering (None) is kept with a flag",
           any(c["course_code"] == "T4" and "offering_unknown" in c["flags"] for c in r([])["candidates"]))
    codes = candidate_codes(m, [], 6)
    report("unit_tests", "candidate_codes(): codes only, credit cap respected",
           isinstance(codes, list) and all(isinstance(x, str) for x in codes) and len(codes) <= 2
           and sum(m[x]["credit"] for x in codes) <= 6, str(codes))
    res = r(["ZZZZ9999"])  # completed course that has no record in the metadata
    report("unit_tests", "completed code missing from the metadata: warned, not fatal",
           any("ZZZZ9999" in w for w in res["warnings"]) and isinstance(res["picked"], list),
           str(res["warnings"]))


# ---------------------------------------------------------------- 3. profile tests
BASIC = ["COMP1011", "COMP2011", "COMP2411"]
CASES = [
    dict(name="P1 AI student, avoids heavy coding, S2",
         args=dict(completed=BASIC + ["COMP2421"], remaining_credit=12, categories=["AI"],
                   avoid_heavy_coding=True, semester="S2"),
         exact={"COMP4423", "COMP4436"}, not_in={"COMP4431", "COMP4432"}),
    dict(name="P2 teammate's original sample (only COMP2411, COMP2421): AI courses are blocked",
         args=dict(completed=["COMP2411", "COMP2421"], remaining_credit=12, categories=["AI"]),
         exact={"COMP4432"}, not_in={"COMP4431", "COMP4423", "COMP4436"}),
    dict(name="P3 S1 planning, no preference",
         args=dict(completed=BASIC, remaining_credit=9, semester="S1"),
         exact={"COMP3011", "COMP3134", "COMP3335", "COMP3421", "COMP3424", "COMP4122", "COMP4123",
                "COMP4422", "COMP4423", "COMP4431", "COMP4432", "COMP4433"},
         not_in={"COMP4342", "COMP4322", "COMP4434"}, max_credit=9),
    dict(name="P4 completed DSAI4203 (= COMP4432): not recommended again",
         args=dict(completed=BASIC + ["DSAI4203"], remaining_credit=6, categories=["AI"], semester="S1"),
         exact={"COMP4431", "COMP4423"}, not_in={"COMP4432"}),
    dict(name="P5 credit cap: only 3 credits left -> exactly one course",
         args=dict(completed=BASIC, remaining_credit=3, semester="S1"),
         picked_count=1, max_credit=3),
    dict(name="P6 S2 planning, no preference",
         args=dict(completed=BASIC, remaining_credit=12, semester="S2"),
         exact={"COMP3121", "COMP3122", "COMP3235", "COMP3422", "COMP3531", "COMP4121", "COMP4127",
                "COMP4133", "COMP4136", "COMP4423", "COMP4431", "COMP4432", "COMP4436"},
         not_in={"COMP4422", "COMP4342"}),
    dict(name="P7 first-year student (nothing completed): only subjects without prerequisites",
         args=dict(completed=[], remaining_credit=12),
         exact={"COMP3121", "COMP3134", "COMP3235", "COMP3531", "COMP4122", "COMP4123", "COMP4127", "COMP4432"}),
    dict(name="P8 System track, completed OS and networking, S1",
         args=dict(completed=BASIC + ["COMP2421", "COMP2432", "COMP2322"], remaining_credit=12,
                   categories=["System"], semester="S1"),
         exact={"COMP4322", "COMP4342"}),
    dict(name="P9 DSAI2201 completed instead of COMP2011 (equivalent), AI, S1",
         args=dict(completed=["COMP1011", "DSAI2201"], remaining_credit=12, categories=["AI"], semester="S1"),
         exact={"COMP4431", "COMP4423", "COMP4432"}),
]


def profile_tests():
    print("\n3. PROFILE TESTS (expectations worked out by hand)")
    for case in CASES:
        res = recommend(courses, **case["args"])
        got, problems = cand(res), []
        if "exact" in case and got != case["exact"]:
            problems.append(f"missing={sorted(case['exact'] - got)} unexpected={sorted(got - case['exact'])}")
        if "not_in" in case and got & case["not_in"]:
            problems.append(f"should not contain {sorted(got & case['not_in'])}")
        if "max_credit" in case and res["total_credit"] > case["max_credit"]:
            problems.append(f"total credit {res['total_credit']} > {case['max_credit']}")
        if "picked_count" in case and len(res["picked"]) != case["picked_count"]:
            problems.append(f"picked {len(res['picked'])} courses, expected {case['picked_count']}")
        report("profile_tests", case["name"], not problems, "; ".join(problems))


# ---------------------------------------------------------------- 4. advisor tests
def advisor_tests():
    print("\n4. ADVISOR TESTS (preference parsing, pipeline, LLM guardrail)")
    import advisor
    A = "advisor_tests"
    p = advisor.parse_preferences
    r = p("I prefer AI‑focused courses and want to avoid heavy coding coursework.")
    report(A, "parse: AI + avoid heavy coding", r["categories"] == ["AI"] and r["avoid_heavy_coding"] and r["semester"] is None, str(r))
    r = p("Interested in security and cloud, semester 2, no heavy programming please")
    report(A, "parse: security + cloud, S2, avoid programming",
           set(r["categories"]) == {"Security", "System"} and r["avoid_heavy_coding"] and r["semester"] == "S2", str(r))
    r = p("I like data analytics and machine learning, sem 1")
    report(A, "parse: data + ML, S1, no coding wish",
           set(r["categories"]) == {"AI", "Data"} and not r["avoid_heavy_coding"] and r["semester"] == "S1", str(r))
    r = p("anything is fine")
    report(A, "parse: no preference", r == {"categories": [], "avoid_heavy_coding": False, "semester": None}, str(r))
    got = advisor.parse_finished("COMP 2411, comp2421 COMP2411")
    report(A, "parse finished courses: spaces, case, duplicates", got == ["COMP2411", "COMP2421"], str(got))

    demo = ["COMP1011", "COMP2011", "COMP2411", "COMP2421"]
    pref = "I prefer AI-focused courses and want to avoid heavy coding coursework."
    run = lambda **kw: advisor.academic_advisor_recommend(demo, 12, pref, **kw)
    old_use, old_call = advisor.USE_LLM, advisor.call_ollama
    try:
        advisor.USE_LLM = False
        out = run(semester="S2")
        report(A, "template answer: demo student gets COMP4423 + COMP4436", out["picked"] == ["COMP4423", "COMP4436"] and not out["used_llm"], str(out["picked"]))
        report(A, "template answer: disclaimer first, 2 source links", out["answer"].startswith("*This is only a suggestion") and len(out["source_files"]) == 2)
        report(A, "template answer: heavy-coding COMP4431 is left out, with reason",
               "COMP4431" not in out["picked"] and "COMP4431 Artificial Intelligence: heavy coding" in out["answer"])
        report(A, "template answer: unrelated areas are not listed as 'left out'", "Cloud" not in out["answer"] and "Theory of Computation" not in out["answer"])

        advisor.USE_LLM = True
        advisor.call_ollama = lambda prompt, timeout=90: ("This is only a suggestion. I recommend COMP4423 and COMP4436, "
                                                          "and also COMP4231 Advanced Computer System.")
        out = run(semester="S2")
        report(A, "guardrail: LLM invents a course -> fallback to template", not out["used_llm"] and "COMP4231" not in out["answer"])
        advisor.call_ollama = lambda prompt, timeout=90: "This is only a suggestion. COMP4423 is a good fit for your AI interest."
        report(A, "guardrail: LLM forgets a picked course -> fallback", not run(semester="S2")["used_llm"])
        advisor.call_ollama = lambda prompt, timeout=90: ("This is only a suggestion. COMP4423 Computer Vision fits your AI interest and COMP4436 "
                                                          "AIoT adds applications, both without heavy coding.")
        out = run(semester="S2")
        report(A, "guardrail: valid LLM answer is accepted and gets the credit footer",
               out["used_llm"] and "Total suggested credit: 6" in out["answer"])
        advisor.call_ollama = lambda prompt, timeout=90: ("This is only a suggestion. COMP4423 (prerequisite COMP1011 or COMP1012) and "
                                                          "COMP4436 both fit your AI interest.")
        report(A, "guardrail: prerequisite alternatives from the form are allowed", run(semester="S2")["used_llm"])

        def down(prompt, timeout=90):
            raise ConnectionRefusedError("Ollama is not running")
        advisor.call_ollama = down
        out = run(semester="S2")
        report(A, "LLM unreachable -> template answer, no crash", not out["used_llm"] and out["picked"] == ["COMP4423", "COMP4436"])

        calls = []
        advisor.call_ollama = lambda prompt, timeout=90: calls.append(1) or "x"
        out = advisor.academic_advisor_recommend([], 12, "I like theory and algorithms")
        report(A, "nothing eligible: clear message and the LLM is not called",
               out["picked"] == [] and "No suitable course" in out["answer"] and not calls)
    finally:
        advisor.USE_LLM, advisor.call_ollama = old_use, old_call


# ---------------------------------------------------------------- 5. llm plumbing tests
def llm_tests():
    """These run WITHOUT Ollama: they check the request we build and the fallback behaviour.

    Regression guards for two bugs seen on a machine whose models differ from the author's:
      * a hard-coded model name that is not pulled -> HTTP 404 -> silent template answer;
      * qwen3.x "thinking" models return an EMPTY 'response' unless think=False is sent.
    """
    print("\n5. LLM PLUMBING TESTS (no Ollama needed)")
    import urllib.error
    import advisor
    A = "llm_tests"
    old_model, old_resolved = advisor.OLLAMA_MODEL, advisor._RESOLVED_MODEL
    old_gen, old_installed, old_call, old_use = (advisor._generate, advisor.installed_models,
                                                 advisor.call_ollama, advisor.USE_LLM)
    try:
        advisor.OLLAMA_MODEL = ""
        advisor._RESOLVED_MODEL = None
        advisor.installed_models = lambda timeout=5: ["foo:1b", "qwen3.5:4b"]
        report(A, "model: auto-picks the first installed fallback", advisor.resolve_model() == "qwen3.5:4b",
               str(advisor.resolve_model()))
        advisor.OLLAMA_MODEL = "mine:7b"
        report(A, "model: an explicit OLLAMA_MODEL always wins", advisor.resolve_model() == "mine:7b")

        sent = []
        advisor._generate = lambda payload, timeout: sent.append(payload) or "ok"
        advisor.call_ollama("hi")
        report(A, "request: ask the model not to think (qwen3.x) and cap the length",
               sent and sent[0].get("think") is False and "num_predict" in sent[0].get("options", {}),
               str(sent[0] if sent else None))

        def rejects_think(payload, timeout):
            sent.append(payload)
            if "think" in payload:
                raise urllib.error.HTTPError("url", 400, "unknown field", {}, None)
            return "ok without think"
        advisor._generate = rejects_think
        sent.clear()
        report(A, "request: HTTP 400 on 'think' -> retry without it",
               advisor.call_ollama("hi") == "ok without think" and "think" not in sent[-1])

        advisor.call_ollama = lambda prompt, timeout=90: ""  # a thinking model that answered nothing
        advisor.USE_LLM = True
        out = advisor.academic_advisor_recommend(
            ["COMP1011", "COMP2011", "COMP2411", "COMP2421"], 12,
            "I prefer AI-focused courses and want to avoid heavy coding coursework.", semester="S2")
        report(A, "empty LLM answer -> template answer, demo still works",
               not out["used_llm"] and out["picked"] == ["COMP4423", "COMP4436"]
               and out["answer"].startswith("*This is only a suggestion"), str(out["picked"]))
    finally:
        (advisor.OLLAMA_MODEL, advisor._RESOLVED_MODEL, advisor._generate,
         advisor.installed_models, advisor.call_ollama, advisor.USE_LLM) = (
            old_model, old_resolved, old_gen, old_installed, old_call, old_use)


if __name__ == "__main__":
    data_checks()
    unit_tests()
    profile_tests()
    advisor_tests()
    llm_tests()
    allr = [t for sec in results.values() for t in sec]
    passed = sum(t["passed"] for t in allr)
    parts = ", ".join(f"{k.split('_')[0]} {sum(t['passed'] for t in v)}/{len(v)}" for k, v in results.items())
    print(f"\nSUMMARY: {passed}/{len(allr)} checks passed ({parts})")
    (HERE / "eval_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    sys.exit(0 if passed == len(allr) else 1)
