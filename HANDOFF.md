# COMP4431 Group Project – Backend B Handoff (PolyU COMP Course Advisor Chatbot)

Written 2026-11-14 for teammates and their AI assistants. Read this first.

## 1. Project
- Course: COMP4431 AI group project (10% of marks). Domain-specific chatbot, group of 2-4.
- Deadline: 18 Nov 2026 23:59 (Thu class) / 19 Nov 2026 23:59 (Fri class). One member submits on Canvas: zip (source + README with all student IDs and names, additional features, how to run, prompts/knowledge-base files) + .pptx.
- Live demo in English: 7 min + 3 min Q&A (19/26 Nov Thu or 20/27 Nov Fri).
- Grading: technical contribution, UI design, creativity, presentation quality. Basic features 3/10, additional features 7/10.
- Domain: university course assistant for PolyU COMP (Scheme in Computing and AI).
- No cloud API available -> local LLM through Ollama.
- Basic feature: RAG course lookup (teammate A). Additional feature: **Academic Advisor** = structured JSON + Rule Engine + RAG + LLM (this handoff, backend B).
- Roles: A = RAG backend, B = courses.json / rule engine / advisor / evaluation, C = Streamlit front-end (app.py), D = slides / README.

## 2. Architecture (Academic Advisor)
```
user text + finished courses + remaining credits + semester
        -> advisor.parse_preferences (keywords -> area, avoid-heavy-coding, semester)
        -> rule_engine.recommend (hard rules on courses.json)
        -> [optional] RAG snippets from teammate A for the picked courses
        -> Ollama LLM writes a short explanation (rule_engine.build_llm_prompt)
        -> guardrail llm_answer_is_safe; if it fails or Ollama is down -> deterministic template answer
        -> footer (total credit, left-out reasons, warnings) appended by program
```
Principle: the rule engine decides; the LLM only words the explanation. The LLM can never add a course.

## 3. Files (all in one folder)
| File | Purpose |
|---|---|
| courses.json | 81 PolyU COMP courses (Level 1-4), the source of truth |
| rule_engine.py | prerequisite / exclusion / credit / category / semester logic, `recommend()`, `build_llm_prompt()`, CLI |
| advisor.py | `academic_advisor_recommend(...)`: preference parsing, Ollama call, guardrail, template, sources |
| app.py | teammate C's Streamlit front-end, patched to call advisor.py (53 lines changed, CRLF kept) |
| evaluate.py | 50 automated checks, writes eval_results.json |
| build_courses.py | generator for courses.json (data may also be edited in the JSON directly) |
| rag_backend.py | **basic feature**: `rag_course_query()` - PDF -> Chroma -> Ollama syllabus lookup |
| knowledge_base/ | 10 official Subject Description Form PDFs (the RAG knowledge source) |
| test_rag_backend.py | offline tests for rag_backend.py (`RAG_E2E=1` adds real index/query tests) |

## 4. How to run (Windows)
```powershell
cd "C:\Users\user\Documents\AI project"
python -m pip install streamlit
python evaluate.py                      # expect 43/43 passed
python -m streamlit run app.py          # http://localhost:8501
```
Ollama (optional, template fallback if absent):
```powershell
ollama pull qwen2.5:3b        # or any model from advisor.MODEL_FALLBACKS
# optional: $env:OLLAMA_MODEL="qwen3.5:9b"; $env:USE_LLM="0" to force template
```
Env vars: `USE_LLM` (default 1), `OLLAMA_URL` (default http://localhost:11434/api/generate), `OLLAMA_TAGS_URL` (default: same host + `/api/tags`), `OLLAMA_MODEL` (default: auto — the first installed model from `MODEL_FALLBACKS = qwen2.5:3b, qwen2.5:7b, qwen3.5:4b, qwen3.5:9b`). Requests are sent with `think: false`: qwen3.x otherwise puts its whole answer in the `thinking` field and returns an empty `response`, which the guardrail rejects.

## 5. courses.json schema
Fields per course: course_code, title, credit, credit_type ("credit"/"training"), elective_status ("elective"/"not_in_elective_list"), non_elective_candidate, level, prerequisite, prerequisite_text, exclusion, category (AI, System, Security, Theory, Application, Data, Foundation, Business, FinTech, Professional, Other), coding_load (low/medium/high), coding_load_basis, assessment {exam_pct, continuous_pct, continuous_detail}, offered_2026_27 (null or {"S1","S2"}), offered_note, equivalent_codes, summary, form_date, source (Subject Description Form PDF URL), note.

- `prerequisite` = list of groups: AND across groups, OR inside a group. Example `[["COMP2011","COMP2013","DSAI2201"],["COMP2411"]]`.
- Equivalent codes: DSAI4201<->COMP4136, DSAI4202<->COMP4142, DSAI4203<->COMP4432, DSAI4204<->COMP4433, DSAI4205<->COMP4434, DSAI4206<->COMP4531, DSAI2201<->COMP2013.
- Counts: 81 entries (L1 13, L2 11, L3 25, L4 32); 39 flagged elective.
- Not offered in 2026/27 (electives): COMP3131, 3133, 3233, 3911, 4125, 4135, 4146, 4332, 4424.
- Not in file: COMP2S01, COMP3921-23 (page could not be fetched), COMP4921 (no title), COMP3901-3903 (transfer credits), DSAI4201/4203/4204/4205 (recorded as equivalents).
- Non-3-credit: COMP1002=4, 1003=1, 1004=2, 3511=2, 4913=6, 4001/4002=12 training (training credits are never recommended).
- To update the course list: edit courses.json directly (or build_courses.py and re-run `python -I build_courses.py courses.json`), then run `python evaluate.py`. advisor.py reloads courses.json on every call.

## 5b. Main functions
- `rule_engine.recommend(...)` -> {profile, candidates, picked, total_credit, excluded, excluded_category, warnings}.
- `rule_engine.candidate_codes(courses, completed, remaining_credit, **prefs)` -> `["COMP4423", ...]` — the clean interface: filtered course codes only, no titles or text.
- `rule_engine.build_llm_prompt(result, user_text="", syllabus_snippets=None)` -> prompt string (pure string assembly over the engine's result; it never decides eligibility).
- `advisor.academic_advisor_recommend(finished_courses, remaining_credit, preference_text, semester=None, rag_fn=None)` -> {answer, source_files, used_llm, picked, excluded, preferences}.
- In app.py set `USE_RAG_SNIPPETS = True` once teammate A's real `rag_course_query` is connected (it should return {"answer": ..., "source_files": [...]}).

## 6. Status
Done:
- courses.json (81 courses, official data from the Subject Description Forms), rule engine, advisor, front-end integration.
- evaluate.py: 50/50 pass (data 7, unit 14, student profiles 9, advisor 15, LLM plumbing 5).
- Demo case: finished COMP1011, COMP2011, COMP2411, COMP2421; 12 credits; AI; avoid heavy coding; S2 -> recommends COMP4423 + COMP4436 (COMP4431/COMP4432 rejected for heavy coding, COMP2014 for missing maths prerequisite).
- Ollama is installed on the dev machine with qwen3.5:4b / qwen3.5:9b / qwen2.5-coder:7b (qwen2.5:3b is NOT pulled), so the model name is resolved at run time.

Latest fix (2026-11-18): the LLM path was silently dead on any machine whose installed model differs from the author's.
1. The hard-coded default `qwen2.5:3b` returned HTTP 404 and the advisor quietly fell back to the template. `resolve_model()` now uses `OLLAMA_MODEL` when set, else the first installed model from `MODEL_FALLBACKS`.
2. qwen3.x models "think" by default: the answer arrives in the `thinking` field and `response` is empty, so the guardrail rejected every answer. `call_ollama` now sends `think: false` (retried without the flag on HTTP 400, for older Ollama) and caps `num_predict` at 300.
Verified end to end with qwen3.5:4b (auto-selected): `used_llm=True`, `picked=['COMP4423','COMP4436']`, `SAFE: True`. Regression tests are section 5 of evaluate.py, so they run without Ollama:
```powershell
python advisor.py      # demo student; prints the answer and "LLM used: True"
python evaluate.py     # expect 50/50
```

## 7. Known issues / TODO
1. ~~Re-test the Ollama path~~ done 2026-11-18 (see section 6). Real Streamlit UI was never run by the author (only a stubbed headless test) – run it once and report errors.
2. The mock `rag_course_query` inside app.py still returns wrong COMP4431 facts (prerequisite COMP2411, assessment 10/40/50). `rag_backend.py` now provides the real one with the same return shape - replace the mock with `from rag_backend import rag_course_query` (see section 9).
3. `coding_load` tags are the author's own judgement, not from official forms – group should review.
4. Confirm the elective list on the PolyU website has no earlier page before COMP2014 (possible missing Level 1-2 electives).
5. DSAI-code offering info comes from a page whose academic year was not visible.
6. Still to write: README (student IDs and names, additional features, how to run), slides (technical challenges, limitations, demo script), ~12 basic-feature lookup test queries for evaluation, agree the RAG interface with teammate A.
7. Disclaimer shown to users: "This is only a suggestion. Please refer to official academic advising."

## 8. Notes for an AI assistant continuing this work
- Do not let the LLM decide eligibility; change rules in rule_engine.py / data in courses.json.
- Keep the guardrail: LLM answers may only mention allowed course codes and must mention every picked course; otherwise use the template.
- After any change run `python evaluate.py` (all 50 should pass) and keep app.py line endings (CRLF).
- Source for course data: https://www.polyu.edu.hk/comp/study/ug-programmes/scheme-in-computing-and-ai/subjects/ and the Subject Description Form PDFs linked in each course's `source`.

## 9. Basic feature: course-lookup RAG (rag_backend.py, member A)
Built 2026-11-18; it did not exist in the repository before.

- `rag_backend.rag_course_query(course_code)` -> `{"answer": str, "source_files": list[str]}`.
  Accepts a bare code ("COMP4431") or a question containing one ("What is the assessment of
  COMP4431?"). **Never raises**: unknown course, empty knowledge base, missing library or Ollama
  down all return `{"answer": "", "source_files": []}`.
- Pipeline: `knowledge_base/*.pdf` -> pypdf text per page -> paragraph chunks (900 chars, 150
  overlap) -> sentence-transformers `all-MiniLM-L6-v2` (normalised) -> Chroma in `chroma_db/` ->
  top-4 chunks **of the asked course** (metadata filter) -> Ollama answer. If Ollama is down the
  answer is a deterministic syllabus excerpt instead, so the basic demo still works offline.
- Persistence: `chroma_db/manifest.json` stores a fingerprint (PDF names + sizes + mtimes + chunk
  settings). Unchanged PDFs -> the index is reused; new/edited PDFs -> rebuilt automatically on the
  next query. `python rag_backend.py --build --force` forces a rebuild.
- `knowledge_base/` holds 10 official Subject Description Forms downloaded from the `source` URLs in
  courses.json: COMP1011, 2011, 2411, 2421, 4423, 4431, 4432, 4433, 4434, 4436. Drop more PDFs in
  (one file per course, named after the code) and the next query re-indexes them.
- Run:
  ```powershell
  python rag_backend.py --build
  python rag_backend.py COMP4431
  python rag_backend.py --ask "What is the assessment scheme of COMP4432?"
  python test_rag_backend.py                 # offline, no Ollama needed
  $env:RAG_E2E="1"; python test_rag_backend.py   # also builds/queries the real index
  ```
- Dependencies `chromadb`, `sentence-transformers`, `pypdf` are in requirements.txt. Python 3.10-3.12
  is recommended (torch wheels); the dev venv `.venv-rag` was created for this and must NOT be zipped.
- Env vars: `RAG_KB_DIR`, `RAG_DB_DIR`, `RAG_EMBED_MODEL`, `RAG_TOP_K`, `RAG_LLM=0` (quote the
  syllabus text only), `RAG_OLLAMA_URL`, `RAG_OLLAMA_MODEL` (default: auto, same fallback list as
  advisor.py).
- Bug fixed during the build: the form header "Jul 2025" matched the course-code regex, so 46 of the
  70 chunks were filed under JUL2025/JUL2022 instead of their course and per-course retrieval
  returned nothing. The index now trusts the PDF filename; only an explicit `Subject Code` line in
  the page overrides it. `test_rag_backend.py` guards both the month-year case and the
  "metadata must match the filename" invariant.
- Wiring to the UI (not done yet, on purpose): replace the mock `rag_course_query` in app.py with
  `from rag_backend import rag_course_query`; the return shape is identical. Set
  `USE_RAG_SNIPPETS = True` in app.py only after that, to feed syllabus excerpts into the advisor.
