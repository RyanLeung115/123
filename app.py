# Run with: streamlit run app.py
# COMP4431 2026/27 Group Project Frontend
# Domain‑specific Chatbot: COMP Academic Consulting Chatbot
# Requirement: Avatar UI, two operation modes, source citation display
import base64
import re

import streamlit as st
import streamlit.components.v1 as components
from streamlit_mic_recorder import speech_to_text

# ====================== Page Config ======================
st.set_page_config(
    page_icon="🎓",
    layout="wide"
)

# Avatar SVG for bot (animated chat avatar, no local image needed)
BOT_AVATAR_SVG = """
<svg width="60" height="60" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="35" r="22" fill="#4285F4"/>
  <circle cx="42" cy="32" r="3" fill="white"/>
  <circle cx="58" cy="32" r="3" fill="white"/>
  <path d="M42 45 Q50 52 58 45" stroke="white" stroke-width="2.5" fill="none"/>
  <rect x="28" y="60" width="44" height="30" rx="10" fill="#4285F4"/>
</svg>
"""

# Default advisor preference, also used when the student has not spoken or typed one
DEFAULT_ADVISOR_PREFERENCE = "I prefer AI‑focused courses and want to avoid heavy coding coursework."

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_mode" not in st.session_state:
    # Two modes: "course_lookup" (basic feature), "advisor" (additional feature)
    st.session_state.current_mode = "course_lookup"

# ===================== Real Backend Interface =====================
# ---------------- Backend A (RAG pipeline, implement by member A) ----------------
from rag_backend import rag_course_query as _raw_rag_course_query

def rag_course_query(user_query: str) -> dict:
    """
    Basic Feature: Course syllabus lookup via RAG
    Input: user natural language query e.g. "Tell me about COMP4431"
    Return dict:
    {
        "answer": str,
        "source_files": list[str]
    }
    """
    res = _raw_rag_course_query(user_query.strip())
    if not res["answer"]:
        return {
            "answer": "No matching course syllabus found in knowledge base.",
            "source_files": []
        }
    return res


# ---------------- Backend B (Rule Engine + Advisor Module, member B) ----------------
# Real implementation lives in advisor.py (+ rule_engine.py + courses.json): keep them in the same folder.
from advisor import academic_advisor_recommend as _advisor

USE_RAG_SNIPPETS = False  # set True once member A's real rag_course_query() is connected

def academic_advisor_recommend(
    finished_courses: list[str],
    remaining_credit: int,
    preference_text: str,
    semester=None
) -> dict:
    """
    Additional Feature: Academic Advisor hybrid pipeline
    Rule Engine (prerequisite / semester / credit / preference) -> optional RAG -> LLM wording
    (falls back to a template answer if the local LLM is unavailable or breaks the rules)
    Return dict: {"answer": str, "source_files": list[str]}
    """
    return _advisor(finished_courses, remaining_credit, preference_text,
                    semester=semester,
                    rag_fn=rag_course_query if USE_RAG_SNIPPETS else None)


def render_speech_output(text: str, instance_id: str) -> None:
    speech_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    speech_text = re.sub(r"[*`#_>]", "", speech_text)
    encoded_text = base64.b64encode(speech_text.encode("utf-8")).decode("ascii")
    components.html(
        f"""
        <button id="speak-{instance_id}" type="button">🔊 Read answer</button>
        <button id="stop-{instance_id}" type="button">⏹ Stop</button>
        <span id="status-{instance_id}" role="status"></span>
        <script>
          const bytes = Uint8Array.from(atob("{encoded_text}"), c => c.charCodeAt(0));
          const text = new TextDecoder().decode(bytes);
          const status = document.getElementById("status-{instance_id}");
          const speakButton = document.getElementById("speak-{instance_id}");
          const stopButton = document.getElementById("stop-{instance_id}");
          if (!window.speechSynthesis) {{
            status.textContent = "Text-to-speech is not supported in this browser.";
            speakButton.disabled = true;
          }} else {{
            speakButton.addEventListener("click", () => {{
              window.speechSynthesis.cancel();
              window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
            }});
            stopButton.addEventListener("click", () => window.speechSynthesis.cancel());
          }}
        </script>
        """,
        height=42,
        scrolling=False,
    )


# ===================== UI Layout =====================
st.title("🎓 COMP Academic Consulting Chatbot")
st.markdown("**COMP4431 Group Project | Domain‑specific Chatbot**")

col_mode1, col_mode2 = st.columns([1,1])
with col_mode1:
    if st.button("📖 Course Lookup (Basic Feature)", use_container_width=True,
                 type="primary" if st.session_state.current_mode == "course_lookup" else "secondary"):
        if st.session_state.current_mode != "course_lookup":
            st.session_state.chat_history = []  # do not mix the two features' conversations
        st.session_state.current_mode = "course_lookup"
        st.rerun()
with col_mode2:
    if st.button("🧑‍🏫 Academic Advisor (Additional Feature)", use_container_width=True,
                 type="primary" if st.session_state.current_mode == "advisor" else "secondary"):
        if st.session_state.current_mode != "advisor":
            st.session_state.chat_history = []
        st.session_state.current_mode = "advisor"
        st.rerun()

st.divider()

# Show current working mode hint
if st.session_state.current_mode == "course_lookup":
    st.info("📖 Mode: Course Lookup — Input course code / course name to fetch syllabus summary, prerequisite, assessment.")
else:
    st.info("🧑‍🏫 Mode: Academic Advisor — Fill your finished courses, remaining credit and preference for course recommendation.")

# Render chat history with Avatar
for chat_index, chat in enumerate(st.session_state.chat_history):
    role = chat["role"]
    content = chat["content"]
    sources = chat.get("sources", [])
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar=BOT_AVATAR_SVG):
            render_speech_output(content, f"history-{chat_index}")
            st.markdown(content)
            if len(sources) > 0:
                st.caption(f"📎 Source reference: {', '.join(sources)}")

# ========= Input Area for different mode =========
if st.session_state.current_mode == "course_lookup":
    st.caption("Voice recognition uses the browser's speech service (internet access and browser support may be required).")
    spoken_input = speech_to_text(
        language="en-US",
        start_prompt="🎙️",
        stop_prompt="⏹ Stop",
        just_once=True,
        key="course_lookup_stt",
    )
    user_input = st.chat_input("Ask about course e.g. Tell me about COMP4431")
    user_input = user_input or spoken_input
    if user_input:
        st.session_state.chat_history.append({"role":"user", "content":user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        # Call backend RAG (the first call may load the embedding model: keep the UI alive)
        with st.spinner("Searching the syllabus knowledge base…"):
            result = rag_course_query(user_input)
        bot_ans = result["answer"]
        src_list = result["source_files"]
        st.session_state.chat_history.append({"role":"assistant", "content":bot_ans, "sources":src_list})
        with st.chat_message("assistant", avatar=BOT_AVATAR_SVG):
            render_speech_output(bot_ans, "latest-course-answer")
            st.markdown(bot_ans)
            if src_list:
                st.caption(f"📎 Source reference: {', '.join(src_list)}")

else:
    # Academic Advisor Mode input panel
    st.caption("Voice recognition uses the browser's speech service (internet access and browser support may be required).")
    advisor_voice_input = speech_to_text(
        language="en-US",
        start_prompt="🎙️ ",
        stop_prompt="⏹ Stop",
        just_once=True,
        key="advisor_stt",
    )
    # A voice transcript arrives after the text area already exists, so a `value=` default
    # would be ignored. Seed the widget's session state instead; Streamlit applies it when
    # the widget is created further down.
    if advisor_voice_input:
        st.session_state["advisor_preference"] = advisor_voice_input
    elif "advisor_preference" not in st.session_state:
        st.session_state["advisor_preference"] = DEFAULT_ADVISOR_PREFERENCE

    with st.form("advisor_input_form"):
        col_a, col_b, col_c = st.columns([3, 1, 1])
        with col_a:
            finished_str = st.text_input("Finished Courses (comma‑separated)", value="COMP1011, COMP2011, COMP2411, COMP2421",
                                        help="Example: COMP2411, COMP2421")
        with col_b:
            remain_credit = st.number_input("Remaining Credit", min_value=0, max_value=40, value=12)
        with col_c:
            semester_label = st.selectbox("Planned Semester", ["Any", "S1", "S2"])
        preference = st.text_area("Your Study Preference", key="advisor_preference")
        submitted = st.form_submit_button("✨ Generate Course Recommendation")
    if submitted:
        finished_list = [c.strip() for c in finished_str.split(",") if c.strip()]
        user_msg = f"""[Advisor Request]
Finished courses: {finished_str}
Remaining credit: {remain_credit}
Semester: {semester_label}
Preference: {preference}"""
        st.session_state.chat_history.append({"role":"user", "content":user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)
        # Call backend B rule‑engine + advisor pipeline
        try:
            with st.spinner("Checking prerequisites, credits and preferences…"):
                advisor_result = academic_advisor_recommend(
                    finished_list, remain_credit, preference,
                    semester=None if semester_label == "Any" else semester_label)
        except Exception as exc:  # never show a raw traceback during the demo
            advisor_result = {
                "answer": f"Sorry, the advisor could not run ({type(exc).__name__}). Please try again.",
                "source_files": [],
            }
        bot_ans = advisor_result["answer"]
        src_list = advisor_result["source_files"]
        st.session_state.chat_history.append({"role":"assistant", "content":bot_ans, "sources":src_list})
        with st.chat_message("assistant", avatar=BOT_AVATAR_SVG):
            render_speech_output(bot_ans, "latest-advisor-answer")
            st.markdown(bot_ans)
            if src_list:
                st.caption(f"📎 Source reference: {', '.join(src_list)}")

st.divider()
with st.expander("ℹ️ Project Info & Limitation (for demo reference)"):
    st.markdown("""
- Basic Feature: Course Syllabus Lookup powered by RAG knowledge base
- Additional Feature: Academic Advisor (Hybrid Rule‑Engine + RAG + LLM)
- Limitation: Only for single‑major COMP undergraduate. Double‑major / exchange student cases are out‑of‑scope.
- All suggestion is for reference only, please check official university documents.
""")
