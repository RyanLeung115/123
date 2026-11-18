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

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_mode" not in st.session_state:
    # Two modes: "course_lookup" (basic feature), "advisor" (additional feature)
    st.session_state.current_mode = "course_lookup"

# ===================== Backend Connection =====================
# All backend glue (imports, wiring, guards, fallbacks) lives in backend_bridge.py:
#   Backend A  rag_backend.py                                  -> rag_course_query()        (basic feature)
#   Backend B  advisor.py + rule_engine.py + courses.json      -> academic_advisor_recommend() (additional)
# The UI below only calls these two functions, never the backend modules directly.
from backend_bridge import (
    academic_advisor_recommend,
    backend_status,
    rag_course_query,
)


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
        st.session_state.current_mode = "course_lookup"
        st.rerun()
with col_mode2:
    if st.button("🧑‍🏫 Academic Advisor (Additional Feature)", use_container_width=True,
                 type="primary" if st.session_state.current_mode == "advisor" else "secondary"):
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
        # Call backend A (RAG) through the connection layer
        with st.spinner("Searching the course knowledge base…"):
            result = rag_course_query(user_input)
        bot_ans = result["answer"]
        src_list = result["source_files"]
        engine = result.get("engine", "")
        st.session_state.chat_history.append({"role":"assistant", "content":bot_ans, "sources":src_list})
        with st.chat_message("assistant", avatar=BOT_AVATAR_SVG):
            render_speech_output(bot_ans, "latest-course-answer")
            st.markdown(bot_ans)
            if src_list:
                st.caption(f"📎 Source reference: {', '.join(src_list)}")
            if engine == "courses.json":
                st.caption("⚙️ No syllabus PDF matched — this answer comes from courses.json.")

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
    advisor_default_preference = (
        advisor_voice_input
        if advisor_voice_input
        else "I prefer AI‑focused courses and want to avoid heavy coding coursework."
    )

    with st.form("advisor_input_form"):
        col_a, col_b, col_c = st.columns([3, 1, 1])
        with col_a:
            finished_str = st.text_input("Finished Courses (comma‑separated)", value="COMP1011, COMP2011, COMP2411, COMP2421",
                                        help="Example: COMP2411, COMP2421")
        with col_b:
            remain_credit = st.number_input("Remaining Credit", min_value=0, max_value=40, value=12)
        with col_c:
            semester_label = st.selectbox("Planned Semester", ["Any", "S1", "S2"])
        preference = st.text_area("Your Study Preference", value=advisor_default_preference)
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
        advisor_result = academic_advisor_recommend(finished_list, remain_credit, preference,
                                                 semester=None if semester_label == "Any" else semester_label)
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
    _st = backend_status()
    st.caption(
        f"🔌 Backend A · course lookup (rag_backend.py): **{_st['rag']}**"
        + (f" — {_st['rag_error']}" if _st["rag_error"] else "")
        + f" | Backend B · advisor (advisor.py): **{_st['advisor']}**"
        + (f" — {_st['advisor_error']}" if _st["advisor_error"] else "")
        + f" | courses.json: {_st['catalog_rows']} courses"
        + f" | syllabus snippets: {'on' if _st['rag_snippets'] else 'off'}"
    )