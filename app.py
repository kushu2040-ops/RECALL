"""Personalized Study Flashcard & Quiz Maker - Streamlit Web Application.
SaaS Dashboard UI featuring the Recoll • Study Quiz Maker brand identity.
"""

import base64
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

import streamlit as st

from llm_service import LLMService, resolve_api_key
from models import (
    ContentType,
    DifficultyLevel,
    FlashcardDeck,
    GenerationRequest,
    GenerationSourceMode,
    QuizSet,
)
from prompts import SAMPLE_STUDY_MATERIALS
from utils import (
    calculate_quiz_score,
    export_flashcards_to_csv,
    export_flashcards_to_json,
    export_quiz_to_csv,
    export_quiz_to_json,
    export_quiz_to_markdown,
    validate_study_input,
)


def get_base64_image(path: str) -> str:
    """Read a local image and return a data URI for embedding in HTML/CSS."""
    p = Path(path)
    if p.exists():
        data = base64.b64encode(p.read_bytes()).decode("utf-8")
        return f"data:image/png;base64,{data}"
    return ""


# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
_logo_icon_path = Path("assets/logo_icon.png")
st.set_page_config(
    page_title="Recoll • Study Quiz Maker",
    page_icon=str(_logo_icon_path) if _logo_icon_path.exists() else "🟣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Figma-Inspired SaaS Dashboard CSS
# ---------------------------------------------------------------------------
FIGMA_DASHBOARD_CSS = """
<style>
/* Import Clean Modern Font */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #111827;
}

/* Background Color for Main App */
.stApp {
    background-color: #F8FAFC;
}

/* Main Container Adjustments */
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 3.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1280px;
}

/* Left Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] div {
    color: #1E293B;
}

[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] input {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] [data-baseweb="base-input"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 10px !important;
}

/* Secondary Buttons (Only action buttons, NEVER tabs or expanders) */
.stButton > button:not([kind="primary"]),
[data-testid="stBaseButton-secondary"],
[data-testid="stDownloadButton"] > button,
[data-testid="stPopover"] > button {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.15s ease !important;
}

.stButton > button:not([kind="primary"]):hover,
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stDownloadButton"] > button:hover,
[data-testid="stPopover"] > button:hover {
    background-color: #F8FAFC !important;
    border-color: #94A3B8 !important;
    color: #0F172A !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06) !important;
}

/* Streamlit Selectbox Customization */
[data-baseweb="select"],
[data-baseweb="select"] * {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}

[data-baseweb="select"] svg {
    fill: #334155 !important;
}

/* Clean Modern Tabs (Figma SaaS Dashboard Style) */
[data-testid="stTabs"] {
    background: transparent !important;
    margin-top: 0.25rem;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    background-color: transparent !important;
    border-bottom: 2px solid #E2E8F0 !important;
    gap: 2.25rem !important;
    padding: 0 !important;
    margin-bottom: 1.5rem !important;
}

[data-testid="stTabs"] button[role="tab"] {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 0.6rem 0.25rem 0.9rem 0.25rem !important;
    color: #64748B !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: color 0.15s ease !important;
}

[data-testid="stTabs"] button[role="tab"]:hover {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #0F172A !important;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #2563EB !important;
    font-weight: 700 !important;
}

[data-testid="stTabs"] button[role="tab"] p,
[data-testid="stTabs"] button[role="tab"] span,
[data-testid="stTabs"] button[role="tab"] div {
    background: transparent !important;
    font-size: 1rem !important;
    font-weight: inherit !important;
    color: inherit !important;
}

/* Style active tab underline with brand blue */
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #2563EB !important;
    height: 3px !important;
    border-radius: 3px 3px 0 0 !important;
}

[data-testid="stTabs"] [data-baseweb="tab-border"] {
    background-color: #E2E8F0 !important;
}

/* Quiz Radio Option Styling - High Contrast Interactive Cards */
div[data-testid="stRadio"] {
    margin-top: 0.5rem;
    margin-bottom: 1.25rem;
}

div[data-testid="stRadio"] > label {
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #64748B !important;
    margin-bottom: 0.85rem !important;
    display: block !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 0.75rem !important;
    width: 100% !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label,
div[data-testid="stRadio"] label[data-baseweb="radio"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    margin: 0 !important;
    cursor: pointer !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    transition: all 0.18s ease-in-out !important;
    display: flex !important;
    align-items: center !important;
    gap: 1rem !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label:hover,
div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    border-color: #3B82F6 !important;
    background-color: #F8FAFC !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1) !important;
    transform: translateY(-1px) !important;
}

/* Selected option card state */
div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background-color: #EFF6FF !important;
    border-color: #2563EB !important;
    box-shadow: 0 0 0 1.5px #2563EB, 0 4px 14px rgba(37, 99, 235, 0.16) !important;
}

/* Force dark, crystal-clear visible text on every element of the radio group */
div[data-testid="stRadio"] * {
    color: #0F172A !important;
}

div[data-testid="stRadio"] p,
div[data-testid="stRadio"] span,
div[data-testid="stRadio"] div,
div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    color: #0F172A !important;
    font-size: 1.02rem !important;
    font-weight: 600 !important;
    line-height: 1.45 !important;
    margin: 0 !important;
}

div[data-testid="stRadio"] input[type="radio"] {
    accent-color: #2563EB !important;
    width: 18px !important;
    height: 18px !important;
}

/* Section Eyebrow Headers */
.eyebrow-tag {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6B7280;
    margin-bottom: 0.4rem;
}

.brand-header-wrap {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.25rem;
}

.brand-icon-box {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.2rem;
    font-weight: bold;
    box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25);
}

.brand-title-text {
    font-size: 1.35rem;
    font-weight: 800;
    color: #111827;
    letter-spacing: -0.02em;
}

.brand-subtitle-tag {
    font-size: 0.70rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6366F1;
    margin-bottom: 1.2rem;
}

/* Section Card Box */
.sidebar-section-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Top Navigation Bar / Workspace Header */
.top-workspace-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 0.75rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid #E5E7EB;
}

.breadcrumb-nav {
    font-size: 0.85rem;
    color: #6B7280;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.breadcrumb-active {
    color: #111827;
    font-weight: 600;
}

.pill-saved-status {
    background-color: #ECFDF5;
    color: #065F46;
    border: 1px solid #A7F3D0;
    border-radius: 9999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

/* Hero Section in Main Stage */
.hero-header-box {
    margin-bottom: 1.25rem;
}

.hero-eyebrow {
    color: #6366F1;
    font-size: 0.82rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.hero-main-title {
    font-size: 2rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.025em;
    line-height: 1.2;
    margin-bottom: 0.35rem;
}

.hero-meta-row {
    font-size: 0.88rem;
    color: #64748B;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

/* Pill Badges */
.badge-tag {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-radius: 6px;
}
.badge-purple {
    background-color: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
}
.badge-emerald {
    background-color: #ECFDF5;
    color: #065F46;
    border: 1px solid #A7F3D0;
}
.badge-amber {
    background-color: #FFFBEB;
    color: #92400E;
    border: 1px solid #FDE68A;
}

/* Figma Flashcard Stage */
.figma-flashcard {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 20px;
    padding: 3rem 2.5rem;
    min-height: 340px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    text-align: center;
    margin-bottom: 1.5rem;
    transition: all 0.2s ease;
}

.figma-flashcard:hover {
    border-color: #CBD5E1;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.03);
}

.flashcard-question-text {
    font-size: 1.65rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.4;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    max-width: 820px;
    margin-left: auto;
    margin-right: auto;
}

.flashcard-answer-box {
    background-color: #F8FAFC;
    border-left: 4px solid #2563EB;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: left;
    margin-top: 1.25rem;
    margin-bottom: 1rem;
}

.flashcard-answer-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1E1B4B;
    margin-bottom: 0.5rem;
}

.flashcard-explanation-text {
    font-size: 0.95rem;
    color: #475569;
    line-height: 1.6;
}

/* Right-Hand Progress Widget ("Your study progress") */
.progress-widget-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.6rem 1.4rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.03);
}

.progress-widget-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 0.15rem;
}

.progress-widget-sub {
    font-size: 0.85rem;
    color: #64748B;
    margin-bottom: 1.25rem;
}

.progress-big-number {
    font-size: 2.8rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1;
}

.progress-number-denom {
    font-size: 0.92rem;
    color: #64748B;
    font-weight: 600;
}

.stat-item-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid #F1F5F9;
    font-size: 0.9rem;
}

.stat-label-with-dot {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #334155;
    font-weight: 500;
}

.dot-green { width: 8px; height: 8px; border-radius: 50%; background-color: #10B981; }
.dot-amber { width: 8px; height: 8px; border-radius: 50%; background-color: #F59E0B; }
.dot-gray { width: 8px; height: 8px; border-radius: 50%; background-color: #94A3B8; }

.tip-banner-box {
    background-color: #F5F3FF;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-top: 1.25rem;
    border: 1px solid #DDD6FE;
}

.tip-banner-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #4C1D95;
    margin-bottom: 0.2rem;
}

.tip-banner-desc {
    font-size: 0.80rem;
    color: #5B21B6;
    line-height: 1.45;
}

/* Quiz Option Card Styling */
.quiz-choice-container {
    background: #FFFFFF;
    border: 1.5px solid #E2E8F0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.85rem;
    cursor: pointer;
    transition: all 0.15s ease;
}

.quiz-choice-container:hover {
    border-color: #6366F1;
    background-color: #F8FAFC;
}

/* Empty State Onboarding Cards */
.onboarding-hero-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
}

.feature-benefit-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.4rem;
    height: 100%;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.02);
}

.benefit-num {
    font-size: 0.75rem;
    font-weight: 800;
    color: #6366F1;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}

.benefit-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.35rem;
}

.benefit-desc {
    font-size: 0.86rem;
    color: #64748B;
    line-height: 1.5;
}

/* Custom Primary Button Styling */
button[kind="primary"] {
    background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.25rem !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.28) !important;
}

button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1E40AF 0%, #1D4ED8 100%) !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.38) !important;
}
</style>
"""
st.markdown(FIGMA_DASHBOARD_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "notes_content": "",
        "subject_input": "Biology",
        "topic_input": "Cell structure & function",
        "goals_input": "Master energy organelles and cellular respiration",
        "focus_input": "Mitochondria, ATP synthesis, and cell membranes",
        "content_type": ContentType.BOTH.value,
        "source_mode": GenerationSourceMode.NOTES_ONLY.value,
        "quiz_style": "Conceptual & Application",
        "num_flashcards": 10,
        "num_quiz": 5,
        "difficulty": DifficultyLevel.MEDIUM.value,
        # Generated artifacts
        "generated_quiz": None,
        "generated_flashcards": None,
        # Quiz state
        "quiz_current_index": 0,
        "quiz_answers": {},  # question_id -> choice string
        "quiz_submitted": {},  # question_id -> bool
        "quiz_finished": False,
        "quiz_view_mode": "Step-by-Step",
        # Flashcard state
        "card_current_index": 0,
        "card_revealed": False,
        "card_statuses": {},  # card_id -> "known" | "review"
        "card_filter_mode": "All Cards",
        "shuffle_seed": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ---------------------------------------------------------------------------
# Sidebar: "Recall • Create a study set"
# ---------------------------------------------------------------------------
with st.sidebar:
    # 1. Brand Logo & Title
    logo_b64 = get_base64_image("assets/logo.png")
    if logo_b64:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 1.25rem; padding: 0.5rem 0 0.85rem 0; border-bottom: 1px solid #E5E7EB;">
                <img src="{logo_b64}" alt="Recoll - Study Quiz Maker" style="width: 185px; max-width: 95%; height: auto; object-fit: contain;" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="brand-header-wrap">
                <div class="brand-icon-box">R</div>
                <div class="brand-title-text">Recoll</div>
            </div>
            <div class="brand-subtitle-tag">Study Quiz Maker</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Create a study set")
    st.caption("Turn your material into momentum.")

    # Section 01: ADD YOUR MATERIAL
    st.markdown('<div class="eyebrow-tag">01 Add Your Material</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input Mode",
        options=["Paste text", "Load sample topic", "Upload file"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_mode == "Paste text":
        notes_text = st.text_area(
            "Study Material Content",
            value=st.session_state.notes_content,
            height=180,
            placeholder="Paste your notes, textbook chapters, or lecture slides here...",
            label_visibility="collapsed",
        )
        st.session_state.notes_content = notes_text
        char_c = len(notes_text.strip())
        word_c = len(notes_text.strip().split()) if char_c > 0 else 0
        st.caption(f"📝 {word_c} words • {char_c} characters")

    elif input_mode == "Load sample topic":
        sample_choice = st.selectbox(
            "Select Curriculum Sample",
            options=list(SAMPLE_STUDY_MATERIALS.keys()),
            index=0,
        )
        if st.button("📥 Load Sample Content", use_container_width=True):
            s_data = SAMPLE_STUDY_MATERIALS[sample_choice]
            st.session_state.notes_content = s_data["content"]
            st.session_state.subject_input = s_data["subject"]
            st.session_state.topic_input = s_data["exam_topic"]
            st.session_state.goals_input = s_data["learning_goals"]
            st.rerun()

        if st.session_state.notes_content:
            st.success(f"✓ Loaded: {st.session_state.topic_input or 'Sample Topic'}")

    else:  # Upload file
        uploaded = st.file_uploader(
            "Drop your notes here or browse files",
            type=["txt", "md", "csv", "pdf"],
            help="Upload raw notes, documents, or exported textbooks (TXT, Markdown, CSV, PDF).",
        )
        if uploaded is not None:
            file_name_lower = uploaded.name.lower()
            if file_name_lower.endswith(".pdf"):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(uploaded)
                    extracted_pages = []
                    for p in reader.pages:
                        ptxt = p.extract_text()
                        if ptxt:
                            extracted_pages.append(ptxt)
                    raw_file_text = "\n\n".join(extracted_pages)
                except Exception as pe:
                    st.error(f"Error parsing PDF: {pe}")
                    raw_file_text = ""
            else:
                raw_file_text = uploaded.read().decode("utf-8", errors="ignore")

            if raw_file_text.strip():
                st.session_state.notes_content = raw_file_text
                st.session_state.topic_input = uploaded.name.rsplit(".", 1)[0]
                st.success(f"✓ {uploaded.name} loaded ({len(raw_file_text.split())} words)")
            else:
                st.warning(f"No readable text could be extracted from {uploaded.name}.")

    # Section 02: MAKE IT YOURS
    st.markdown("---")
    st.markdown('<div class="eyebrow-tag">02 Make It Yours</div>', unsafe_allow_html=True)

    # 1. Question Source Grounding Mode (Strictly from text vs From Subject & Topic vs Hybrid)
    source_mode_choice = st.radio(
        "Generate Questions From:",
        options=[
            GenerationSourceMode.NOTES_ONLY.value,
            GenerationSourceMode.TOPIC_EXPANDED.value,
            GenerationSourceMode.HYBRID.value,
        ],
        index=0,
        help=(
            "• 'Strictly From Given Notes / File': All questions are strictly extracted only from the text above.\n"
            "• 'From Subject & Topic': AI generates full curriculum exam questions based on Subject/Topic.\n"
            "• 'Hybrid': Anchored in your notes but tests broader conceptual mastery."
        ),
    )
    st.session_state.source_mode = source_mode_choice

    if source_mode_choice == GenerationSourceMode.NOTES_ONLY.value:
        st.info("📖 **Text Mode Active**: Questions will be generated **strictly from your provided text or file**.")
    elif source_mode_choice == GenerationSourceMode.TOPIC_EXPANDED.value:
        st.info("🎓 **Make It Yours Active**: Questions will test **broad curriculum knowledge** from your Subject & Exam Topic below.")
    else:
        st.info("⚡ **Hybrid Mode**: Questions test your source notes combined with broader conceptual applications.")

    # 2. Content Format Selector (Both, Quiz Only, Flashcards Only)
    content_format_choice = st.radio(
        "Study Format",
        options=[
            ContentType.BOTH.value,
            ContentType.FLASHCARDS.value,
            ContentType.QUIZ.value,
        ],
        horizontal=True,
    )
    st.session_state.content_type = content_format_choice

    # Subject & Exam Topic row
    c_sub_a, c_sub_b = st.columns(2)
    with c_sub_a:
        subj_val = st.text_input(
            "Subject focus",
            value=st.session_state.subject_input,
            placeholder="e.g. Molecular Biology",
        )
        st.session_state.subject_input = subj_val

    with c_sub_b:
        topic_val = st.text_input(
            "Exam Topic / Chapter",
            value=st.session_state.topic_input,
            placeholder="e.g. Cell structure & function",
        )
        st.session_state.topic_input = topic_val

    # Difficulty & Question Focus row
    c_diff_a, c_diff_b = st.columns(2)
    with c_diff_a:
        diff_choice = st.selectbox(
            "Difficulty",
            options=["Beginner (Easy)", "Intermediate (Medium)", "Advanced (Hard)"],
            index=1,
        )
        if "Beginner" in diff_choice:
            st.session_state.difficulty = DifficultyLevel.EASY.value
        elif "Advanced" in diff_choice:
            st.session_state.difficulty = DifficultyLevel.HARD.value
        else:
            st.session_state.difficulty = DifficultyLevel.MEDIUM.value

    with c_diff_b:
        quiz_style_choice = st.selectbox(
            "Question Style",
            options=[
                "Conceptual & Application",
                "Factual & Definitions",
                "Scenario & Problem Solving",
                "Exam-Style Hard Distractors",
            ],
            index=0,
            help="Choose how the AI frames questions and distractors.",
        )
        st.session_state.quiz_style = quiz_style_choice

    # Stepper Quantity row (conditioned on Content Format)
    if content_format_choice == ContentType.BOTH.value:
        sc1, sc2 = st.columns(2)
        with sc1:
            st.session_state.num_flashcards = st.number_input(
                "Flashcards",
                min_value=3,
                max_value=30,
                value=st.session_state.num_flashcards,
                step=1,
            )
        with sc2:
            st.session_state.num_quiz = st.number_input(
                "Quiz questions",
                min_value=3,
                max_value=20,
                value=st.session_state.num_quiz,
                step=1,
            )
    elif content_format_choice == ContentType.FLASHCARDS.value:
        st.session_state.num_flashcards = st.number_input(
            "Number of Flashcards",
            min_value=3,
            max_value=40,
            value=st.session_state.num_flashcards,
            step=1,
        )
    else:
        st.session_state.num_quiz = st.number_input(
            "Number of Quiz Questions",
            min_value=3,
            max_value=30,
            value=st.session_state.num_quiz,
            step=1,
        )

    # Optional Custom Focus / Subtopic Emphasis
    with st.expander("🎯 Focus & Learning Goals (Optional)", expanded=False):
        goals_val = st.text_input(
            "Learning Goals",
            value=st.session_state.goals_input,
            placeholder="e.g. Master energy organelles and ATP cycle",
            help="Specify what you want to achieve from this session.",
        )
        st.session_state.goals_input = goals_val

        focus_val = st.text_input(
            "Target Subtopics / Keywords",
            value=st.session_state.focus_input,
            placeholder="e.g. Focus on thylakoid membrane and Calvin cycle",
            help="Direct the AI to prioritize specific concepts or formulas.",
        )
        st.session_state.focus_input = focus_val

    # API Configuration Collapsible - Google Gemini
    with st.expander("🔑 LLM Settings (Google Gemini / Demo)", expanded=True):
        api_key_default = resolve_api_key() or ""
        user_api_key = st.text_input(
            "Gemini API Key",
            value=api_key_default,
            type="password",
            help="Google Gemini API Key from https://aistudio.google.com/app/apikey. Kept in memory, never stored to disk.",
        )
        if user_api_key and user_api_key.startswith("sk-"):
            st.info(
                "ℹ️ Detected an OpenAI key ('sk-...'). Recoll is now powered by **Google Gemini**! "
                "Please paste a free Gemini API key from [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) (starts with 'AIzaSy...'), "
                "or use Demo Mode below."
            )

        model_options = [
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
            "gemini-3.6-flash",
            "gemini-3.1-flash-lite",
        ]
        model_name_input = st.selectbox(
            "Gemini Model",
            options=model_options,
            index=0,
            help="gemini-3.5-flash-lite is Google's ultra-lightweight, high-throughput model optimized for low-latency study material generation.",
        )

        # Auto-detect if user has an API key configured (from .env or input)
        has_api_key = bool(user_api_key)
        demo_default = not has_api_key
        if "force_demo_mode" in st.session_state:
            demo_default = st.session_state["force_demo_mode"]

        mock_mode_toggle = st.checkbox(
            "Demo Mode (Offline - No Key Required)",
            value=demo_default,
            help="Generate study materials instantly without calling external API or incurring quota costs.",
            key="mock_mode_checkbox",
        )
        # Store user's explicit choice in session state
        st.session_state["force_demo_mode"] = mock_mode_toggle

        st.caption("✨ **Google Gemini Free Tier:** Get a free API key with generous RPM/TPM limits at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).")

    # Primary Action Button
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    generate_clicked = st.button("🚀 Generate study set", type="primary", use_container_width=True)

    st.markdown(
        """
        <div style="text-align: center; margin-top: 0.6rem; font-size: 0.78rem; color: #6B7280;">
            ✓ Your work stays in this session.<br>Switch views without losing progress.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🗑️ Reset All Progress", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        init_session_state()
        st.rerun()


# ---------------------------------------------------------------------------
# Generation Handler
# ---------------------------------------------------------------------------
if generate_clicked:
    raw_notes = st.session_state.notes_content.strip()
    # If no notes entered, auto-load Biology sample as friendly default
    if not raw_notes:
        bio_sample = SAMPLE_STUDY_MATERIALS["Biology: Cellular Respiration"]
        raw_notes = bio_sample["content"]
        st.session_state.notes_content = raw_notes
        if not st.session_state.subject_input:
            st.session_state.subject_input = bio_sample["subject"]
        if not st.session_state.topic_input:
            st.session_state.topic_input = bio_sample["exam_topic"]

    is_valid, validation_msg = validate_study_input(raw_notes)
    if not is_valid:
        st.sidebar.error(f"❌ {validation_msg}")
    else:
        # Generation mode: only use demo mode if explicitly toggled ON or if no API key is provided
        is_demo_active = mock_mode_toggle or not bool(user_api_key)
        service = LLMService(
            api_key=user_api_key,
            model_name=model_name_input,
            mock_mode=is_demo_active,
        )

        # Combine custom focus and question style into focus instructions
        style_instruction = f"Question Style: {st.session_state.get('quiz_style', 'Conceptual & Application')}."
        user_focus = st.session_state.get("focus_input", "")
        combined_focus = f"{style_instruction} {user_focus}".strip()

        # Extract source grounding mode
        chosen_source_mode = GenerationSourceMode(st.session_state.get("source_mode", GenerationSourceMode.NOTES_ONLY.value))

        with st.spinner("🧠 Generating your study set..."):
            deck = None
            fc_err = None
            quiz = None
            q_err = None

            # 1. Generate Flashcards (if requested)
            if content_format_choice in (ContentType.BOTH.value, ContentType.FLASHCARDS.value):
                fc_req = GenerationRequest(
                    content=raw_notes,
                    content_type=ContentType.FLASHCARDS,
                    source_mode=chosen_source_mode,
                    subject=st.session_state.subject_input,
                    exam_topic=st.session_state.topic_input,
                    learning_goals=st.session_state.goals_input,
                    num_items=st.session_state.num_flashcards,
                    difficulty=DifficultyLevel(st.session_state.difficulty),
                    focus_instructions=combined_focus,
                )
                deck, fc_err = service.generate_flashcards(fc_req)

            # 2. Generate Quiz (if requested)
            if content_format_choice in (ContentType.BOTH.value, ContentType.QUIZ.value):
                q_req = GenerationRequest(
                    content=raw_notes,
                    content_type=ContentType.QUIZ,
                    source_mode=chosen_source_mode,
                    subject=st.session_state.subject_input,
                    exam_topic=st.session_state.topic_input,
                    learning_goals=st.session_state.goals_input,
                    num_items=st.session_state.num_quiz,
                    difficulty=DifficultyLevel(st.session_state.difficulty),
                    focus_instructions=combined_focus,
                )
                quiz, q_err = service.generate_quiz(q_req)

            # Evaluate errors based on what was requested
            errors = [e for e in (fc_err, q_err) if e]
            if errors and not (deck or quiz):
                st.session_state["generation_error"] = " • ".join(errors)
            else:
                st.session_state["generation_error"] = None
                if deck:
                    st.session_state.generated_flashcards = deck
                    st.session_state.card_current_index = 0
                    st.session_state.card_revealed = False
                    st.session_state.card_statuses = {}
                elif content_format_choice == ContentType.QUIZ.value:
                    st.session_state.generated_flashcards = None

                if quiz:
                    st.session_state.generated_quiz = quiz
                    st.session_state.quiz_current_index = 0
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = {}
                    st.session_state.quiz_finished = False
                elif content_format_choice == ContentType.FLASHCARDS.value:
                    st.session_state.generated_quiz = None

                st.success("✨ Study set generated successfully!")
                st.rerun()


# ---------------------------------------------------------------------------
# Top Navigation Bar & Breadcrumb
# ---------------------------------------------------------------------------
active_cards: FlashcardDeck = st.session_state.generated_flashcards
active_quiz: QuizSet = st.session_state.generated_quiz

topic_display = st.session_state.topic_input or "Cell structure & function"
subject_display = st.session_state.subject_input or "Biology"

st.markdown(
    f"""
    <div class="top-workspace-bar">
        <div class="breadcrumb-nav">
            <span>My workspace</span>
            <span>/</span>
            <span class="breadcrumb-active">{subject_display}</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span class="pill-saved-status">✓ Progress saved for this session</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render diagnostic alert if an error occurred during generation
if st.session_state.get("generation_error"):
    g_err = st.session_state["generation_error"]
    st.error(f"⚠️ **Generation Notice:**\n\n{g_err}")

    if "quota" in g_err.lower() or "429" in g_err.lower() or "billing" in g_err.lower() or "account" in g_err.lower():
        ec1, ec2 = st.columns([0.45, 0.55])
        with ec1:
            if st.button("⚡ Switch to Demo Mode & Generate Now", type="primary", key="switch_to_demo_now_btn", use_container_width=True):
                st.session_state["force_demo_mode"] = True
                st.session_state["generation_error"] = None
                # Auto-generate offline set
                demo_srv = LLMService(mock_mode=True)
                raw_mat = st.session_state.notes_content or SAMPLE_STUDY_MATERIALS["Biology: Cellular Respiration"]["content"]
                st.session_state.notes_content = raw_mat
                d_fc_req = GenerationRequest(
                    content=raw_mat,
                    content_type=ContentType.FLASHCARDS,
                    subject=st.session_state.subject_input or "Biology",
                    exam_topic=st.session_state.topic_input or "Cell structure & function",
                    num_items=st.session_state.num_flashcards,
                    difficulty=DifficultyLevel(st.session_state.difficulty),
                )
                d_q_req = GenerationRequest(
                    content=raw_mat,
                    content_type=ContentType.QUIZ,
                    subject=st.session_state.subject_input or "Biology",
                    exam_topic=st.session_state.topic_input or "Cell structure & function",
                    num_items=st.session_state.num_quiz,
                    difficulty=DifficultyLevel(st.session_state.difficulty),
                )
                d_deck, _ = demo_srv.generate_flashcards(d_fc_req)
                d_quiz, _ = demo_srv.generate_quiz(d_q_req)
                if d_deck:
                    st.session_state.generated_flashcards = d_deck
                    st.session_state.card_current_index = 0
                    st.session_state.card_revealed = False
                    st.session_state.card_statuses = {}
                if d_quiz:
                    st.session_state.generated_quiz = d_quiz
                    st.session_state.quiz_current_index = 0
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = {}
                    st.session_state.quiz_finished = False
                st.rerun()
        with ec2:
            st.info("💡 **Google Gemini API:** You can get a free Google Gemini API key with generous limits at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey). Demo Mode lets you explore all features offline.")


# ---------------------------------------------------------------------------
# Workspace Content Tabs
# ---------------------------------------------------------------------------
flashcards_count = len(active_cards.cards) if active_cards else st.session_state.num_flashcards
quiz_count = len(active_quiz.questions) if active_quiz else st.session_state.num_quiz

tab_flashcards, tab_quiz, tab_notes = st.tabs([
    f"Flashcards  ({flashcards_count})",
    f"Practice quiz  ({quiz_count})",
    "Source notes",
])


# ===========================================================================
# TAB 1: FLASHCARDS (Figma Exact Layout)
# ===========================================================================
with tab_flashcards:
    if not active_cards:
        # Onboarding / Empty State (Matching Figma "Study workspace · Get started")
        logo_icon_b64 = get_base64_image("assets/logo.png")
        hero_brand_visual = (
            f'<div style="display: flex; justify-content: center; margin-bottom: 1.25rem;">'
            f'<img src="{logo_icon_b64}" alt="Recoll Logo" style="width: 175px; height: auto;" />'
            f'</div>'
            if logo_icon_b64
            else '<div style="font-size: 2.8rem; margin-bottom: 0.8rem;">📘</div>'
        )
        st.markdown(
            f"""
            <div class="onboarding-hero-card">
                {hero_brand_visual}
                <h2 style="font-size: 1.8rem; font-weight: 800; color: #0F172A; margin-bottom: 0.5rem;">
                    Your next "aha" starts here.
                </h2>
                <p style="color: #64748B; font-size: 1.05rem; max-width: 620px; margin: 0 auto 1.5rem auto;">
                    Turn your material into momentum. Paste your lecture notes or pick a curriculum sample from the left sidebar to generate active-recall flashcards and practice quizzes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            st.markdown(
                """
                <div class="feature-benefit-card">
                    <div class="benefit-num">01 Add your material</div>
                    <div class="benefit-title">Analyze dense notes</div>
                    <div class="benefit-desc">Condense chapters and slides into targeted active-recall prompts.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b_col2:
            st.markdown(
                """
                <div class="feature-benefit-card">
                    <div class="benefit-num">02 Make it yours</div>
                    <div class="benefit-title">Match your pace</div>
                    <div class="benefit-desc">Calibrate question difficulty from foundational definitions to exam-level deductive problem solving.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b_col3:
            st.markdown(
                """
                <div class="feature-benefit-card">
                    <div class="benefit-num">03 Test & export</div>
                    <div class="benefit-title">Lock in retention</div>
                    <div class="benefit-desc">Self-assess with spaced repetition, view live mastery metrics, and export decks directly to Anki or CSV.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        # Hero Header Row
        h_left, h_right = st.columns([0.75, 0.25])
        with h_left:
            st.markdown(
                f"""
                <div class="hero-header-box">
                    <div class="hero-eyebrow">A little practice. A lot of progress.</div>
                    <div class="hero-main-title">{active_cards.title}</div>
                    <div class="hero-meta-row">
                        <span>From notes</span>
                        <span>•</span>
                        <span>{len(active_cards.cards)} flashcards</span>
                        <span>•</span>
                        <span>{active_cards.subject or 'General'}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h_right:
            # Export Dropdown
            with st.popover("↓ Export set", use_container_width=True):
                st.download_button(
                    "📥 Download JSON",
                    data=export_flashcards_to_json(active_cards),
                    file_name="recall_flashcards.json",
                    mime="application/json",
                    use_container_width=True,
                )
                st.download_button(
                    "📊 Download CSV (Anki)",
                    data=export_flashcards_to_csv(active_cards),
                    file_name="recall_flashcards.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        # Main Stage: 2-Column Layout (Central Flashcard + Right Progress Widget)
        stage_col, progress_col = st.columns([0.68, 0.32], gap="large")

        all_cards = active_cards.cards
        known_count = sum(1 for cid, s in st.session_state.card_statuses.items() if s == "known")
        review_count = sum(1 for cid, s in st.session_state.card_statuses.items() if s == "review")
        unreviewed_count = max(0, len(all_cards) - (known_count + review_count))

        with stage_col:
            # Action row above card
            act_c1, act_c2 = st.columns([0.7, 0.3])
            with act_c1:
                cur_idx = max(0, min(st.session_state.card_current_index, len(all_cards) - 1))
                st.markdown(
                    f'<div class="eyebrow-tag">FLASHCARD {cur_idx + 1:02d} OF {len(all_cards):02d}</div>',
                    unsafe_allow_html=True,
                )
            with act_c2:
                if st.button("Shuffle cards ⇄", key="shuffle_btn", use_container_width=True):
                    import random
                    random.shuffle(active_cards.cards)
                    st.session_state.card_current_index = 0
                    st.session_state.card_revealed = False
                    st.rerun()

            current_card = all_cards[cur_idx]
            card_id = current_card.id
            is_revealed = st.session_state.card_revealed

            # Flashcard Render Box
            if not is_revealed:
                st.markdown(
                    f"""
                    <div class="figma-flashcard">
                        <div>
                            <span class="badge-tag badge-purple">{current_card.key_concept}</span>
                            <div style="font-size: 0.75rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-top: 1.5rem;">
                                QUESTION
                            </div>
                            <div class="flashcard-question-text">
                                {current_card.front}
                            </div>
                        </div>
                        <div style="color: #94A3B8; font-size: 0.88rem; font-style: italic; margin-top: 1rem;">
                            Take a moment to recall before revealing the answer.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="figma-flashcard">
                        <div>
                            <span class="badge-tag badge-purple">{current_card.key_concept}</span>
                            <div style="font-size: 0.82rem; color: #64748B; margin-top: 1rem;">
                                <strong>Question:</strong> {current_card.front}
                            </div>
                            <div class="flashcard-answer-box">
                                <div class="flashcard-answer-title">💡 {current_card.back}</div>
                                <div class="flashcard-explanation-text">{current_card.explanation}</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Center Action: Reveal / Hide Toggle
            rev_c1, rev_c2, rev_c3 = st.columns([0.25, 0.5, 0.25])
            with rev_c2:
                btn_text = "🔄 Flip to front" if is_revealed else "↻ Reveal answer"
                if st.button(btn_text, key=f"rev_btn_{card_id}", type="primary", use_container_width=True):
                    st.session_state.card_revealed = not st.session_state.card_revealed
                    st.rerun()

            # Navigation & Bottom Rating Row
            st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
            nav_col1, nav_col2, nav_col3 = st.columns([0.3, 0.4, 0.3])
            with nav_col1:
                if st.button("← Previous", disabled=cur_idx == 0, use_container_width=True):
                    st.session_state.card_current_index = max(0, cur_idx - 1)
                    st.session_state.card_revealed = False
                    st.rerun()

            with nav_col2:
                # Stepper display
                st.markdown(
                    f"<div style='text-align: center; font-size: 0.92rem; font-weight: 700; color: #475569; padding-top: 0.45rem;'>"
                    f"{cur_idx + 1} / {len(all_cards)}</div>",
                    unsafe_allow_html=True,
                )

            with nav_col3:
                if st.button("Next card →", disabled=cur_idx >= len(all_cards) - 1, use_container_width=True):
                    st.session_state.card_current_index = min(len(all_cards) - 1, cur_idx + 1)
                    st.session_state.card_revealed = False
                    st.rerun()

            # Rating Buttons (Still Learning vs. Know It)
            st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
            rate_c1, rate_c2 = st.columns(2)
            with rate_c1:
                if st.button("🟡 Still learning", key=f"learn_{card_id}", use_container_width=True):
                    st.session_state.card_statuses[card_id] = "review"
                    if cur_idx < len(all_cards) - 1:
                        st.session_state.card_current_index = cur_idx + 1
                        st.session_state.card_revealed = False
                    st.rerun()

            with rate_c2:
                if st.button("🟢 Know it", key=f"know_{card_id}", use_container_width=True):
                    st.session_state.card_statuses[card_id] = "known"
                    if cur_idx < len(all_cards) - 1:
                        st.session_state.card_current_index = cur_idx + 1
                        st.session_state.card_revealed = False
                    st.rerun()

        # Right-Hand Column: "Your study progress" Widget
        with progress_col:
            total_c = len(all_cards)
            reviewed_c = known_count + review_count
            pct = (reviewed_c / total_c) if total_c > 0 else 0.0

            st.markdown(
                f"""
                <div class="progress-widget-card">
                    <div class="progress-widget-title">Your study progress</div>
                    <div class="progress-widget-sub">Small steps add up.</div>
                    <div style="display: flex; align-items: baseline; gap: 0.4rem; margin-bottom: 0.75rem;">
                        <span class="progress-big-number">{reviewed_c}</span>
                        <span class="progress-number-denom">/ {total_c} cards reviewed</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(pct)

            progress_breakdown_html = f"""
<div class="progress-widget-card" style="margin-top: 0.75rem; border-top: none;">
<div class="stat-item-row">
<span class="stat-label-with-dot"><span class="dot-green"></span> Know it</span>
<span style="font-weight: 700; color: #111827;">{known_count}</span>
</div>
<div class="stat-item-row">
<span class="stat-label-with-dot"><span class="dot-amber"></span> Still learning</span>
<span style="font-weight: 700; color: #111827;">{review_count}</span>
</div>
<div class="stat-item-row">
<span class="stat-label-with-dot"><span class="dot-gray"></span> Not reviewed</span>
<span style="font-weight: 700; color: #111827;">{unreviewed_count}</span>
</div>
<div class="tip-banner-box">
<div class="tip-banner-title">💡 Make it stick</div>
<div class="tip-banner-desc">Review 'Still learning' cards once more before your exam to lock them into long-term memory.</div>
</div>
</div>
"""
            st.markdown(progress_breakdown_html, unsafe_allow_html=True)


# ===========================================================================
# TAB 2: PRACTICE QUIZ (Figma SaaS Multiple Choice)
# ===========================================================================
with tab_quiz:
    if not active_quiz:
        st.info("📋 No quiz generated yet. Click **'🚀 Generate study set'** in the sidebar to create one.")
    else:
        q_left, q_right = st.columns([0.75, 0.25])
        with q_left:
            st.markdown(
                f"""
                <div class="hero-header-box">
                    <div class="hero-eyebrow">Exam-Ready Multiple Choice</div>
                    <div class="hero-main-title">{active_quiz.title}</div>
                    <div class="hero-meta-row">
                        <span>{len(active_quiz.questions)} Questions</span>
                        <span>•</span>
                        <span>Difficulty: {active_quiz.difficulty.value}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with q_right:
            with st.popover("↓ Export quiz", use_container_width=True):
                st.download_button(
                    "📥 Download JSON",
                    data=export_quiz_to_json(active_quiz),
                    file_name="recall_quiz.json",
                    mime="application/json",
                    use_container_width=True,
                )
                st.download_button(
                    "📊 Download CSV",
                    data=export_quiz_to_csv(active_quiz),
                    file_name="recall_quiz.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.download_button(
                    "📝 Markdown Guide",
                    data=export_quiz_to_markdown(active_quiz),
                    file_name="recall_quiz_guide.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        q_stage, q_side = st.columns([0.7, 0.3], gap="large")

        questions = active_quiz.questions
        q_idx = max(0, min(st.session_state.quiz_current_index, len(questions) - 1))
        curr_q = questions[q_idx]
        q_id = curr_q.id

        with q_stage:
            st.markdown(
                f"""
                <div style="background: white; border: 1px solid #E2E8F0; border-radius: 16px; padding: 2rem; margin-bottom: 1.25rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <span class="badge-tag badge-purple">{curr_q.reference_concept or 'Concept'}</span>
                        <span style="font-size: 0.8rem; font-weight: 700; color: #64748B;">QUESTION {q_idx + 1} OF {len(questions)}</span>
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #0F172A; line-height: 1.45;">
                        {curr_q.question}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Choice Selection
            saved_choice = st.session_state.quiz_answers.get(q_id, None)
            letters = ["A", "B", "C", "D"]
            choice_map = {c: f"{letters[i]}.   {c}" for i, c in enumerate(curr_q.choices)}
            
            chosen_option = st.radio(
                f"Choose the correct answer for Question {q_idx + 1}:",
                options=curr_q.choices,
                format_func=lambda c: choice_map.get(c, c),
                index=curr_q.choices.index(saved_choice) if saved_choice in curr_q.choices else None,
                key=f"figma_radio_{q_id}_{q_idx}",
            )

            c_sub1, c_sub2 = st.columns([0.35, 0.65])
            with c_sub1:
                if st.button("Submit answer", key=f"q_sub_{q_id}", type="primary", use_container_width=True):
                    if chosen_option:
                        st.session_state.quiz_answers[q_id] = chosen_option
                        st.session_state.quiz_submitted[q_id] = True
                        st.rerun()
                    else:
                        st.warning("Please select an answer first.")

            # Immediate Feedback Box
            if st.session_state.quiz_submitted.get(q_id, False):
                user_ans = st.session_state.quiz_answers.get(q_id)
                is_correct = user_ans.strip().lower() == curr_q.correct_answer.strip().lower()
                if is_correct:
                    st.success(f"🎉 **Correct!** Great work.\n\n**Your Answer:** {user_ans}")
                else:
                    st.error(f"❌ **Incorrect.**\n\n**Your Answer:** {user_ans}\n\n**Correct Answer:** {curr_q.correct_answer}")

                with st.expander("💡 View Detailed Explanation", expanded=True):
                    st.markdown(f"**Explanation:** {curr_q.explanation}")

            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            q_nav1, q_nav2, q_nav3 = st.columns([0.3, 0.4, 0.3])
            with q_nav1:
                if st.button("← Previous question", disabled=q_idx == 0, use_container_width=True):
                    st.session_state.quiz_current_index = max(0, q_idx - 1)
                    st.rerun()
            with q_nav2:
                st.markdown(
                    f"<div style='text-align: center; font-size: 0.9rem; font-weight: 700; color: #475569; padding-top: 0.45rem;'>"
                    f"{q_idx + 1} of {len(questions)}</div>",
                    unsafe_allow_html=True,
                )
            with q_nav3:
                if q_idx < len(questions) - 1:
                    if st.button("Next question →", use_container_width=True):
                        st.session_state.quiz_current_index = q_idx + 1
                        st.rerun()
                else:
                    if st.button("🏁 Finish Quiz", type="primary", use_container_width=True):
                        st.session_state.quiz_finished = True
                        st.rerun()

        with q_side:
            score_data = calculate_quiz_score(st.session_state.quiz_answers, questions)
            st.markdown(
                f"""
                <div class="progress-widget-card">
                    <div class="progress-widget-title">Quiz Performance</div>
                    <div class="progress-widget-sub">{score_data['grade_label']}</div>
                    <div style="display: flex; align-items: baseline; gap: 0.4rem; margin-bottom: 0.5rem;">
                        <span class="progress-big-number">{score_data['percentage']}%</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #475569; margin-bottom: 1rem;">
                        <strong>{score_data['correct_count']}</strong> of {score_data['total_questions']} correct
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(score_data["percentage"] / 100.0)

            # Question map pills
            st.markdown("<div style='margin-top: 1rem; font-size: 0.82rem; font-weight: 700; color: #64748B;'>QUESTION DIRECTORY</div>", unsafe_allow_html=True)
            for idx, q_item in enumerate(questions):
                is_done = st.session_state.quiz_submitted.get(q_item.id, False)
                icon = "✓" if is_done else "○"
                btn_lbl = f"Q{idx + 1} {icon}"
                if st.button(btn_lbl, key=f"q_dir_{q_item.id}", use_container_width=True):
                    st.session_state.quiz_current_index = idx
                    st.rerun()


# ===========================================================================
# TAB 3: SOURCE NOTES
# ===========================================================================
with tab_notes:
    st.markdown("### 📄 Source Study Material")
    if st.session_state.notes_content:
        st.text_area(
            "Original Text Content",
            value=st.session_state.notes_content,
            height=320,
            disabled=True,
        )
    else:
        st.info("No study notes entered yet. Use the left sidebar to add notes or load a sample.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; font-size: 0.8rem; color: #94A3B8; margin-top: 3.5rem; border-top: 1px solid #E5E7EB; padding-top: 1.25rem;">
        Recoll • Study Quiz Maker • AI-Generated Study Material • Verified with Pydantic & Structured Outputs
    </div>
    """,
    unsafe_allow_html=True,
)
