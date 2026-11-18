# 🎓 Personalized Study Flashcard & Quiz Maker

A production-ready, interactive Streamlit web application that converts raw textbook excerpts, lecture slides, and study notes into high-yield **multiple-choice practice quizzes** and **spaced-repetition flashcards** using **Google Gemini API** (`google-genai` official SDK) with **Structured Outputs** and **Pydantic schema validation**.

---

## 🌟 Key Features

### 1. Intelligent Input Interface
- **Substantive Note Processing**: Paste long-form textbook chapters, articles, or lecture notes.
- **Optional Metadata**: Tailor items with Subject, Exam Topic, and specific Learning Goals.
- **Granular Controls**:
  - **Content Format**: Multiple-Choice Quiz or Q&A Flashcards.
  - **Item Quantity**: Configurable from 3 to 20 items.
  - **Cognitive Difficulty**: *Easy* (recall), *Medium* (application/comparison), or *Hard* (synthesis/deductive problem-solving).
  - **Focus Instructions**: Direct the AI toward specific sub-concepts, formulas, or question types.
- **One-Click Sample Loader**: Preloaded with rich study material across Biology, Operating Systems, and Macroeconomics.

### 2. Robust LLM Integration & Structured Outputs
- **Google Gemini API Engine**: Powered by Google's lightweight and high-speed models (`gemini-3.5-flash-lite`, `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.1-flash-lite`) using the official `google-genai` SDK with native JSON structured output schemas.
- **Strict Pydantic Validation**: Every LLM response is enforced against strict Pydantic models.
  - Guarantees **exactly 4 distinct choices** per question.
  - Enforces that `correct_answer` strictly matches one of the 4 choices.
  - Requires in-depth explanations and concept tags.
- **Multi-Chunk Long Document Splitting**: Automatically partitions extensive notes into overlapping paragraph/sentence chunks, distributes question generation proportionally across chunks, and merges them without losing conceptual coverage.
- **Offline / Demo Mode**: Built-in deterministic generator allows full end-to-end evaluation without an API key or when offline.

### 3. Interactive Quiz Mode
- **Two Display Modes**:
  - **Step-by-Step**: Focus on one question at a time with instant color-coded feedback, detailed explanation, concept reference, and progress tracking.
  - **Full Quiz View**: Review all questions in a clean list format with a single bulk submission.
- **Instant Pedagogical Feedback**: Highlights correct choice, displays reason for incorrect answers, and explains distractor misconceptions.
- **Scoring & Performance Analysis**: Calculates percentage score, performance tier label, and detailed question breakdown.
- **Retake & Regenerate**: Restart the quiz with one click or regenerate with new parameters.
- **Export Formats**: Download as **JSON**, **CSV**, or a formatted **Markdown Study Guide**.

### 4. Spaced-Repetition Flashcard Mode
- **Active Recall Interaction**: Front card prompt with a smooth **"Flip to Reveal Answer"** toggle.
- **Back Details**: Concise answer, comprehensive explanation, key concept badge, and difficulty tier.
- **Self-Assessment Workflow**:
  - **Mark as Known (✓ Mastered)**
  - **Review Again (↺ Needs Review)**
  - **Skip / Next**
- **Dynamic Queue Filtering**: Filter view to review only cards marked as "Review Again".
- **Progress Tracking**: Real-time counter of Mastered vs. Needs Review cards and overall deck mastery percentage.
- **Export Formats**: Download flashcards as **JSON** or **CSV** (compatible with Anki and spreadsheets).

### 5. Modern, Responsive UX
- Clean card layout with high-contrast typography, difficulty pills, and mobile-friendly controls.
- Sidebar controls keep the main workspace clutter-free.
- Streamlit session state preserves all generated items, user answers, and deck statuses across re-renders.
- AI-generated content clearly badged and labeled.

---

## 📂 Project Structure

```
Genaiproject/
├── app.py              # Main Streamlit application entry point & UI
├── llm_service.py      # LLM integration (Google Gemini structured outputs, chunking, demo mode)
├── models.py           # Pydantic data contracts (QuizQuestion, QuizSet, Flashcard, etc.)
├── prompts.py          # System prompts, user prompt builders, and curated sample texts
├── utils.py            # Input validation, paragraph chunker, scoring, and data exporters
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment variable template
├── README.md           # Documentation and setup guide
└── tests/
    ├── test_core.py    # Schema validation, scoring, chunking, and export unit tests
    └── test_flows.py   # End-to-end integration tests for all 5 core workflows
```

---

## 🚀 Setup & Installation Instructions

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed on your system.
- Git (optional, for version control).

---

### Windows Setup

1. **Open PowerShell or Command Prompt**:
   ```powershell
   cd d:\Genaiproject
   ```

2. **Create and Activate a Virtual Environment** (recommended):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   *(If script execution is disabled, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*

3. **Install Dependencies**:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. **Configure Environment Variables** (optional if using sidebar or demo mode):
   ```powershell
   Copy-Item .env.example .env
   ```
   Edit `.env` and set your `GEMINI_API_KEY`:
   ```env
   GEMINI_API_KEY=AIzaSy...
   GEMINI_MODEL=gemini-3.5-flash-lite
   ```

5. **Run the Streamlit Application**:
   ```powershell
   streamlit run app.py
   ```
   The application will automatically open in your default browser at `http://localhost:8501`.

---

### macOS & Linux Setup

1. **Open Terminal**:
   ```bash
   cd /path/to/Genaiproject
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   ```bash
   cp .env.example .env
   nano .env  # or code .env
   ```
   Add your `GEMINI_API_KEY`.

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---

## ⚙️ Configuration & API Key Options

The application supports three flexible ways to supply your API credentials:

1. **Interactive Sidebar Input**: Paste your Gemini API key (starts with `AIzaSy...`) directly into the **LLM Settings** in the sidebar. The key is kept strictly in memory for your current session and never stored to disk.
2. **Environment Variable**: Set `GEMINI_API_KEY` in your shell or inside a `.env` file in the project root. You can get a free API key with generous limits from [Google AI Studio](https://aistudio.google.com/app/apikey).
3. **Streamlit Secrets**: Add `GEMINI_API_KEY = "AIzaSy..."` in `.streamlit/secrets.toml`.
4. **Demo Mode (Offline - No Key Required)**: Check the **Demo Mode** checkbox in the sidebar to generate instant, realistic study materials offline without calling any external API.

---

## 🧪 Running Unit & Integration Tests

The project includes a test suite covering schema validation, chunking, scoring, export formats, error recovery, and end-to-end user flows.

Run all 25 unit and integration tests:
```bash
python -m pytest tests/ -v
```

Expected output:
```
tests/test_core.py::test_valid_quiz_question PASSED
tests/test_core.py::test_quiz_question_letter_answer_normalization PASSED
tests/test_core.py::test_quiz_question_rejects_invalid_choice_count PASSED
tests/test_core.py::test_quiz_question_rejects_duplicate_choices PASSED
tests/test_core.py::test_quiz_question_rejects_unmatched_correct_answer PASSED
tests/test_core.py::test_quiz_set_deduplication_and_indexing PASSED
tests/test_core.py::test_flashcard_validation PASSED
tests/test_core.py::test_flashcard_deck_deduplication PASSED
tests/test_core.py::test_validate_study_input PASSED
tests/test_core.py::test_chunk_text_short PASSED
tests/test_core.py::test_chunk_text_long_paragraphs PASSED
tests/test_core.py::test_calculate_quiz_score PASSED
tests/test_core.py::test_export_quiz_formats PASSED
tests/test_core.py::test_export_flashcards_formats PASSED
tests/test_core.py::test_mock_quiz_generation PASSED
tests/test_core.py::test_mock_flashcard_generation PASSED
tests/test_core.py::test_service_handles_empty_input_gracefully PASSED
tests/test_core.py::test_resolve_api_key PASSED
tests/test_core.py::test_malformed_json_handling PASSED
tests/test_core.py::test_invalid_schema_recovery PASSED
tests/test_flows.py::test_flow_1_generate_and_complete_quiz PASSED
tests/test_flows.py::test_flow_2_generate_and_review_flashcards PASSED
tests/test_flows.py::test_flow_3_handle_invalid_or_empty_input PASSED
tests/test_flows.py::test_flow_4_handle_invalid_api_responses PASSED
tests/test_flows.py::test_flow_5_export_study_materials PASSED
============================= 25 passed in 1.14s ==============================
```

---

## 🏗️ Architecture & Design Decisions

### 1. Separation of Concerns
- **UI Logic (`app.py`)**: Responsible solely for rendering widgets, managing user state transitions, and invoking actions. Contains zero direct LLM or HTTP networking logic.
- **Service Layer (`llm_service.py`)**: Encapsulates LLM API invocation, structured output parsing, JSON mode fallback, error classification, and chunk distribution.
- **Domain Models (`models.py`)**: Serves as the single source of truth for all data representations. Pydantic models validate data before it can enter the UI state.
- **Utilities (`utils.py`)**: Pure helper functions with no side effects for parsing, scoring, chunking, and serialization.

### 2. Strict Schema Validation & Defense-in-Depth
- OpenAI Structured Outputs (`client.beta.chat.completions.parse`) guarantee response adherence at the model level.
- Pydantic models apply secondary validation:
  - Ensures exactly 4 choices per question.
  - Enforces case-insensitive uniqueness across distractors.
  - Normalizes letter indicators (`A`, `Choice B`, `(C)`) to the exact choice string.
  - Auto-deduplicates questions and cards, renumbering IDs sequentially.

### 3. Graceful Error Handling
- Network timeouts, rate limits (`429`), authentication errors (`401`), and JSON parsing failures return actionable user messages rather than unhandled tracebacks.
- Input validation prevents sending wasteful or empty requests to LLM endpoints.

### 4. Text Chunking for Long Materials
- Textbook chapters often exceed single-request context or generation limits. The chunking algorithm preserves paragraph breaks and sentence boundaries with an overlapping buffer to prevent truncated thoughts.
- When generating across multiple chunks, target item quotas are distributed proportionally across chunks, and the resulting items are aggregated and validated into a single coherent set.

---

## 📄 License
MIT License. Created for students, educators, and lifelong learners.
