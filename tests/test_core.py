"""Unit tests for Personalized Study Flashcard & Quiz Maker.

Covers schema validation, scoring, chunking, input validation, export helpers,
and mock LLM generation.
"""

import json
import pytest
from pydantic import ValidationError

from models import (
    ContentType,
    DifficultyLevel,
    Flashcard,
    FlashcardDeck,
    GenerationRequest,
    QuizQuestion,
    QuizSet,
)
from utils import (
    calculate_quiz_score,
    chunk_text,
    export_flashcards_to_csv,
    export_flashcards_to_json,
    export_quiz_to_csv,
    export_quiz_to_json,
    export_quiz_to_markdown,
    validate_study_input,
)
from llm_service import LLMService, resolve_api_key


# ===========================================================================
# 1. Schema Validation Tests
# ===========================================================================
def test_valid_quiz_question():
    """Ensure a properly formed QuizQuestion validates and normalizes."""
    q = QuizQuestion(
        id=1,
        question="What is the net ATP yield of glycolysis?",
        choices=["2 ATP", "4 ATP", "32 ATP", "36 ATP"],
        correct_answer="2 ATP",
        explanation="Glycolysis produces 4 ATP gross but consumes 2 ATP, yielding net 2 ATP.",
        reference_concept="Glycolysis",
    )
    assert q.id == 1
    assert len(q.choices) == 4
    assert q.correct_answer == "2 ATP"


def test_quiz_question_letter_answer_normalization():
    """Ensure letter indicators like 'A', 'B', 'Choice C' normalize to choice text."""
    q1 = QuizQuestion(
        id=1,
        question="Which molecule carries electrons to the electron transport chain?",
        choices=["NADH", "Glucose", "H2O", "Pyruvate"],
        correct_answer="A",
        explanation="NADH carries high-energy electrons.",
    )
    assert q1.correct_answer == "NADH"

    q2 = QuizQuestion(
        id=2,
        question="Which is an anaerobic stage?",
        choices=["Krebs Cycle", "Glycolysis", "Oxidative Phosphorylation", "ETC"],
        correct_answer="Choice B",
        explanation="Glycolysis does not require oxygen.",
    )
    assert q2.correct_answer == "Glycolysis"


def test_quiz_question_rejects_invalid_choice_count():
    """Ensure validation fails if choices list does not have exactly 4 items."""
    with pytest.raises(ValidationError):
        QuizQuestion(
            question="Invalid question?",
            choices=["A", "B", "C"],  # Only 3
            correct_answer="A",
            explanation="Explanation",
        )

    with pytest.raises(ValidationError):
        QuizQuestion(
            question="Invalid question?",
            choices=["A", "B", "C", "D", "E"],  # 5 choices
            correct_answer="A",
            explanation="Explanation",
        )


def test_quiz_question_rejects_duplicate_choices():
    """Ensure validation fails if duplicate choices exist."""
    with pytest.raises(ValidationError):
        QuizQuestion(
            question="Duplicate choices test",
            choices=["ATP", "atp", "NADH", "FADH2"],
            correct_answer="ATP",
            explanation="Explanation",
        )


def test_quiz_question_rejects_unmatched_correct_answer():
    """Ensure validation fails if correct answer is not among the 4 choices."""
    with pytest.raises(ValidationError):
        QuizQuestion(
            question="Unmatched answer test",
            choices=["Option 1", "Option 2", "Option 3", "Option 4"],
            correct_answer="Completely unrelated answer",
            explanation="Explanation",
        )


def test_quiz_set_deduplication_and_indexing():
    """Ensure QuizSet automatically eliminates duplicate questions and assigns sequential IDs."""
    q1 = QuizQuestion(
        id=10,
        question="What is ATP?",
        choices=["Energy currency", "Vitamin", "Lipid", "Mineral"],
        correct_answer="Energy currency",
        explanation="ATP powers cellular work.",
    )
    q2 = QuizQuestion(
        id=20,
        question="What is ATP?",  # Duplicate question
        choices=["Energy currency", "Vitamin", "Lipid", "Mineral"],
        correct_answer="Energy currency",
        explanation="ATP powers cellular work.",
    )
    q3 = QuizQuestion(
        id=30,
        question="Where does glycolysis happen?",
        choices=["Cytoplasm", "Nucleus", "Ribosome", "Golgi"],
        correct_answer="Cytoplasm",
        explanation="It takes place in the cytosol.",
    )

    qs = QuizSet(title="Biology Quiz", questions=[q1, q2, q3])
    assert len(qs.questions) == 2
    assert qs.questions[0].id == 1
    assert qs.questions[1].id == 2


def test_flashcard_validation():
    """Ensure Flashcard validates and rejects blank fronts or backs."""
    card = Flashcard(
        id=1,
        front="What is a mutex?",
        back="Mutual exclusion lock.",
        explanation="Guarantees exclusive resource access.",
        difficulty=DifficultyLevel.EASY,
        key_concept="Concurrency",
    )
    assert card.front == "What is a mutex?"

    with pytest.raises(ValidationError):
        Flashcard(front="   ", back="Non-blank back")

    with pytest.raises(ValidationError):
        Flashcard(front="Valid front", back="   ")


def test_flashcard_deck_deduplication():
    """Ensure FlashcardDeck eliminates duplicate cards and re-indexes."""
    c1 = Flashcard(id=5, front="Term A", back="Def A")
    c2 = Flashcard(id=6, front="term a", back="Def A alternate")
    c3 = Flashcard(id=7, front="Term B", back="Def B")

    deck = FlashcardDeck(title="Test Deck", cards=[c1, c2, c3])
    assert len(deck.cards) == 2
    assert deck.cards[0].id == 1
    assert deck.cards[1].id == 2


# ===========================================================================
# 2. Input Validation Tests
# ===========================================================================
def test_validate_study_input():
    """Test various input strings for validation rules."""
    # Empty string
    valid, msg = validate_study_input("")
    assert not valid
    assert "cannot be empty" in msg

    # Too short
    valid, msg = validate_study_input("Hello there")
    assert not valid
    assert "too brief" in msg

    # Too few words / gibberish symbols
    valid, msg = validate_study_input("----------------------------------------")
    assert not valid
    assert "too few discernible words" in msg

    # Valid substance
    sample = "Cellular respiration breaks down glucose to generate ATP and water through metabolic pathways."
    valid, msg = validate_study_input(sample)
    assert valid
    assert msg is None


# ===========================================================================
# 3. Text Chunking Tests
# ===========================================================================
def test_chunk_text_short():
    """Short text should return as a single chunk without alterations."""
    short_text = "This is a brief study guide on basic algebra principles."
    chunks = chunk_text(short_text, max_chunk_chars=500)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_chunk_text_long_paragraphs():
    """Long text with paragraphs should be split into multiple chunks cleanly."""
    para1 = "Paragraph 1: " + "Biological systems maintain homeostasis through regulatory feedback. " * 15
    para2 = "Paragraph 2: " + "Negative feedback loops counteract changes to preserve set points. " * 15
    para3 = "Paragraph 3: " + "Positive feedback amplifies physiological responses until a terminal event. " * 15
    full_text = f"{para1}\n\n{para2}\n\n{para3}"

    chunks = chunk_text(full_text, max_chunk_chars=600, overlap_chars=100)
    assert len(chunks) >= 2
    # Ensure every chunk has meaningful length
    for c in chunks:
        assert len(c) > 50


# ===========================================================================
# 4. Quiz Scoring Tests
# ===========================================================================
def test_calculate_quiz_score():
    """Verify scoring logic for full, partial, and zero score cases."""
    q1 = QuizQuestion(
        id=1,
        question="What is the first question?",
        choices=["A", "B", "C", "D"],
        correct_answer="A",
        explanation="Exp 1",
    )
    q2 = QuizQuestion(
        id=2,
        question="What is the second question?",
        choices=["A", "B", "C", "D"],
        correct_answer="B",
        explanation="Exp 2",
    )

    questions = [q1, q2]

    # Perfect Score
    res_perfect = calculate_quiz_score({1: "A", 2: "B"}, questions)
    assert res_perfect["total_questions"] == 2
    assert res_perfect["correct_count"] == 2
    assert res_perfect["percentage"] == 100.0
    assert res_perfect["is_perfect"] is True

    # 50% Score
    res_half = calculate_quiz_score({1: "A", 2: "C"}, questions)
    assert res_half["correct_count"] == 1
    assert res_half["incorrect_count"] == 1
    assert res_half["percentage"] == 50.0

    # Unanswered / Partial
    res_partial = calculate_quiz_score({1: "A"}, questions)
    assert res_partial["answered_questions"] == 1
    assert res_partial["correct_count"] == 1

    # Empty answers
    res_empty = calculate_quiz_score({}, questions)
    assert res_empty["correct_count"] == 0
    assert res_empty["percentage"] == 0.0


# ===========================================================================
# 5. Export Serialization Tests
# ===========================================================================
def test_export_quiz_formats():
    """Verify JSON, CSV, and Markdown exports for QuizSet."""
    q = QuizQuestion(
        id=1,
        question="What is 2 + 2?",
        choices=["3", "4", "5", "6"],
        correct_answer="4",
        explanation="Basic arithmetic.",
        reference_concept="Arithmetic",
    )
    quiz = QuizSet(title="Math Quiz", subject="Math", questions=[q])

    # JSON export
    json_str = export_quiz_to_json(quiz)
    parsed = json.loads(json_str)
    assert parsed["title"] == "Math Quiz"
    assert len(parsed["questions"]) == 1
    assert parsed["questions"][0]["correct_answer"] == "4"

    # CSV export
    csv_str = export_quiz_to_csv(quiz)
    assert "What is 2 + 2?" in csv_str
    assert "Choice A" in csv_str
    assert "Arithmetic" in csv_str

    # Markdown export
    md_str = export_quiz_to_markdown(quiz)
    assert "# Math Quiz" in md_str
    assert "Question 1" in md_str
    assert "(✓ Correct Answer)" in md_str


def test_export_flashcards_formats():
    """Verify JSON and CSV exports for FlashcardDeck."""
    card = Flashcard(
        id=1,
        front="What is DNS?",
        back="Domain Name System",
        explanation="Maps hostnames to IP addresses.",
        key_concept="Networking",
    )
    deck = FlashcardDeck(title="Networking Deck", subject="CS", cards=[card])

    # JSON export
    json_str = export_flashcards_to_json(deck)
    parsed = json.loads(json_str)
    assert parsed["title"] == "Networking Deck"
    assert len(parsed["cards"]) == 1
    assert parsed["cards"][0]["front"] == "What is DNS?"

    # CSV export
    csv_str = export_flashcards_to_csv(deck)
    assert "What is DNS?" in csv_str
    assert "Domain Name System" in csv_str
    assert "Networking" in csv_str


# ===========================================================================
# 6. LLM Service & Mock Generation Tests
# ===========================================================================
def test_mock_quiz_generation():
    """Verify mock generator outputs valid, compliant QuizSet."""
    service = LLMService(mock_mode=True)
    req = GenerationRequest(
        content="Cellular respiration is a metabolic pathway that breaks down glucose and produces ATP.",
        content_type=ContentType.QUIZ,
        subject="Biology",
        exam_topic="Cellular Respiration",
        num_items=4,
        difficulty=DifficultyLevel.HARD,
    )
    quiz, err = service.generate_quiz(req)
    assert err is None
    assert quiz is not None
    assert len(quiz.questions) == 4
    for q in quiz.questions:
        assert len(q.choices) == 4
        assert q.correct_answer in q.choices
        assert len(q.explanation) > 5


def test_mock_flashcard_generation():
    """Verify mock generator outputs valid, compliant FlashcardDeck."""
    service = LLMService(mock_mode=True)
    req = GenerationRequest(
        content="Concurrent processes share resources. Mutex locks and semaphores prevent race conditions.",
        content_type=ContentType.FLASHCARDS,
        subject="Computer Science",
        exam_topic="Operating Systems",
        num_items=5,
        difficulty=DifficultyLevel.MEDIUM,
    )
    deck, err = service.generate_flashcards(req)
    assert err is None
    assert deck is not None
    assert len(deck.cards) == 5
    for c in deck.cards:
        assert len(c.front) > 3
        assert len(c.back) > 0
        assert len(c.key_concept) > 0


def test_service_handles_empty_input_gracefully():
    """Verify service rejects empty or too-short input with helpful error."""
    service = LLMService(mock_mode=True)
    # Pydantic schema rejects short content (< 20 chars)
    with pytest.raises(ValidationError):
        GenerationRequest(
            content="Short text",
            num_items=3,
        )

    # validate_study_input rejects < 30 chars
    valid, err = validate_study_input("Short text")
    assert not valid
    assert "too brief" in err


def test_resolve_api_key(monkeypatch):
    """Verify API key resolution precedence."""
    # Direct argument takes highest precedence
    assert resolve_api_key("direct_key") == "direct_key"

    # Environment variable fallback
    monkeypatch.setenv("GEMINI_API_KEY", "gemini_secret_key")
    # Patch dotenv to avoid re-reading .env file during unit test
    monkeypatch.setattr("llm_service.load_dotenv", lambda **kwargs: None)
    assert resolve_api_key(None) == "gemini_secret_key"


def test_malformed_json_handling():
    """Verify that malformed JSON strings fail gracefully with Pydantic."""
    malformed_json = '{"title": "Test", "questions": [{"id": 1, "question": "Incomplete'
    with pytest.raises(Exception):
        QuizSet.model_validate_json(malformed_json)


def test_invalid_schema_recovery():
    """Verify that an invalid schema missing choices raises validation error."""
    invalid_data = {
        "title": "Bad Quiz",
        "questions": [
            {
                "id": 1,
                "question": "Where are choices?",
                "choices": ["Only One"],
                "correct_answer": "Only One",
                "explanation": "Missing 3 choices",
            }
        ],
    }
    with pytest.raises(ValidationError):
        QuizSet.model_validate(invalid_data)

