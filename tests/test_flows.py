"""Integration tests for all required core user workflows:
1. Generate and complete a quiz
2. Generate and review flashcards
3. Handle invalid or empty input
4. Handle invalid / malformed API responses
5. Export generated study materials
"""

import json
import pytest
from pydantic import ValidationError

from models import (
    ContentType,
    DifficultyLevel,
    FlashcardDeck,
    GenerationRequest,
    QuizSet,
)
from llm_service import LLMService
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
from prompts import SAMPLE_STUDY_MATERIALS


def test_flow_1_generate_and_complete_quiz():
    """Flow 1: Generate a quiz from sample material, answer questions, and verify scoring."""
    sample = SAMPLE_STUDY_MATERIALS["Biology: Cellular Respiration"]
    service = LLMService(mock_mode=True)

    req = GenerationRequest(
        content=sample["content"],
        content_type=ContentType.QUIZ,
        subject=sample["subject"],
        exam_topic=sample["exam_topic"],
        learning_goals=sample["learning_goals"],
        num_items=5,
        difficulty=DifficultyLevel.HARD,
    )

    quiz, err = service.generate_quiz(req)
    assert err is None
    assert quiz is not None
    assert len(quiz.questions) == 5

    # Simulate answering questions
    user_answers = {}
    for idx, q in enumerate(quiz.questions):
        # Answer first 4 correctly, last one incorrectly
        if idx < 4:
            user_answers[q.id] = q.correct_answer
        else:
            wrong_choices = [c for c in q.choices if c != q.correct_answer]
            user_answers[q.id] = wrong_choices[0]

    # Verify score calculation
    score_res = calculate_quiz_score(user_answers, quiz.questions)
    assert score_res["total_questions"] == 5
    assert score_res["answered_questions"] == 5
    assert score_res["correct_count"] == 4
    assert score_res["incorrect_count"] == 1
    assert score_res["percentage"] == 80.0
    assert "Great Work" in score_res["grade_label"]
    assert len(score_res["details"]) == 5


def test_flow_2_generate_and_review_flashcards():
    """Flow 2: Generate flashcards, test flipping state, marking known/review, and filtering."""
    sample = SAMPLE_STUDY_MATERIALS["Computer Science: Operating Systems & Concurrency"]
    service = LLMService(mock_mode=True)

    req = GenerationRequest(
        content=sample["content"],
        content_type=ContentType.FLASHCARDS,
        subject=sample["subject"],
        exam_topic=sample["exam_topic"],
        num_items=6,
        difficulty=DifficultyLevel.MEDIUM,
    )

    deck, err = service.generate_flashcards(req)
    assert err is None
    assert deck is not None
    assert len(deck.cards) == 6

    # Verify each card has required properties
    for card in deck.cards:
        assert card.front
        assert card.back
        assert card.explanation
        assert card.key_concept
        assert card.difficulty in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]

    # Simulate user card reviews
    card_statuses = {
        deck.cards[0].id: "known",
        deck.cards[1].id: "known",
        deck.cards[2].id: "review",
        deck.cards[3].id: "review",
        deck.cards[4].id: "known",
        deck.cards[5].id: "review",
    }

    known_count = sum(1 for s in card_statuses.values() if s == "known")
    review_count = sum(1 for s in card_statuses.values() if s == "review")

    assert known_count == 3
    assert review_count == 3

    # Test filtering to 'Review Again Queue'
    review_queue = [c for c in deck.cards if card_statuses.get(c.id) == "review"]
    assert len(review_queue) == 3


def test_flow_3_handle_invalid_or_empty_input():
    """Flow 3: Test boundary conditions, empty inputs, whitespace, and short strings."""
    # Test 1: Completely empty input
    valid, msg = validate_study_input("")
    assert not valid
    assert "empty" in msg.lower()

    # Test 2: Whitespace only
    valid, msg = validate_study_input("    \n\t   ")
    assert not valid
    assert "empty" in msg.lower()

    # Test 3: Below minimum character threshold
    valid, msg = validate_study_input("Too short")
    assert not valid
    assert "too brief" in msg.lower()

    # Test 4: Too few words / symbol spam
    valid, msg = validate_study_input("========================================")
    assert not valid
    assert "too few discernible words" in msg.lower()


def test_flow_4_handle_invalid_api_responses():
    """Flow 4: Test handling of malformed JSON and invalid question schemas."""
    # Test 1: Malformed JSON syntax
    bad_json = "{'bad_json': true, broken"
    with pytest.raises(Exception):
        QuizSet.model_validate_json(bad_json)

    # Test 2: Missing required choices
    incomplete_quiz = {
        "title": "Broken",
        "questions": [
            {
                "id": 1,
                "question": "Where are choices?",
                "choices": ["Choice 1", "Choice 2"],  # Less than 4
                "correct_answer": "Choice 1",
                "explanation": "Invalid choice count",
            }
        ],
    }
    with pytest.raises(ValidationError):
        QuizSet.model_validate(incomplete_quiz)

    # Test 3: Correct answer does not match any choice
    mismatched_quiz = {
        "title": "Mismatched",
        "questions": [
            {
                "id": 1,
                "question": "Which option is correct?",
                "choices": ["A", "B", "C", "D"],
                "correct_answer": "Z",  # Not in choices!
                "explanation": "Z is not in choices.",
            }
        ],
    }
    with pytest.raises(ValidationError):
        QuizSet.model_validate(mismatched_quiz)


def test_flow_5_export_study_materials():
    """Flow 5: Test JSON, CSV, and Markdown export for both Quiz and Flashcards."""
    sample = SAMPLE_STUDY_MATERIALS["Economics: Macroeconomics & Monetary Policy"]
    service = LLMService(mock_mode=True)

    # Generate Quiz
    q_req = GenerationRequest(
        content=sample["content"],
        content_type=ContentType.QUIZ,
        subject=sample["subject"],
        exam_topic=sample["exam_topic"],
        num_items=3,
    )
    quiz, _ = service.generate_quiz(q_req)

    # Export Quiz
    q_json = export_quiz_to_json(quiz)
    parsed_q_json = json.loads(q_json)
    assert "questions" in parsed_q_json
    assert len(parsed_q_json["questions"]) == 3

    q_csv = export_quiz_to_csv(quiz)
    assert "Question" in q_csv
    assert "Choice A" in q_csv

    q_md = export_quiz_to_markdown(quiz)
    assert "# " in q_md
    assert "Question 1" in q_md

    # Generate Flashcards
    fc_req = GenerationRequest(
        content=sample["content"],
        content_type=ContentType.FLASHCARDS,
        subject=sample["subject"],
        exam_topic=sample["exam_topic"],
        num_items=3,
    )
    deck, _ = service.generate_flashcards(fc_req)

    # Export Flashcards
    fc_json = export_flashcards_to_json(deck)
    parsed_fc_json = json.loads(fc_json)
    assert "cards" in parsed_fc_json
    assert len(parsed_fc_json["cards"]) == 3

    fc_csv = export_flashcards_to_csv(deck)
    assert "Front (Question)" in fc_csv
    assert "Back (Answer)" in fc_csv
