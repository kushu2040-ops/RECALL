"""Utility helpers for input validation, text chunking, quiz scoring, and data export.
"""

import csv
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from models import FlashcardDeck, QuizQuestion, QuizSet


def validate_study_input(text: str, min_chars: int = 30) -> Tuple[bool, Optional[str]]:
    """Validate user input content for minimum substance and readability.

    Args:
        text: Raw user notes or textbook excerpt.
        min_chars: Minimum character threshold.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not text or not text.strip():
        return False, "Input cannot be empty. Please paste your lecture notes or textbook material."

    cleaned = text.strip()
    if len(cleaned) < min_chars:
        return (
            False,
            f"Input is too brief ({len(cleaned)} characters). Please provide at least {min_chars} characters of substantive study material.",
        )

    # Check if input has at least a few words rather than repeated symbols
    words = re.findall(r"\b\w+\b", cleaned)
    if len(words) < 5:
        return False, "Input contains too few discernible words to generate meaningful study items."

    return True, None


def chunk_text(text: str, max_chunk_chars: int = 4000, overlap_chars: int = 300) -> List[str]:
    """Split large text into overlapping chunks respecting paragraph or sentence boundaries.

    Args:
        text: Full source content.
        max_chunk_chars: Maximum characters per chunk.
        overlap_chars: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    cleaned = text.strip()
    if len(cleaned) <= max_chunk_chars:
        return [cleaned]

    # Split primarily on double linebreaks (paragraphs)
    paragraphs = re.split(r"\n\s*\n", cleaned)
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue

        # If a single paragraph is longer than max_chunk_chars, split by sentence
        if len(para_clean) > max_chunk_chars:
            sentences = re.split(r"(?<=[.!?])\s+", para_clean)
            for sentence in sentences:
                sent_len = len(sentence)
                if current_length + sent_len > max_chunk_chars and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    # Retain overlap from end of chunk
                    overlap_text = current_chunk[-1] if len(current_chunk[-1]) <= overlap_chars else sentence
                    current_chunk = [overlap_text, sentence]
                    current_length = len(overlap_text) + sent_len
                else:
                    current_chunk.append(sentence)
                    current_length += sent_len + 1
        else:
            para_len = len(para_clean)
            if current_length + para_len > max_chunk_chars and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Retain tail paragraph for overlap if within bounds
                overlap_text = current_chunk[-1] if len(current_chunk[-1]) <= overlap_chars else ""
                if overlap_text:
                    current_chunk = [overlap_text, para_clean]
                    current_length = len(overlap_text) + para_len + 2
                else:
                    current_chunk = [para_clean]
                    current_length = para_len
            else:
                current_chunk.append(para_clean)
                current_length += para_len + 2

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks if chunks else [cleaned]


def calculate_quiz_score(
    user_answers: Dict[int, str],
    questions: List[QuizQuestion],
) -> Dict[str, Any]:
    """Calculate aggregate and per-question score metrics.

    Args:
        user_answers: Mapping of question ID to the user's selected choice.
        questions: List of QuizQuestion objects.

    Returns:
        Dictionary with score breakdown and feedback.
    """
    total = len(questions)
    if total == 0:
        return {
            "total_questions": 0,
            "answered_questions": 0,
            "correct_count": 0,
            "incorrect_count": 0,
            "percentage": 0.0,
            "grade_label": "No Questions",
            "is_perfect": False,
            "details": [],
        }

    correct_count = 0
    answered_count = 0
    details = []

    for q in questions:
        user_ans = user_answers.get(q.id)
        is_answered = user_ans is not None
        if is_answered:
            answered_count += 1

        is_correct = False
        if is_answered:
            is_correct = user_ans.strip().lower() == q.correct_answer.strip().lower()
            if is_correct:
                correct_count += 1

        details.append({
            "question_id": q.id,
            "question": q.question,
            "user_answer": user_ans,
            "correct_answer": q.correct_answer,
            "is_correct": is_correct,
            "is_answered": is_answered,
            "explanation": q.explanation,
            "reference_concept": q.reference_concept,
        })

    percentage = round((correct_count / total) * 100.0, 1)

    if percentage == 100.0:
        grade_label = "Mastery Achieved! Outstanding job!"
    elif percentage >= 80.0:
        grade_label = "Great Work! Strong conceptual understanding."
    elif percentage >= 60.0:
        grade_label = "Good Effort! Review missed concepts to solidify retention."
    else:
        grade_label = "Keep Practicing! Review the explanations below and try again."

    return {
        "total_questions": total,
        "answered_questions": answered_count,
        "correct_count": correct_count,
        "incorrect_count": total - correct_count,
        "percentage": percentage,
        "grade_label": grade_label,
        "is_perfect": percentage == 100.0,
        "details": details,
    }


def export_flashcards_to_json(deck: FlashcardDeck) -> str:
    """Serialize flashcard deck to formatted JSON string."""
    data = {
        "title": deck.title,
        "subject": deck.subject,
        "total_cards": len(deck.cards),
        "cards": [
            {
                "id": card.id,
                "front": card.front,
                "back": card.back,
                "explanation": card.explanation,
                "difficulty": card.difficulty.value,
                "key_concept": card.key_concept,
            }
            for card in deck.cards
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def export_flashcards_to_csv(deck: FlashcardDeck) -> str:
    """Serialize flashcard deck to CSV string (Anki and spreadsheet compatible)."""
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["ID", "Front (Question)", "Back (Answer)", "Explanation", "Difficulty", "Key Concept"])

    for card in deck.cards:
        writer.writerow([
            card.id,
            card.front,
            card.back,
            card.explanation,
            card.difficulty.value,
            card.key_concept,
        ])

    return output.getvalue()


def export_quiz_to_json(quiz: QuizSet) -> str:
    """Serialize quiz set to formatted JSON string."""
    data = {
        "title": quiz.title,
        "subject": quiz.subject,
        "difficulty": quiz.difficulty.value,
        "total_questions": len(quiz.questions),
        "questions": [
            {
                "id": q.id,
                "question": q.question,
                "choices": q.choices,
                "correct_answer": q.correct_answer,
                "explanation": q.explanation,
                "reference_concept": q.reference_concept,
            }
            for q in quiz.questions
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def export_quiz_to_csv(quiz: QuizSet) -> str:
    """Serialize quiz set to CSV string."""
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    writer.writerow([
        "ID",
        "Question",
        "Choice A",
        "Choice B",
        "Choice C",
        "Choice D",
        "Correct Answer",
        "Explanation",
        "Reference Concept",
    ])

    for q in quiz.questions:
        c = q.choices + [""] * (4 - len(q.choices))
        writer.writerow([
            q.id,
            q.question,
            c[0],
            c[1],
            c[2],
            c[3],
            q.correct_answer,
            q.explanation,
            q.reference_concept,
        ])

    return output.getvalue()


def export_quiz_to_markdown(quiz: QuizSet) -> str:
    """Serialize quiz set to a formatted Markdown study guide."""
    lines = [
        f"# {quiz.title}",
        f"**Subject:** {quiz.subject or 'General'} | **Difficulty:** {quiz.difficulty.value}",
        f"**Total Questions:** {len(quiz.questions)}\n",
        "---",
    ]

    for q in quiz.questions:
        lines.append(f"\n### Question {q.id}: {q.question}")
        letters = ["A", "B", "C", "D"]
        for idx, choice in enumerate(q.choices):
            marker = " (✓ Correct Answer)" if choice == q.correct_answer else ""
            lines.append(f"- **{letters[idx]}**: {choice}{marker}")
        lines.append(f"\n> **Explanation:** {q.explanation}")
        if q.reference_concept:
            lines.append(f"> **Concept:** {q.reference_concept}")

    return "\n".join(lines)
