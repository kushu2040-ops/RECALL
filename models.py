"""Pydantic models for the Personalized Study Flashcard & Quiz Maker.

Provides strict data validation for study requests, multiple-choice quiz questions,
flashcards, and full study sets with custom validators ensuring schema integrity.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class DifficultyLevel(str, Enum):
    """Supported difficulty levels for study materials."""
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class ContentType(str, Enum):
    """Supported output formats."""
    BOTH = "Both Flashcards & Quiz"
    QUIZ = "Multiple-Choice Quiz Only"
    FLASHCARDS = "Q&A Flashcards Only"


class GenerationSourceMode(str, Enum):
    """Source mode for generation."""
    NOTES_ONLY = "Strictly From Given Notes / File"
    TOPIC_EXPANDED = "From Subject & Topic (AI Curriculum Knowledge)"
    HYBRID = "Hybrid (Notes + Broad Topic Applications)"


class GenerationRequest(BaseModel):
    """Configuration and input content for study material generation."""
    content: str = Field(..., min_length=20, description="Source study material or notes.")
    content_type: ContentType = Field(default=ContentType.QUIZ, description="Target study format.")
    source_mode: GenerationSourceMode = Field(
        default=GenerationSourceMode.NOTES_ONLY,
        description="Source grounding mode for questions."
    )
    subject: Optional[str] = Field(default="", description="Subject or discipline (e.g. Biology, CS).")
    exam_topic: Optional[str] = Field(default="", description="Target exam or chapter topic.")
    learning_goals: Optional[str] = Field(default="", description="Desired learning outcomes.")
    num_items: int = Field(default=5, ge=1, le=40, description="Number of items to generate.")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM, description="Target difficulty level.")
    focus_instructions: Optional[str] = Field(default="", description="Specific subtopics or guidelines.")


class QuizQuestion(BaseModel):
    """A single multiple-choice quiz question with exactly four choices."""
    id: int = Field(default=1, description="Question sequential index.")
    question: str = Field(..., min_length=5, description="Clear, concise question prompt.")
    choices: List[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="List of exactly four distinct answer choices."
    )
    correct_answer: str = Field(
        ...,
        min_length=1,
        description="The correct answer text, matching one of the four choices."
    )
    explanation: str = Field(
        ...,
        min_length=5,
        description="Pedagogical explanation of why the correct answer is right and distractors are wrong."
    )
    reference_concept: Optional[str] = Field(
        default="General Concept",
        description="Key textbook concept or section reference."
    )

    @field_validator("choices")
    @classmethod
    def validate_choices(cls, v: List[str]) -> List[str]:
        cleaned = [c.strip() for c in v if c and c.strip()]
        if len(cleaned) != 4:
            raise ValueError(f"Quiz question must have exactly 4 non-empty choices, got {len(cleaned)}")
        # Check for case-insensitive uniqueness
        lower_seen = set()
        for choice in cleaned:
            c_low = choice.lower()
            if c_low in lower_seen:
                raise ValueError(f"Duplicate choice detected: '{choice}'")
            lower_seen.add(c_low)
        return cleaned

    @model_validator(mode="after")
    def validate_correct_answer(self) -> "QuizQuestion":
        ans = self.correct_answer.strip()
        choices = self.choices

        # Check if the answer directly matches one choice (case-insensitive)
        matched_choice = None
        for choice in choices:
            if ans.lower() == choice.lower():
                matched_choice = choice
                break

        # Check if the answer is a letter indicator like "A", "B", "C", "D" or "A) ..."
        if not matched_choice:
            letter_map = {"a": 0, "b": 1, "c": 2, "d": 3}
            # Check single letter or "Choice A" or "A)" or "(A)"
            cleaned_ans = ans.replace("(", "").replace(")", "").replace(".", "").strip().lower()
            if cleaned_ans in letter_map:
                idx = letter_map[cleaned_ans]
                if idx < len(choices):
                    matched_choice = choices[idx]
            elif cleaned_ans.startswith("choice ") and cleaned_ans[7:].strip() in letter_map:
                idx = letter_map[cleaned_ans[7:].strip()]
                if idx < len(choices):
                    matched_choice = choices[idx]
            else:
                # Try prefix check, e.g. "A) Mitochondria"
                for idx, letter in enumerate(["a", "b", "c", "d"]):
                    prefix = f"{letter}:"
                    prefix2 = f"{letter})"
                    if cleaned_ans.startswith(prefix) or cleaned_ans.startswith(prefix2):
                        matched_choice = choices[idx]
                        break

        if not matched_choice:
            raise ValueError(
                f"Correct answer '{self.correct_answer}' does not match any of the 4 choices: {choices}"
            )

        # Standardize correct_answer to the exact string in choices
        self.correct_answer = matched_choice
        return self


class QuizSet(BaseModel):
    """A collection of quiz questions with metadata."""
    title: str = Field(default="Personalized Study Quiz", description="Quiz title.")
    subject: Optional[str] = Field(default="", description="Subject or course name.")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM, description="Overall difficulty.")
    questions: List[QuizQuestion] = Field(..., min_length=1, description="List of validated questions.")

    @model_validator(mode="after")
    def ensure_unique_questions(self) -> "QuizSet":
        seen = set()
        unique_questions = []
        for q in self.questions:
            q_text_norm = q.question.strip().lower()
            if q_text_norm not in seen:
                seen.add(q_text_norm)
                q.id = len(unique_questions) + 1
                unique_questions.append(q)
        if not unique_questions:
            raise ValueError("QuizSet must contain at least one valid question.")
        self.questions = unique_questions
        return self


class Flashcard(BaseModel):
    """A single flashcard for spaced repetition and self-testing."""
    id: int = Field(default=1, description="Card sequential index.")
    front: str = Field(..., min_length=3, description="Question or prompt on the front of the card.")
    back: str = Field(..., min_length=1, description="Answer or core fact on the back of the card.")
    explanation: str = Field(default="", description="In-depth contextual explanation.")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM, description="Card difficulty level.")
    key_concept: str = Field(default="Core Concept", description="Categorization or concept tag.")

    @field_validator("front", "back")
    @classmethod
    def strip_text(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Flashcard front and back cannot be blank.")
        return s


class FlashcardDeck(BaseModel):
    """A collection of flashcards with metadata."""
    title: str = Field(default="Personalized Study Flashcards", description="Deck title.")
    subject: Optional[str] = Field(default="", description="Subject or course name.")
    cards: List[Flashcard] = Field(..., min_length=1, description="List of validated flashcards.")

    @model_validator(mode="after")
    def ensure_unique_cards(self) -> "FlashcardDeck":
        seen = set()
        unique_cards = []
        for card in self.cards:
            front_norm = card.front.strip().lower()
            if front_norm not in seen:
                seen.add(front_norm)
                card.id = len(unique_cards) + 1
                unique_cards.append(card)
        if not unique_cards:
            raise ValueError("FlashcardDeck must contain at least one valid card.")
        self.cards = unique_cards
        return self
