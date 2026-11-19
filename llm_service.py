"""LLM service handling Google Gemini API calls (via official google-genai SDK),
structured outputs with Pydantic validation, chunked text processing,
and mock generator for offline/demo use.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(override=True)

from models import (
    ContentType,
    DifficultyLevel,
    Flashcard,
    FlashcardDeck,
    GenerationRequest,
    QuizQuestion,
    QuizSet,
)
from prompts import (
    FLASHCARD_SYSTEM_PROMPT,
    QUIZ_SYSTEM_PROMPT,
    build_flashcard_prompt,
    build_quiz_prompt,
)
from utils import chunk_text, validate_study_input

logger = logging.getLogger(__name__)


def resolve_api_key(provided_key: Optional[str] = None) -> Optional[str]:
    """Resolve Gemini API key with precedence: passed arg -> env var -> Streamlit secrets."""
    load_dotenv(override=True)
    if provided_key and provided_key.strip():
        return provided_key.strip()

    # Search for GEMINI_API_KEY, GOOGLE_API_KEY, or legacy OPENAI_API_KEY
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(env_name, "").strip()
        if val:
            return val

    # Attempt to read from streamlit secrets if available
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for sec_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
                if sec_name in st.secrets:
                    sec_key = st.secrets[sec_name]
                    if sec_key and str(sec_key).strip():
                        return str(sec_key).strip()
    except Exception:
        pass

    return None


def format_api_error(err: Exception, api_key: Optional[str] = None) -> str:
    """Produce clear, actionable diagnostic messages for Gemini API failures."""
    err_str = str(err)
    err_lower = err_str.lower()
    key_str = (api_key or "").strip()

    # OpenAI Key entered warning
    if key_str.startswith("sk-"):
        return (
            "Detected an OpenAI key ('sk-...'). The app is now powered by Google Gemini.\n\n"
            "Please paste a Google Gemini API key from https://aistudio.google.com/app/apikey (starts with 'AIzaSy...'), "
            "or use Demo Mode below."
        )

    # 403 / API Key Invalid
    if "api_key_invalid" in err_lower or "403" in err_lower or "permission_denied" in err_lower:
        return (
            "Invalid Google Gemini API Key (HTTP 403).\n\n"
            "Please verify that you copied the complete key from https://aistudio.google.com/app/apikey."
        )

    # 429 / Quota / Rate limit
    if "resource_exhausted" in err_lower or "429" in err_lower or "quota" in err_lower:
        return (
            "Gemini API Rate Limit / Quota Exceeded (HTTP 429: RESOURCE_EXHAUSTED).\n\n"
            "Please wait a few seconds and try again, or switch to 'Demo Mode (Offline)' in the sidebar."
        )

    # 404 / Model Not Found / Deprecated
    if "not_found" in err_lower or "404" in err_lower or "no longer available" in err_lower:
        return (
            "Gemini Model Deprecated or Not Found (HTTP 404).\n\n"
            "The requested model is no longer available. "
            "Please select 'gemini-3.5-flash-lite' or 'gemini-3.8-flash' in the sidebar under LLM Settings."
        )

    return f"Gemini API Error: {err_str}"


class LLMService:
    """Manages Google Gemini API interactions with structured outputs and offline mock fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "gemini-3.5-flash-lite",
        mock_mode: bool = False,
    ):
        self.api_key = resolve_api_key(api_key)
        # Automatically upgrade deprecated model codes to gemini-3.5-flash-lite
        if model_name in ("gemini-2.5-flash", "models/gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-pro"):
            self.model_name = "gemini-3.5-flash-lite"
        else:
            self.model_name = model_name or "gemini-3.5-flash-lite"
        self.mock_mode = mock_mode
        self.base_url = base_url

        self._client = None
        if not self.mock_mode and self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Google GenAI Client: {e}")
                self._client = None

    @property
    def is_configured(self) -> bool:
        """Check if an active API key or mock mode is ready."""
        return bool(self.mock_mode or (self.api_key and self._client is not None))

    def generate_quiz(self, request: GenerationRequest) -> Tuple[Optional[QuizSet], Optional[str]]:
        """Generate a validated QuizSet from user content with chunking support."""
        valid, err = validate_study_input(request.content)
        if not valid:
            return None, err

        if self.mock_mode or not self._client:
            return self._generate_mock_quiz(request), None

        chunks = chunk_text(request.content, max_chunk_chars=8000)

        # Single chunk path
        if len(chunks) == 1:
            return self._call_quiz_api(request, chunks[0], num_items=request.num_items)

        # Multi-chunk path: distribute target items across chunks
        total_items = request.num_items
        num_chunks = len(chunks)
        items_per_chunk = max(1, total_items // num_chunks)
        remainder = total_items % num_chunks

        all_questions: List[QuizQuestion] = []
        for i, chunk in enumerate(chunks):
            chunk_items = items_per_chunk + (1 if i < remainder else 0)
            if chunk_items <= 0:
                continue

            q_set, err = self._call_quiz_api(request, chunk, num_items=chunk_items)
            if err:
                logger.warning(f"Error on chunk {i + 1}/{num_chunks}: {err}")
                continue
            if q_set and q_set.questions:
                all_questions.extend(q_set.questions)

        if not all_questions:
            return None, "Failed to generate quiz questions from any text chunks. Please verify your Gemini API key and input."

        try:
            merged_quiz = QuizSet(
                title=f"{request.exam_topic or request.subject or 'Personalized'} Study Quiz",
                subject=request.subject or "General",
                difficulty=request.difficulty,
                questions=all_questions[:total_items],
            )
            return merged_quiz, None
        except Exception as e:
            return None, f"Schema validation failed when assembling quiz: {e}"

    def generate_flashcards(self, request: GenerationRequest) -> Tuple[Optional[FlashcardDeck], Optional[str]]:
        """Generate a validated FlashcardDeck from user content with chunking support."""
        valid, err = validate_study_input(request.content)
        if not valid:
            return None, err

        if self.mock_mode or not self._client:
            return self._generate_mock_flashcards(request), None

        chunks = chunk_text(request.content, max_chunk_chars=8000)

        # Single chunk path
        if len(chunks) == 1:
            return self._call_flashcard_api(request, chunks[0], num_items=request.num_items)

        # Multi-chunk path: distribute target items across chunks
        total_items = request.num_items
        num_chunks = len(chunks)
        items_per_chunk = max(1, total_items // num_chunks)
        remainder = total_items % num_chunks

        all_cards: List[Flashcard] = []
        for i, chunk in enumerate(chunks):
            chunk_items = items_per_chunk + (1 if i < remainder else 0)
            if chunk_items <= 0:
                continue

            deck, err = self._call_flashcard_api(request, chunk, num_items=chunk_items)
            if err:
                logger.warning(f"Error on chunk {i + 1}/{num_chunks}: {err}")
                continue
            if deck and deck.cards:
                all_cards.extend(deck.cards)

        if not all_cards:
            return None, "Failed to generate flashcards from text chunks. Please verify your Gemini API key and input."

        try:
            merged_deck = FlashcardDeck(
                title=f"{request.exam_topic or request.subject or 'Personalized'} Flashcards",
                subject=request.subject or "General",
                cards=all_cards[:total_items],
            )
            return merged_deck, None
        except Exception as e:
            return None, f"Schema validation failed when assembling flashcard deck: {e}"

    def _invoke_gemini(
        self,
        prompt: str,
        system_instruction: str,
        response_schema: Any,
    ):
        """Invoke Gemini API with automatic model migration and fallback."""
        from google.genai import types

        model_to_use = self.model_name
        if model_to_use in ("gemini-2.5-flash", "models/gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-pro"):
            model_to_use = "gemini-3.5-flash-lite"
            self.model_name = "gemini-3.5-flash-lite"

        try:
            return self._client.models.generate_content(
                model=model_to_use,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.7,
                ),
            )
        except Exception as err:
            err_str = str(err)
            if "404" in err_str or "NOT_FOUND" in err_str or "no longer available" in err_str:
                # If 3.5-flash-lite failed, fallback to 3.8-flash; otherwise try 3.5-flash-lite
                fallback_target = "gemini-3.8-flash" if model_to_use == "gemini-3.5-flash-lite" else "gemini-3.5-flash-lite"
                logger.warning(f"Model {model_to_use} unavailable (404/deprecated). Retrying with {fallback_target}...")
                self.model_name = fallback_target
                return self._client.models.generate_content(
                    model=fallback_target,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.7,
                    ),
                )
            raise err

    def _call_quiz_api(
        self,
        request: GenerationRequest,
        text_chunk: str,
        num_items: int,
    ) -> Tuple[Optional[QuizSet], Optional[str]]:
        """Execute Google Gemini API call with native Structured Outputs for QuizSet."""
        req_copy = request.model_copy(update={"num_items": num_items})
        prompt = build_quiz_prompt(req_copy, text_chunk)

        try:
            response = self._invoke_gemini(
                prompt=prompt,
                system_instruction=QUIZ_SYSTEM_PROMPT,
                response_schema=QuizSet,
            )

            # 1. Direct parsed object
            if hasattr(response, "parsed") and response.parsed:
                parsed_quiz = response.parsed
                if isinstance(parsed_quiz, QuizSet) and parsed_quiz.questions:
                    return parsed_quiz, None

            # 2. Parse from response.text JSON string
            raw_text = getattr(response, "text", None) or ""
            if not raw_text.strip():
                return None, "Gemini API returned an empty response body."

            clean_json = raw_text.strip()
            if clean_json.startswith("```"):
                clean_json = re.sub(r"^```(?:json)?\n?", "", clean_json)
                clean_json = re.sub(r"\n?```$", "", clean_json)

            data = json.loads(clean_json)
            if "questions" not in data and "quiz" in data:
                data = data["quiz"]
            elif "questions" not in data and "items" in data:
                data["questions"] = data["items"]

            quiz = QuizSet.model_validate(data)
            return quiz, None

        except json.JSONDecodeError as jde:
            return None, f"Gemini returned malformed JSON: {jde}"
        except Exception as err:
            return None, format_api_error(err, self.api_key)

    def _call_flashcard_api(
        self,
        request: GenerationRequest,
        text_chunk: str,
        num_items: int,
    ) -> Tuple[Optional[FlashcardDeck], Optional[str]]:
        """Execute Google Gemini API call with native Structured Outputs for FlashcardDeck."""
        req_copy = request.model_copy(update={"num_items": num_items})
        prompt = build_flashcard_prompt(req_copy, text_chunk)

        try:
            response = self._invoke_gemini(
                prompt=prompt,
                system_instruction=FLASHCARD_SYSTEM_PROMPT,
                response_schema=FlashcardDeck,
            )

            if hasattr(response, "parsed") and response.parsed:
                parsed_deck = response.parsed
                if isinstance(parsed_deck, FlashcardDeck) and parsed_deck.cards:
                    return parsed_deck, None

            raw_text = getattr(response, "text", None) or ""
            if not raw_text.strip():
                return None, "Gemini API returned an empty response body."

            clean_json = raw_text.strip()
            if clean_json.startswith("```"):
                clean_json = re.sub(r"^```(?:json)?\n?", "", clean_json)
                clean_json = re.sub(r"\n?```$", "", clean_json)

            data = json.loads(clean_json)
            if "cards" not in data and "flashcards" in data:
                data["cards"] = data["flashcards"]
            elif "cards" not in data and "deck" in data:
                data = data["deck"]

            deck = FlashcardDeck.model_validate(data)
            return deck, None

        except json.JSONDecodeError as jde:
            return None, f"Gemini returned malformed JSON: {jde}"
        except Exception as err:
            return None, format_api_error(err, self.api_key)

    def _generate_mock_quiz(self, request: GenerationRequest) -> QuizSet:
        """Synthesize intelligent mock quiz items derived directly from the provided content or topic."""
        content = request.content.strip()
        lines = [line.strip() for line in re.split(r"(?<=[.?!])\s+|\n+", content) if len(line.strip()) > 20]
        subject = request.subject or "Study Material"
        topic = request.exam_topic or "Core Concepts"

        from models import GenerationSourceMode
        source_mode = getattr(request, "source_mode", GenerationSourceMode.NOTES_ONLY)

        stopwords = {
            "this", "that", "with", "from", "which", "other", "into", "place", "take",
            "using", "during", "their", "there", "these", "those", "where", "when",
            "form", "used", "state", "turn", "energy", "process", "could", "would",
            "should", "about", "above", "after", "again", "against", "between", "through"
        }

        # Collect key vocabulary terms across the user's specific text
        all_terms = []
        for line in lines:
            for w in re.findall(r"\b[A-Za-z]{4,}\b", line):
                if w.lower() not in stopwords and w.capitalize() not in all_terms:
                    all_terms.append(w.capitalize())

        stock_distractors = [
            "Equilibrium Dynamics", "System Feedback", "Regulatory Inhibition",
            "State Coherence", "Substrate Coupling", "Catalytic Threshold",
            "Thermodynamic Gradient", "Kinetic Synthesis", "Metabolic Turnover"
        ]

        topic_curriculum_templates = [
            {
                "question": f"In {subject}, what is the governing principle behind {topic}?",
                "choices": [
                    f"It coordinates systematic operational transitions in {topic}.",
                    "It operates strictly under perpetual non-interacting conditions.",
                    "It permanently disables feedback loops across sub-processes.",
                    "It applies exclusively to static non-interacting entities.",
                ],
                "correct_answer": f"It coordinates systematic operational transitions in {topic}.",
                "explanation": f"Core curriculum definition for {topic} in {subject}.",
                "reference_concept": f"{topic} Core Mechanism",
            },
            {
                "question": f"Which critical requirement is emphasized in academic study of {topic}?",
                "choices": [
                    "Maintaining systematic equilibrium and bounded state progression.",
                    "Disregarding conservation constraints during state transitions.",
                    "Eliminating all parallel and concurrent operational pathways.",
                    "Restricting interaction solely to synthetic homogeneous systems.",
                ],
                "correct_answer": "Maintaining systematic equilibrium and bounded state progression.",
                "explanation": f"Curriculum requirement for {topic}.",
                "reference_concept": f"{topic} Dynamics",
            },
            {
                "question": f"In {subject}, how is tension between throughput and regulatory control resolved in {topic}?",
                "choices": [
                    "Through intermediate regulatory feedback checkpoints.",
                    "By allowing uncontrolled exponential propagation without limits.",
                    "By completely halting turnover upon initial stimulus.",
                    "Through static rigid configurations that reject dynamic adaptation.",
                ],
                "correct_answer": "Through intermediate regulatory feedback checkpoints.",
                "explanation": f"Standard regulatory mechanisms in {subject}.",
                "reference_concept": "Feedback & Regulation",
            },
            {
                "question": f"What occurs when foundational constraints of {topic} are violated?",
                "choices": [
                    "System degradation, race conditions, or loss of state coherence.",
                    "Immediate instantaneous synchronization of all components.",
                    "Spontaneous formation of perpetual kinetic cycles.",
                    "Zero observable deviation from baseline steady state.",
                ],
                "correct_answer": "System degradation, race conditions, or loss of state coherence.",
                "explanation": f"System vulnerability analysis for {topic}.",
                "reference_concept": "Vulnerabilities & Edge Cases",
            },
        ]

        questions: List[QuizQuestion] = []
        for i in range(request.num_items):
            # If TOPIC_EXPANDED mode chosen, generate curriculum-level questions from Make It Yours
            if source_mode == GenerationSourceMode.TOPIC_EXPANDED:
                t_tmpl = topic_curriculum_templates[i % len(topic_curriculum_templates)]
                q_text = t_tmpl["question"]
                if request.num_items > len(topic_curriculum_templates) and i >= len(topic_curriculum_templates):
                    q_text = f"[{i + 1}] {q_text}"
                questions.append(
                    QuizQuestion(
                        id=i + 1,
                        question=q_text,
                        choices=t_tmpl["choices"],
                        correct_answer=t_tmpl["correct_answer"],
                        explanation=t_tmpl["explanation"],
                        reference_concept=t_tmpl["reference_concept"],
                    )
                )
            elif lines:
                # NOTES_ONLY or HYBRID mode: extract directly from the provided sentences
                line = lines[i % len(lines)]
                words = [w for w in re.findall(r"\b[A-Za-z]{4,}\b", line) if w.lower() not in stopwords]
                target_word = words[i % len(words)] if words else (topic.split()[0] if topic else "Principle")
                target_cap = target_word.capitalize()

                blanked = re.sub(r"\b" + re.escape(target_word) + r"\b", "_______", line, count=1, flags=re.IGNORECASE)
                question_templates = [
                    f"Fill in the blank based on your notes:\n\n\"{blanked}\"",
                    f"Regarding your notes on '{line[:65]}...', which term correctly completes the concept:\n\n\"{blanked}\"?",
                    f"According to the source text on {topic}, which term belongs in the blank:\n\n\"{blanked}\"?",
                    f"Which key concept from your notes completes this statement:\n\n\"{blanked}\"?",
                    f"From your study material, identify the missing concept:\n\n\"{blanked}\"?",
                ]
                question_text = question_templates[i % len(question_templates)]
                if request.num_items > len(question_templates) and i >= len(question_templates):
                    question_text = f"[{i + 1}] {question_text}"

                distractors = [t for t in all_terms if t.lower() != target_cap.lower()]
                for sd in stock_distractors:
                    if len(distractors) >= 3:
                        break
                    if sd.lower() != target_cap.lower() and sd not in distractors:
                        distractors.append(sd)

                choices = [target_cap] + distractors[:3]
                rot = (i * 2 + 1) % 4
                choices = choices[rot:] + choices[:rot]

                questions.append(
                    QuizQuestion(
                        id=i + 1,
                        question=question_text,
                        choices=choices,
                        correct_answer=target_cap,
                        explanation=f"Directly referenced in your study notes: \"{line}\"",
                        reference_concept=f"{topic} - {target_cap}",
                    )
                )
            else:
                # Generic fallback if no text sentences
                questions.append(
                    QuizQuestion(
                        id=i + 1,
                        question=f"Regarding {topic} in {subject}, which assertion is fundamental ({i + 1})?",
                        choices=[
                            f"It constitutes the core foundational mechanism of {topic}.",
                            "It operates exclusively in isolated, non-interacting systems.",
                            "It invalidates the conservation laws of {subject}.",
                            "It produces entirely random fluctuations without bounded limits.",
                        ],
                        correct_answer=f"It constitutes the core foundational mechanism of {topic}.",
                        explanation=f"Foundational principle for {topic} in {subject}.",
                        reference_concept=f"{topic} Fundamentals",
                    )
                )

        mode_label = "Notes Mode" if source_mode == GenerationSourceMode.NOTES_ONLY else ("Topic Mode" if source_mode == GenerationSourceMode.TOPIC_EXPANDED else "Hybrid Mode")
        return QuizSet(
            title=f"{topic} Practice Quiz [{mode_label}]",
            subject=subject,
            difficulty=request.difficulty,
            questions=questions,
        )

    def _generate_mock_flashcards(self, request: GenerationRequest) -> FlashcardDeck:
        """Synthesize intelligent mock flashcards derived directly from the provided content or topic."""
        content = request.content.strip()
        lines = [line.strip() for line in re.split(r"(?<=[.?!])\s+|\n+", content) if len(line.strip()) > 20]
        subject = request.subject or "Study Material"
        topic = request.exam_topic or "Key Principles"

        from models import GenerationSourceMode
        source_mode = getattr(request, "source_mode", GenerationSourceMode.NOTES_ONLY)

        stopwords = {
            "this", "that", "with", "from", "which", "other", "into", "place", "take",
            "using", "during", "their", "there", "these", "those", "where", "when",
            "form", "used", "state", "turn", "energy", "process", "could", "would"
        }

        topic_curriculum_flashcards = [
            (
                f"What is the foundational definition of {topic}?",
                f"In {subject}, {topic} defines the principal mechanism coordinating system components and energetic pathways.",
                f"{topic} Core Definition"
            ),
            (
                f"What is the primary governing mechanism of {topic}?",
                f"It regulates progression through bounded feedback checkpoints and systematic equilibrium.",
                f"{topic} Mechanisms"
            ),
            (
                f"How is {topic} practically applied in {subject}?",
                f"It is applied to optimize yield, maintain state coherence, and avoid critical boundary degradation.",
                f"{topic} Application"
            ),
            (
                f"What critical failure occurs when constraints of {topic} are violated?",
                f"System instability, loss of regulatory feedback, and functional degradation.",
                f"{topic} Vulnerabilities"
            ),
            (
                f"What prerequisite condition is required for {topic} activation?",
                f"Appropriate potential difference, substrate concentration gradients, and coupling dynamics.",
                f"{topic} Prerequisites"
            ),
        ]

        cards: List[Flashcard] = []
        for i in range(request.num_items):
            if source_mode == GenerationSourceMode.TOPIC_EXPANDED:
                t_front, t_back, t_concept = topic_curriculum_flashcards[i % len(topic_curriculum_flashcards)]
                if request.num_items > len(topic_curriculum_flashcards) and i >= len(topic_curriculum_flashcards):
                    t_front = f"[{i + 1}] {t_front}"
                cards.append(
                    Flashcard(
                        id=i + 1,
                        front=t_front,
                        back=t_back,
                        explanation=f"Core curriculum concept for {topic} in {subject}.",
                        difficulty=request.difficulty,
                        key_concept=t_concept,
                    )
                )
            elif lines:
                line = lines[i % len(lines)]
                words = [w for w in re.findall(r"\b[A-Za-z]{4,}\b", line) if w.lower() not in stopwords]
                kw = words[0].capitalize() if words else (topic.split()[0] if topic else "Key Concept")

                prompt_templates = [
                    f"What does your study material state regarding '{kw}'?",
                    f"How is the principle of '{kw}' explained in your notes?",
                    f"What is the key mechanism or role of '{kw}' according to the text?",
                    f"Explain the significance of '{kw}' in the context: \"{line[:50]}...\"?",
                    f"What key fact about '{kw}' is emphasized in the source material?",
                ]
                front = prompt_templates[i % len(prompt_templates)]
                if request.num_items > len(prompt_templates) and i >= len(prompt_templates):
                    front = f"[{i + 1}] {front}"

                back = line
                concept = f"{topic} • {kw}"

                cards.append(
                    Flashcard(
                        id=i + 1,
                        front=front,
                        back=back,
                        explanation=f"Extracted directly from your study notes on {topic} ({request.difficulty.value} difficulty).",
                        difficulty=request.difficulty,
                        key_concept=concept,
                    )
                )
            else:
                front = f"What is the foundational principle governing {topic} ({i + 1})?"
                back = f"It provides the essential operational framework for {topic} in {subject}."
                concept = f"{topic} Foundations"

                cards.append(
                    Flashcard(
                        id=i + 1,
                        front=front,
                        back=back,
                        explanation=f"Foundational principle for {topic} in {subject}.",
                        difficulty=request.difficulty,
                        key_concept=concept,
                    )
                )

        mode_label = "Notes Deck" if source_mode == GenerationSourceMode.NOTES_ONLY else ("Curriculum Deck" if source_mode == GenerationSourceMode.TOPIC_EXPANDED else "Hybrid Deck")
        return FlashcardDeck(
            title=f"{topic} Study Deck [{mode_label}]",
            subject=subject,
            cards=cards,
        )
