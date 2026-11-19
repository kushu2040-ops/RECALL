"""Prompts and educational engineering templates for study material generation.
"""

from typing import Dict
from models import GenerationRequest

QUIZ_SYSTEM_PROMPT = """You are an expert instructional designer and academic examiner.
Your task is to create high-yield, pedagogically rigorous Multiple-Choice Questions (MCQs) from the provided study material.

Guidelines for questions:
1. Every question must have EXACTLY FOUR distinct, realistic answer choices.
2. Exactly ONE choice must be demonstrably correct.
3. Distractors (incorrect choices) must be plausible misconceptions, not obviously absurd or trivial.
4. Avoid "All of the above" or "None of the above".
5. The correct_answer field must be the EXACT text of the correct choice from the choices list.
6. Provide a thorough, crystal-clear explanation outlining WHY the correct answer is right and why common distractors are mistaken.
7. Tag each question with a specific 'reference_concept' (e.g., 'Krebs Cycle - ATP Yield').
8. Calibrate the questions strictly to the requested difficulty level:
   - Easy: Direct recall of definitions, terminology, and foundational facts.
   - Medium: Application of principles, cause-and-effect relationships, and comparative analysis.
   - Hard: Multi-step deductive reasoning, edge cases, synthesis of multiple concepts, or problem-solving scenarios.
9. Return valid JSON conforming strictly to the requested schema.
"""

FLASHCARD_SYSTEM_PROMPT = """You are an expert learning scientist specializing in spaced repetition and active recall.
Your task is to transform the provided study material into high-impact Q&A flashcards for maximum memory retention.

Guidelines for flashcards:
1. 'front': A focused, unambiguous question, scenario, or concept prompt. Avoid vague or overly broad prompts.
2. 'back': A concise, authoritative answer that directly satisfies the front prompt. Keep it punchy and memorable.
3. 'explanation': Contextual elaboration, mnemonic cues, or deeper rationale to reinforce understanding when reviewed.
4. 'key_concept': The core topic, law, or terminology tag (e.g., 'Amdahl's Law', 'Calvin Cycle').
5. 'difficulty': Label as 'Easy', 'Medium', or 'Hard' based on cognitive complexity.
6. Calibrate to the requested difficulty:
   - Easy: Definitions, vocabulary, foundational formulas.
   - Medium: Mechanism explanations, comparisons, structural relationships.
   - Hard: Troubleshooting, mathematical derivation, edge conditions, deep conceptual distinctions.
7. Return valid JSON conforming strictly to the requested schema.
"""


def build_quiz_prompt(req: GenerationRequest, text_content: str) -> str:
    """Construct the user prompt for quiz generation based on selected source mode."""
    subject_str = f"Subject: {req.subject}\n" if req.subject else ""
    topic_str = f"Exam Topic: {req.exam_topic}\n" if req.exam_topic else ""
    goals_str = f"Learning Goals: {req.learning_goals}\n" if req.learning_goals else ""
    focus_str = f"Specific Focus / Instructions: {req.focus_instructions}\n" if req.focus_instructions else ""

    from models import GenerationSourceMode
    source_mode = getattr(req, "source_mode", GenerationSourceMode.NOTES_ONLY)

    if source_mode == GenerationSourceMode.NOTES_ONLY:
        mode_instruction = (
            "STRICT SOURCE GROUNDING MODE: Generate questions, choices, and explanations strictly and exclusively "
            "from the facts, definitions, and statements explicitly present in the provided Source Study Material below. "
            "Do NOT introduce external topics or unmentioned facts outside the provided notes."
        )
    elif source_mode == GenerationSourceMode.TOPIC_EXPANDED:
        mode_instruction = (
            f"MAKE IT YOURS (TOPIC-EXPANDED MODE): Generate a comprehensive curriculum quiz on the Subject '{req.subject}' "
            f"and Topic '{req.exam_topic}', using the provided text as starting context. Draw upon standard academic curriculum "
            "knowledge to test core theorems, standard problem types, mechanisms, and foundational principles for this topic."
        )
    else:  # HYBRID
        mode_instruction = (
            "HYBRID MODE: Formulate questions that directly test the provided notes while connecting them to broader "
            "conceptual applications, comparative analyses, and curriculum principles in this subject."
        )

    return f"""Please generate a {req.difficulty.value}-level Multiple-Choice Quiz with exactly {req.num_items} questions.

{mode_instruction}

{subject_str}{topic_str}{goals_str}{focus_str}Difficulty: {req.difficulty.value}
Number of Questions: {req.num_items}

Source Study Material:
\"\"\"
{text_content}
\"\"\"

Requirements:
- Ensure each question has exactly 4 distinct choices.
- 'correct_answer' must exactly match one of the 4 choices.
- Include a helpful explanation and reference concept for each item.
- Generate questions that test deep comprehension, not just surface-level keyword matching.
"""


def build_flashcard_prompt(req: GenerationRequest, text_content: str) -> str:
    """Construct the user prompt for flashcard generation based on selected source mode."""
    subject_str = f"Subject: {req.subject}\n" if req.subject else ""
    topic_str = f"Exam Topic: {req.exam_topic}\n" if req.exam_topic else ""
    goals_str = f"Learning Goals: {req.learning_goals}\n" if req.learning_goals else ""
    focus_str = f"Specific Focus / Instructions: {req.focus_instructions}\n" if req.focus_instructions else ""

    from models import GenerationSourceMode
    source_mode = getattr(req, "source_mode", GenerationSourceMode.NOTES_ONLY)

    if source_mode == GenerationSourceMode.NOTES_ONLY:
        mode_instruction = (
            "STRICT SOURCE GROUNDING MODE: Every flashcard front prompt and back answer must be extracted directly "
            "and exclusively from the provided source notes. Do NOT include concepts not mentioned in the text."
        )
    elif source_mode == GenerationSourceMode.TOPIC_EXPANDED:
        mode_instruction = (
            f"MAKE IT YOURS (TOPIC-EXPANDED MODE): Generate high-yield study flashcards for {req.subject} - {req.exam_topic}, "
            "covering essential terms, laws, formulas, and key definitions across the broader academic curriculum for this topic."
        )
    else:  # HYBRID
        mode_instruction = (
            "HYBRID MODE: Anchor flashcards in the provided source material while enriching them with core conceptual definitions "
            "and active recall mnemonics."
        )

    return f"""Please generate a set of exactly {req.num_items} active-recall Flashcards at {req.difficulty.value} difficulty.

{mode_instruction}

{subject_str}{topic_str}{goals_str}{focus_str}Difficulty: {req.difficulty.value}
Number of Cards: {req.num_items}

Source Study Material:
\"\"\"
{text_content}
\"\"\"

Requirements:
- Front: Focused, active recall question or prompt.
- Back: Clear, punchy answer.
- Explanation: In-depth rationale or mnemonic.
- Key Concept: Precise terminology or topic tag.
"""


SAMPLE_STUDY_MATERIALS: Dict[str, Dict[str, str]] = {
    "Biology: Cellular Respiration": {
        "subject": "Cellular Biology",
        "exam_topic": "Cellular Respiration & Bioenergetics",
        "learning_goals": "Understand glycolysis, Krebs cycle, and electron transport chain ATP synthesis.",
        "content": """Cellular respiration is a metabolic pathway that breaks down glucose and produces ATP. The overall equation is C6H12O6 + 6O2 -> 6CO2 + 6H2O + ATP.

The process consists of three main stages:
1. Glycolysis: Takes place in the cytoplasm of the cell. One glucose molecule (6 carbons) is broken down into two pyruvate molecules (3 carbons each). This stage is anaerobic (does not require oxygen) and yields a net gain of 2 ATP and 2 NADH. Substrate-level phosphorylation produces the ATP.

2. The Citric Acid Cycle (Krebs Cycle): Occurs in the mitochondrial matrix. Before entering the cycle, pyruvate is converted into Acetyl-CoA during the transition reaction, releasing CO2 and producing 1 NADH per pyruvate. In the cycle, Acetyl-CoA combines with oxaloacetate (4 carbons) to form citrate (6 carbons). For each turn of the cycle (two turns per glucose), 1 ATP (or GTP), 3 NADH, and 1 FADH2 are generated, with CO2 released as a waste byproduct.

3. Oxidative Phosphorylation & Electron Transport Chain (ETC): Situated across the inner mitochondrial membrane (cristae). High-energy electrons carried by NADH and FADH2 are transferred through a series of multi-protein complexes (Complexes I-IV). As electrons move down the redox gradient, protons (H+) are pumped from the mitochondrial matrix into the intermembrane space, creating a steep electrochemical proton-motive force. Oxygen serves as the terminal electron acceptor, binding protons to form water. Protons diffuse back into the matrix through ATP Synthase, driving the rotational phosphorylation of ADP into ATP (chemiosmosis). This stage produces approximately 26 to 28 ATP molecules, making it the most productive phase of aerobic respiration.""",
    },
    "Computer Science: Operating Systems & Concurrency": {
        "subject": "Computer Science",
        "exam_topic": "Operating Systems & Concurrency Control",
        "learning_goals": "Master critical sections, race conditions, semaphores, and deadlocks.",
        "content": """Concurrent processes often share resources such as memory, files, or hardware devices. When multiple processes access and manipulate the same data concurrently and the outcome depends on the particular order of execution, a race condition occurs.

To prevent race conditions, the segment of code accessing shared resources is known as the Critical Section. A valid solution to the Critical Section problem must satisfy three fundamental criteria:
1. Mutual Exclusion: If process Pi is executing in its critical section, no other processes can be executing in their critical sections.
2. Progress: If no process is executing in its critical section and some processes wish to enter, only those processes not executing in their remainder sections can participate in deciding who enters next, and this selection cannot be postponed indefinitely.
3. Bounded Waiting: There must be a bound on the number of times other processes are allowed to enter their critical sections after a process has made a request to enter and before that request is granted.

Synchronization mechanisms include:
- Mutex Locks: A boolean lock used to protect critical regions. A thread must acquire the lock before entering and release it upon exit.
- Semaphores: An integer variable accessed only through atomic operations: wait() (also known as P) and signal() (also known as V). A counting semaphore can control access to a resource pool with finite instances, while a binary semaphore behaves similarly to a mutex lock.
- Deadlock: A condition where every process in a set is waiting for an event that only another process in the set can cause. The four necessary Coffman conditions for deadlock are Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait.""",
    },
    "Economics: Macroeconomics & Monetary Policy": {
        "subject": "Macroeconomics",
        "exam_topic": "Monetary Policy & Inflation",
        "learning_goals": "Analyze how central banks manage inflation and unemployment through interest rates.",
        "content": """Monetary policy is the process by which a nation's central bank manages money supply and interest rates to achieve macroeconomic objectives: price stability, maximum employment, and moderate long-term interest rates.

Central banks possess three primary tools:
1. Open Market Operations (OMO): The buying and selling of government securities in the open market. Buying bonds injects reserves into the banking system, lowering the short-term policy interest rate (expansionary). Selling bonds drains bank reserves, pushing interest rates higher (contractionary).
2. The Reserve Requirement: The fraction of deposits banks must hold in reserve rather than lend out. Lowering the reserve ratio multiplies the money supply via the money multiplier effect (1 / Reserve Ratio).
3. The Discount Rate / Policy Rate: The interest rate charged by central banks on short-term loans to commercial banks. A higher discount rate discourages borrowing and slows credit creation.

Inflation Dynamics & Phillips Curve:
- Demand-Pull Inflation occurs when aggregate demand exceeds aggregate supply ('too much money chasing too few goods').
- Cost-Push Inflation results from supply shocks (such as sudden energy price spikes) that raise input costs.
- The Phillips Curve illustrates the short-run inverse relationship between inflation and unemployment. However, the Long-Run Phillips Curve (LRPC) is vertical at the Natural Rate of Unemployment (NAIRU), indicating that monetary stimulus cannot permanently suppress unemployment below its natural equilibrium without triggering accelerating inflation.""",
    },
}
