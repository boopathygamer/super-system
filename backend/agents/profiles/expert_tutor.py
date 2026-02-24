"""
Expert Tutor Engine — Research-Backed Adaptive Teaching System
──────────────────────────────────────────────────────────────
An elite tutor that automatically detects when LLM knowledge is insufficient,
triggers deep internet research (web + academic + social), and delivers
expert-level coaching using 5 proven teaching techniques.

Teaching Techniques:
  🧪 Feynman       — Explain complex ideas simply with analogies
  🏗️ Scaffolding   — Build knowledge layer by layer
  🦉 Socratic      — Guide via probing questions
  🌉 Analogy Bridge — Connect unknowns to student's existing knowledge
  🧩 Chunking      — Break massive topics into micro-lessons

Key Innovation:
  If the LLM response shows uncertainty (hedging language, vagueness),
  the tutor AUTOMATICALLY triggers deep web + social research and weaves
  real-world expert knowledge into the coaching session.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Enums & Data Models
# ──────────────────────────────────────────────

class TeachingTechnique(Enum):
    FEYNMAN = "feynman"
    SCAFFOLDING = "scaffolding"
    SOCRATIC = "socratic"
    ANALOGY_BRIDGE = "analogy_bridge"
    CHUNKING = "chunking"


class StudentLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class DiagnosticResult:
    """Result of diagnosing the student's current level."""
    level: StudentLevel = StudentLevel.BEGINNER
    confidence: float = 0.0
    knowledge_gaps: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)


@dataclass
class ResearchIntel:
    """Teaching material compiled from deep research."""
    topic: str = ""
    eli5_explanations: List[str] = field(default_factory=list)
    expert_insights: List[str] = field(default_factory=list)
    real_world_examples: List[str] = field(default_factory=list)
    practice_problems: List[str] = field(default_factory=list)
    academic_findings: List[str] = field(default_factory=list)
    social_wisdom: List[str] = field(default_factory=list)
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class TutoringSession:
    """State for an active tutoring session."""
    session_id: str = ""
    topic: str = ""
    student_level: StudentLevel = StudentLevel.BEGINNER
    current_technique: TeachingTechnique = TeachingTechnique.SCAFFOLDING
    research_intel: Optional[ResearchIntel] = None
    research_triggered: bool = False
    history: List[Dict[str, str]] = field(default_factory=list)
    diagnostic: Optional[DiagnosticResult] = None
    lesson_plan: List[str] = field(default_factory=list)
    current_lesson_step: int = 0
    confidence_scores: List[float] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)


# ──────────────────────────────────────────────
# Confidence / Uncertainty Detection
# ──────────────────────────────────────────────

# Patterns that indicate the LLM is uncertain or lacks knowledge
_UNCERTAINTY_PATTERNS = [
    r"\bi(?:'m| am) not (?:entirely |completely |100%? )?sure\b",
    r"\bi(?:'m| am) not (?:entirely )?certain\b",
    r"\bi think\b(?! you)",
    r"\bit(?:'s| is) possible that\b",
    r"\bi believe\b",
    r"\bperhaps\b",
    r"\bprobably\b",
    r"\bmight be\b",
    r"\bcould be\b",
    r"\bI don'?t have (?:enough |sufficient )?(?:information|knowledge|data)\b",
    r"\bI(?:'m| am) not (?:an )?expert\b",
    r"\bmy (?:knowledge|training|data) (?:is |was )?(?:limited|cut ?off)\b",
    r"\bI (?:can'?t|cannot) (?:verify|confirm|guarantee)\b",
    r"\bgenerally speaking\b",
    r"\bin general\b",
    r"\bas far as I know\b",
    r"\bto the best of my knowledge\b",
    r"\byou (?:should|might want to) (?:consult|check|verify|look up)\b",
]

_VAGUE_PATTERNS = [
    r"\bsome (?:people|experts|sources) (?:say|think|believe)\b",
    r"\bit depends\b",
    r"\bthere are (?:many|various|different) (?:ways|approaches|methods)\b",
    r"\bthis is a complex topic\b",
    r"\bthe (?:answer|topic) is (?:nuanced|complex|multifaceted)\b",
]

_compiled_uncertainty = [re.compile(p, re.IGNORECASE) for p in _UNCERTAINTY_PATTERNS]
_compiled_vague = [re.compile(p, re.IGNORECASE) for p in _VAGUE_PATTERNS]


def detect_uncertainty(response: str) -> Tuple[float, List[str]]:
    """
    Analyze an LLM response for uncertainty signals.
    
    Returns:
        (uncertainty_score, matched_signals) where score is 0.0-1.0.
        Score > 0.4 means research should be triggered.
    """
    matched = []
    
    # Check uncertainty patterns (high weight)
    for pattern in _compiled_uncertainty:
        if pattern.search(response):
            matched.append(f"uncertainty: {pattern.pattern[:40]}")
    
    # Check vague patterns (medium weight)
    for pattern in _compiled_vague:
        if pattern.search(response):
            matched.append(f"vague: {pattern.pattern[:40]}")
    
    # Check response length (very short responses often indicate lack of knowledge)
    word_count = len(response.split())
    if word_count < 30:
        matched.append("very_short_response")
    
    # Calculate score
    uncertainty_weight = sum(1 for m in matched if m.startswith("uncertainty"))
    vague_weight = sum(0.5 for m in matched if m.startswith("vague"))
    short_weight = 0.3 if "very_short_response" in matched else 0.0
    
    raw_score = uncertainty_weight * 0.15 + vague_weight * 0.15 + short_weight
    score = min(1.0, raw_score)
    
    return score, matched


# ──────────────────────────────────────────────
# Teaching Technique Prompts
# ──────────────────────────────────────────────

_TECHNIQUE_PROMPTS = {
    TeachingTechnique.FEYNMAN: """\
🧪 FEYNMAN TECHNIQUE MODE:
You are teaching using the Feynman Technique — the gold standard for deep understanding.
Rules:
1. Explain the concept as if teaching a smart 12-year-old
2. Use REAL-WORLD ANALOGIES for every abstract concept
3. When you hit something complex, break it into simpler sub-concepts
4. Use "Imagine you're..." or "Think of it like..." frequently
5. After explaining, ask: "Can you explain this back to me in your own words?"
6. If the student's explanation has gaps, gently point them out
7. NO jargon without immediately defining it with an analogy""",

    TeachingTechnique.SCAFFOLDING: """\
🏗️ SCAFFOLDING MODE:
You are building knowledge layer by layer, like constructing a building.
Rules:
1. Start with the FOUNDATION — what does the student already know?
2. Each new concept builds on the PREVIOUS one — never skip a level
3. Provide a "knowledge checkpoint" after every 2-3 concepts
4. If the student fails a checkpoint, go back ONE layer and reinforce
5. Use this pattern: Foundation → Core Concept → Application → Integration
6. Give specific examples at each layer
7. End each message with: "Before we go deeper, let me check: [checkpoint question]" """,

    TeachingTechnique.SOCRATIC: """\
🦉 SOCRATIC METHOD MODE:
You are a strict Socratic professor who NEVER gives direct answers.
Rules:
1. NEVER tell them the answer directly
2. Ask a leading question that forces them to discover the answer
3. When they answer correctly, praise them: "Excellent! Now consider..."
4. When they answer wrong, don't say "wrong" — instead ask a clarifying question
5. Break the problem into smaller sub-questions they can reason through
6. Your goal: make the student feel like THEY discovered the knowledge
7. Only reveal information after they've attempted to reason through it""",

    TeachingTechnique.ANALOGY_BRIDGE: """\
🌉 ANALOGY BRIDGE MODE:
You are connecting unknown concepts to things the student already understands.
Rules:
1. ALWAYS start by asking what the student is already familiar with
2. Build BRIDGES from their existing knowledge to new concepts
3. Every new concept gets at least TWO analogies from different domains
4. Use "It's like...", "Think of it as...", "Remember how X works? This is similar because..."
5. After the bridge, test it: "Where does the analogy break down?"
6. This forces deeper understanding than surface-level comparison
7. Use visual/spatial analogies when possible (maps, buildings, rivers)""",

    TeachingTechnique.CHUNKING: """\
🧩 CHUNKING MODE:
You are breaking a massive topic into tiny, digestible micro-lessons.
Rules:
1. Split the entire topic into 5-7 numbered CHUNKS
2. Present ONLY ONE chunk at a time — don't overwhelm
3. Each chunk should take ~2 minutes to understand
4. Format: [Chunk Title] → [Core Idea in 1 sentence] → [Example] → [Practice]
5. After each chunk: "Ready for the next piece of the puzzle?"
6. At the end, show how ALL chunks connect into the complete picture
7. Use progress indicators: "Chunk 3/6: [Title]" """,
}


# ──────────────────────────────────────────────
# Difficulty Adaptation Prompts
# ──────────────────────────────────────────────

_LEVEL_PROMPTS = {
    StudentLevel.BEGINNER: (
        "The student is a COMPLETE BEGINNER. "
        "Use everyday language. No jargon. Lots of analogies and examples. "
        "Assume ZERO prior knowledge. Be warm and encouraging."
    ),
    StudentLevel.INTERMEDIATE: (
        "The student has BASIC understanding and some experience. "
        "Use standard terminology but define advanced terms. "
        "Focus on WHY things work, not just HOW. Challenge them a little."
    ),
    StudentLevel.ADVANCED: (
        "The student is ADVANCED and understands core concepts well. "
        "Use technical language freely. Focus on edge cases, trade-offs, "
        "and real-world applications. Push them toward expert-level thinking."
    ),
    StudentLevel.EXPERT: (
        "The student is near-EXPERT level. "
        "Engage in peer-level discussion. Focus on cutting-edge research, "
        "open problems, and frontier knowledge. Cite specific papers/sources. "
        "Challenge their assumptions with contrarian viewpoints."
    ),
}


# ──────────────────────────────────────────────
# Expert Tutor Engine
# ──────────────────────────────────────────────

class ExpertTutorEngine:
    """
    Research-backed adaptive teaching engine.
    
    When the LLM doesn't know enough about a topic, it automatically
    triggers deep web + social media research and converts the results
    into expert-level coaching material that's easy to understand.
    """

    def __init__(self, generate_fn: Callable, agent_controller=None):
        """
        Args:
            generate_fn: The LLM generation function (prompt, system_prompt, temperature)
            agent_controller: Optional AgentController for deep research access
        """
        self.generate_fn = generate_fn
        self.agent = agent_controller
        self._sessions: Dict[str, TutoringSession] = {}
        self._researcher = None
        
        # Lazy-init researcher
        if agent_controller:
            try:
                from agents.profiles.deep_researcher import DeepWebResearcher
                self._researcher = DeepWebResearcher(agent_controller)
            except ImportError:
                logger.warning("DeepWebResearcher not available")

        logger.info("🎓 ExpertTutorEngine initialized")

    # ──────────────────────────────────────
    # Session Management
    # ──────────────────────────────────────

    def start_session(self, topic: str, session_id: str = None) -> TutoringSession:
        """Start a new tutoring session on a topic."""
        import uuid
        session_id = session_id or uuid.uuid4().hex[:10]
        
        session = TutoringSession(
            session_id=session_id,
            topic=topic,
        )
        
        # Step 1: Generate initial LLM response to assess confidence
        probe_prompt = (
            f"Provide a comprehensive, expert-level explanation of: {topic}. "
            f"Cover the key concepts, principles, and latest developments. "
            f"Be specific and cite facts."
        )
        
        probe_result = self._call_llm(probe_prompt, "You are a knowledge assessment probe.")
        uncertainty_score, signals = detect_uncertainty(probe_result)
        
        logger.info(
            f"🎓 Topic '{topic}' — LLM uncertainty: {uncertainty_score:.2f} "
            f"(signals: {len(signals)})"
        )
        
        # Step 2: If uncertain, trigger deep research
        if uncertainty_score > 0.3:
            logger.info(f"🔬 Triggering deep research for topic: {topic}")
            session.research_intel = self._research_for_teaching(topic)
            session.research_triggered = True
        
        # Step 3: Build the lesson plan
        session.lesson_plan = self._build_lesson_plan(topic, session.research_intel)
        
        # Step 4: Choose initial teaching technique
        session.current_technique = self._select_technique(topic)
        
        self._sessions[session_id] = session
        
        logger.info(
            f"🎓 Session {session_id}: topic='{topic}', "
            f"technique={session.current_technique.value}, "
            f"research={'YES' if session.research_triggered else 'NO'}, "
            f"lesson_steps={len(session.lesson_plan)}"
        )
        
        return session

    def get_session(self, session_id: str) -> Optional[TutoringSession]:
        return self._sessions.get(session_id)

    # ──────────────────────────────────────
    # Teaching Flow
    # ──────────────────────────────────────

    def begin_teaching(self, session: TutoringSession) -> str:
        """Generate the opening diagnostic + first teaching message."""
        system_prompt = self._build_system_prompt(session)
        
        # Build opening that includes a diagnostic question
        opening_prompt = (
            f"You are starting a tutoring session on: {session.topic}\n\n"
            f"LESSON PLAN:\n"
        )
        for i, step in enumerate(session.lesson_plan):
            opening_prompt += f"  {i+1}. {step}\n"
        
        opening_prompt += (
            "\nFirst, give a brief exciting introduction about WHY this topic matters "
            "(2-3 sentences). Then ask ONE diagnostic question to assess the student's "
            "current level. The question should have 3 difficulty levels embedded "
            "(easy/medium/hard) so you can gauge where they are."
        )
        
        # Include research intelligence if available
        if session.research_intel:
            opening_prompt += self._format_research_context(session.research_intel)
        
        response = self._call_llm(opening_prompt, system_prompt, temperature=0.7)
        
        session.history.append({"role": "assistant", "content": response})
        return response

    def respond_to_student(
        self, session: TutoringSession, student_message: str
    ) -> str:
        """
        Process student's response and generate next coaching step.
        
        This is the core teaching loop:
          1. Analyze student response for understanding level
          2. Update difficulty
          3. Check if we need more research
          4. Generate next coaching step using active technique
        """
        session.history.append({"role": "user", "content": student_message})
        
        # Diagnose student level from their response
        diagnosis = self._diagnose_student(session, student_message)
        session.diagnostic = diagnosis
        session.student_level = diagnosis.level
        
        # Check if student asked about something we might not know enough about
        follow_up_uncertainty = self._check_followup_knowledge(
            session, student_message
        )
        
        if follow_up_uncertainty > 0.4 and not session.research_triggered:
            logger.info(f"🔬 Student question triggered deep research: {student_message[:60]}")
            session.research_intel = self._research_for_teaching(
                f"{session.topic}: {student_message}"
            )
            session.research_triggered = True
        
        # Build next coaching response
        system_prompt = self._build_system_prompt(session)
        
        # Build conversational context (last 10 exchanges max)
        context_messages = session.history[-20:]
        chat_context = ""
        for msg in context_messages:
            role = "STUDENT" if msg["role"] == "user" else "COACH"
            chat_context += f"{role}: {msg['content']}\n\n"
        
        # Progress tracking
        progress = ""
        if session.lesson_plan:
            step_idx = min(session.current_lesson_step, len(session.lesson_plan) - 1)
            progress = (
                f"\n\nCURRENT PROGRESS: Step {step_idx + 1}/{len(session.lesson_plan)}: "
                f"{session.lesson_plan[step_idx]}"
            )
        
        teaching_prompt = (
            f"Dialogue so far:\n{chat_context}\n"
            f"STUDENT DIAGNOSIS: Level={diagnosis.level.value}, "
            f"Gaps={diagnosis.knowledge_gaps}, Strengths={diagnosis.strengths}\n"
            f"{progress}\n\n"
            f"The student just said: \"{student_message}\"\n\n"
            f"Generate your next coaching response. Remember to use the "
            f"{session.current_technique.value} technique. "
            f"If the student shows they understand the current concept, "
            f"advance to the next lesson step."
        )
        
        # Include research if available
        if session.research_intel:
            teaching_prompt += self._format_research_context(session.research_intel)
        
        response = self._call_llm(teaching_prompt, system_prompt, temperature=0.7)
        
        # Track confidence and advance lesson
        session.history.append({"role": "assistant", "content": response})
        
        # Auto-advance lesson if student shows understanding
        if diagnosis.confidence > 0.7 and session.lesson_plan:
            if session.current_lesson_step < len(session.lesson_plan) - 1:
                session.current_lesson_step += 1
        
        return response

    # ──────────────────────────────────────
    # Interactive CLI Session
    # ──────────────────────────────────────

    def start_interactive(self, topic: str):
        """Start an interactive tutoring session in the console."""
        print(f"\n{'='*60}")
        print("  🎓 EXPERT TUTOR ENGINE — Interactive Session")
        print(f"  Topic: {topic}")
        print(f"{'='*60}")
        
        session = self.start_session(topic)
        
        if session.research_triggered:
            print("\n🔬 Deep Research triggered — I'll be using expert sources from")
            print("   the web, academic papers, and social media to teach you.\n")
        
        print(f"📚 Teaching technique: {session.current_technique.value}")
        print(f"📋 Lesson plan: {len(session.lesson_plan)} steps")
        print("\nType 'exit', 'quit', or 'done' to end the session.")
        print("Type 'switch <technique>' to change teaching style.")
        print("  Techniques: feynman, scaffolding, socratic, analogy_bridge, chunking")
        print(f"{'─'*60}\n")
        
        # Opening message
        opening = self.begin_teaching(session)
        print(f"🎓 Coach: {opening}\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ('exit', 'quit', 'done'):
                    self._end_session_summary(session)
                    break
                
                # Handle technique switching
                if user_input.lower().startswith('switch '):
                    technique_name = user_input[7:].strip().lower()
                    try:
                        session.current_technique = TeachingTechnique(technique_name)
                        print(f"\n📚 Switched to {technique_name} technique!\n")
                        continue
                    except ValueError:
                        print(f"\n❌ Unknown technique: {technique_name}")
                        print("   Available: feynman, scaffolding, socratic, analogy_bridge, chunking\n")
                        continue
                
                # Generate coaching response
                response = self.respond_to_student(session, user_input)
                
                # Show progress
                step = min(session.current_lesson_step + 1, len(session.lesson_plan))
                total = len(session.lesson_plan) or 1
                level = session.student_level.value
                print(f"\n[📊 Level: {level} | Step: {step}/{total}]")
                print(f"🎓 Coach: {response}\n")
                
            except KeyboardInterrupt:
                self._end_session_summary(session)
                break
            except Exception as e:
                logger.error(f"Tutor error: {e}", exc_info=True)
                print("\n⚠️ Teaching error occurred. Let me try again.\n")

    def _end_session_summary(self, session: TutoringSession):
        """Print a session summary when the student exits."""
        elapsed = time.time() - session.started_at
        minutes = int(elapsed / 60)
        exchanges = len([m for m in session.history if m["role"] == "user"])
        
        print(f"\n{'='*60}")
        print("  📊 SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"  Topic: {session.topic}")
        print(f"  Duration: {minutes} minutes")
        print(f"  Exchanges: {exchanges}")
        print(f"  Final Level: {session.student_level.value}")
        print(f"  Research Used: {'Yes' if session.research_triggered else 'No'}")
        print(f"  Technique: {session.current_technique.value}")
        if session.lesson_plan:
            step = min(session.current_lesson_step + 1, len(session.lesson_plan))
            print(f"  Progress: {step}/{len(session.lesson_plan)} steps completed")
        print("\n  🎓 Great work today! Keep learning! 🚀")
        print(f"{'='*60}\n")

    # ──────────────────────────────────────
    # Deep Research for Teaching
    # ──────────────────────────────────────

    def _research_for_teaching(self, topic: str) -> ResearchIntel:
        """
        Trigger deep research and convert raw intelligence into
        structured teaching material.
        """
        intel = ResearchIntel(topic=topic)
        
        try:
            from agents.tools.web_search import advanced_web_search
            
            # 1. Surface web — practical explanations
            surface = advanced_web_search(
                f"explain {topic} simply with examples",
                network="surface", max_results=5, deep_scrape=True,
            )
            for item in surface.get("results", []):
                content = item.get("full_content") or item.get("snippet", "")
                if content and len(content) > 50:
                    intel.real_world_examples.append(content[:800])
                    intel.sources.append({
                        "title": item.get("title", ""),
                        "url": item.get("href", ""),
                        "type": "surface",
                    })
            
            # 2. Social media — Reddit ELI5 + expert discussions
            social = advanced_web_search(
                f"{topic} ELI5 explained simple",
                network="social", max_results=5,
            )
            for item in social.get("results", []):
                snippet = item.get("snippet", "")
                if snippet and len(snippet) > 30:
                    intel.social_wisdom.append(snippet[:600])
                    intel.eli5_explanations.append(snippet[:400])
                    intel.sources.append({
                        "title": item.get("title", ""),
                        "url": item.get("href", ""),
                        "type": "social",
                    })
            
            # 3. Academic — deep/ArXiv for expert-level insights
            academic = advanced_web_search(
                topic, network="deep", max_results=5,
            )
            for item in academic.get("results", []):
                snippet = item.get("snippet", "")
                if snippet:
                    intel.academic_findings.append(snippet[:800])
                    intel.expert_insights.append(
                        f"[{item.get('title', 'Research')}]: {snippet[:300]}"
                    )
                    intel.sources.append({
                        "title": item.get("title", ""),
                        "url": item.get("href", ""),
                        "type": "academic",
                    })
            
            logger.info(
                f"🔬 Research compiled: {len(intel.real_world_examples)} examples, "
                f"{len(intel.eli5_explanations)} ELI5s, "
                f"{len(intel.academic_findings)} academic, "
                f"{len(intel.social_wisdom)} social"
            )
            
        except Exception as e:
            logger.error(f"Research failed: {e}", exc_info=True)
        
        # Generate practice problems using LLM
        if intel.real_world_examples or intel.academic_findings:
            try:
                problem_prompt = (
                    f"Based on this topic: {topic}\n"
                    f"Generate 3 practice problems/exercises for a student, "
                    f"ordered from easy to hard. Format as numbered list. "
                    f"Each problem should have a brief hint."
                )
                problems = self._call_llm(problem_prompt, "You are an expert educator.")
                intel.practice_problems = [problems]
            except Exception:
                pass
        
        return intel

    def _format_research_context(self, intel: ResearchIntel) -> str:
        """Format research intelligence for injection into the teaching prompt."""
        parts = ["\n\n--- DEEP RESEARCH INTELLIGENCE (use to enrich your teaching) ---"]
        
        if intel.eli5_explanations:
            parts.append("\n📱 COMMUNITY EXPLANATIONS (Reddit/Social):")
            for i, exp in enumerate(intel.eli5_explanations[:3]):
                parts.append(f"  {i+1}. {exp[:300]}")
        
        if intel.expert_insights:
            parts.append("\n🔬 ACADEMIC/EXPERT INSIGHTS:")
            for i, insight in enumerate(intel.expert_insights[:3]):
                parts.append(f"  {i+1}. {insight[:300]}")
        
        if intel.real_world_examples:
            parts.append("\n🌍 REAL-WORLD EXAMPLES:")
            for i, ex in enumerate(intel.real_world_examples[:2]):
                parts.append(f"  {i+1}. {ex[:300]}")
        
        if intel.practice_problems:
            parts.append("\n✏️ PRACTICE PROBLEMS (offer these to the student):")
            for prob in intel.practice_problems[:1]:
                parts.append(f"  {prob[:500]}")
        
        parts.append("\n--- END RESEARCH INTELLIGENCE ---")
        parts.append(
            "\nIMPORTANT: Weave this intelligence naturally into your teaching. "
            "Don't dump it all at once. Use it to give specific, real examples "
            "and cite expert sources when it makes the explanation richer."
        )
        
        return "\n".join(parts)

    # ──────────────────────────────────────
    # Student Diagnosis
    # ──────────────────────────────────────

    def _diagnose_student(
        self, session: TutoringSession, student_msg: str
    ) -> DiagnosticResult:
        """Analyze the student's response to determine their level."""
        result = DiagnosticResult()
        msg_lower = student_msg.lower()
        word_count = len(student_msg.split())
        
        # Indicators of different levels
        beginner_signals = [
            "i don't know", "what is", "what's", "i have no idea",
            "never heard", "confused", "don't understand", "huh",
            "idk", "no clue", "explain",
        ]
        
        intermediate_signals = [
            "i think", "maybe because", "is it like", "i remember",
            "so basically", "i've heard", "i know a little",
        ]
        
        advanced_signals = [
            "because", "therefore", "the reason is", "this implies",
            "trade-off", "alternatively", "however", "specifically",
            "in my experience", "the key insight",
        ]
        
        expert_signals = [
            "according to", "the paper by", "the algorithm",
            "complexity is", "formally", "proof", "theorem",
            "implementation detail", "the specification",
        ]
        
        # Score each level
        scores = {
            StudentLevel.BEGINNER: sum(1 for s in beginner_signals if s in msg_lower),
            StudentLevel.INTERMEDIATE: sum(1 for s in intermediate_signals if s in msg_lower),
            StudentLevel.ADVANCED: sum(1 for s in advanced_signals if s in msg_lower),
            StudentLevel.EXPERT: sum(1 for s in expert_signals if s in msg_lower),
        }
        
        # Longer, more detailed responses indicate higher level
        if word_count > 50:
            scores[StudentLevel.ADVANCED] += 1
        if word_count > 100:
            scores[StudentLevel.EXPERT] += 1
        if word_count < 10:
            scores[StudentLevel.BEGINNER] += 1
        
        # Pick highest
        best_level = max(scores, key=scores.get)
        best_score = scores[best_level]
        
        if best_score == 0:
            # No signals — keep current level
            result.level = session.student_level
            result.confidence = 0.5
        else:
            result.level = best_level
            result.confidence = min(1.0, best_score * 0.25)
        
        # Detect knowledge gaps (questions the student asks)
        if "?" in student_msg:
            questions = [q.strip() + "?" for q in student_msg.split("?") if q.strip()]
            result.knowledge_gaps = questions[:3]
        
        # Detect strengths (correct terms they use)
        tech_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', student_msg)  # CamelCase
        if tech_terms:
            result.strengths = tech_terms[:3]
        
        return result

    # ──────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────

    def _call_llm(
        self, prompt: str, system_prompt: str = "", temperature: float = 0.7
    ) -> str:
        """Call the LLM generation function safely."""
        try:
            result = self.generate_fn(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )
            if hasattr(result, 'answer'):
                return result.answer
            if hasattr(result, 'error') and result.error:
                return f"[LLM Error: {result.error}]"
            return str(result)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "[Unable to generate response — using cached knowledge]"

    def _build_system_prompt(self, session: TutoringSession) -> str:
        """Build the full system prompt combining technique + level + research."""
        parts = [
            "You are an ELITE expert tutor with decades of teaching experience.",
            "You adapt your teaching to the student's exact level and learning style.",
            "",
            _TECHNIQUE_PROMPTS.get(session.current_technique, ""),
            "",
            _LEVEL_PROMPTS.get(session.student_level, ""),
        ]
        
        if session.lesson_plan:
            parts.append(f"\nLESSON PLAN: {json.dumps(session.lesson_plan)}")
        
        return "\n".join(parts)

    def _build_lesson_plan(
        self, topic: str, research: Optional[ResearchIntel] = None
    ) -> List[str]:
        """Generate a structured lesson plan for the topic."""
        context = ""
        if research and research.expert_insights:
            context = (
                "\nUse these expert insights to inform the plan:\n"
                + "\n".join(research.expert_insights[:3])
            )
        
        prompt = (
            f"Create a lesson plan for teaching: {topic}\n"
            f"Generate exactly 5-7 steps, ordered from foundation to mastery.\n"
            f"Each step should be a short phrase (5-10 words max).\n"
            f"Format: one step per line, no numbers or bullets.\n"
            f"{context}"
        )
        
        result = self._call_llm(prompt, "You are a curriculum designer.", temperature=0.5)
        
        # Parse into list
        lines = [line.strip().lstrip("0123456789.-) ") for line in result.split("\n")]
        plan = [line for line in lines if line and len(line) > 5 and len(line) < 100]
        
        if not plan:
            # Fallback generic plan
            plan = [
                f"What is {topic} and why it matters",
                "Core concepts and terminology",
                f"How {topic} works step by step",
                "Real-world applications and examples",
                "Common mistakes and misconceptions",
                "Practice problems and exercises",
                "Advanced topics and next steps",
            ]
        
        return plan[:7]

    def _select_technique(self, topic: str) -> TeachingTechnique:
        """Select the best teaching technique for a topic."""
        topic_lower = topic.lower()
        
        # Math/science → scaffolding (build layer by layer)
        if any(w in topic_lower for w in [
            "math", "calculus", "algebra", "physics", "chemistry",
            "algorithm", "data structure", "proof", "theorem",
        ]):
            return TeachingTechnique.SCAFFOLDING
        
        # Abstract concepts → analogy bridge
        if any(w in topic_lower for w in [
            "quantum", "philosophy", "theory", "abstract",
            "consciousness", "relativity", "economics",
        ]):
            return TeachingTechnique.ANALOGY_BRIDGE
        
        # Practical skills → Feynman
        if any(w in topic_lower for w in [
            "programming", "code", "python", "javascript", "web",
            "cooking", "fitness", "design", "build", "create",
        ]):
            return TeachingTechnique.FEYNMAN
        
        # Large topics → chunking
        if any(w in topic_lower for w in [
            "history", "overview", "complete guide", "everything about",
            "introduction to", "full course",
        ]):
            return TeachingTechnique.CHUNKING
        
        # Default → scaffolding (most universally effective)
        return TeachingTechnique.SCAFFOLDING

    def _check_followup_knowledge(
        self, session: TutoringSession, student_msg: str
    ) -> float:
        """Check if a student's follow-up question requires more research."""
        # If student asks about something very specific we might not know
        specific_indicators = [
            "latest", "newest", "recent", "2024", "2025", "2026",
            "current", "today", "right now", "updated",
            "specific paper", "who invented", "exact number",
        ]
        
        msg_lower = student_msg.lower()
        specificity_score = sum(
            0.15 for indicator in specific_indicators 
            if indicator in msg_lower
        )
        
        return min(1.0, specificity_score)

    # ──────────────────────────────────────
    # API-Compatible Methods
    # ──────────────────────────────────────

    def api_start_session(self, topic: str) -> Dict[str, Any]:
        """API endpoint compatible: start a session and return opening message."""
        session = self.start_session(topic)
        opening = self.begin_teaching(session)
        
        return {
            "session_id": session.session_id,
            "topic": session.topic,
            "technique": session.current_technique.value,
            "research_used": session.research_triggered,
            "lesson_plan": session.lesson_plan,
            "opening_message": opening,
        }

    def api_respond(self, session_id: str, message: str) -> Dict[str, Any]:
        """API endpoint compatible: process student response."""
        session = self.get_session(session_id)
        if not session:
            return {"error": f"Session not found: {session_id}"}
        
        response = self.respond_to_student(session, message)
        
        return {
            "session_id": session_id,
            "response": response,
            "student_level": session.student_level.value,
            "technique": session.current_technique.value,
            "lesson_progress": {
                "current_step": session.current_lesson_step + 1,
                "total_steps": len(session.lesson_plan),
                "current_topic": (
                    session.lesson_plan[session.current_lesson_step]
                    if session.lesson_plan else ""
                ),
            },
            "research_used": session.research_triggered,
        }
