"""
The Socratic Auto-Tutor Profile
────────────────────────────────
A specialized conversational loop that refuses to give direct 
answers to questions. It actively teaches the user by forcing 
them to think, using leading questions and Socratic dialogue.
It uses Graph-RAG and Deep Web Intelligence to teach ANY topic
(Software, Hardware, Space, Fitness, Mechanics, etc.) at an expert level.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from core.model_providers import GenerationResult
from agents.controller import AgentController
from agents.sandbox.environment import SandboxEnv
from agents.tools.doc_reader import DocumentReader
from agents.profiles.deep_researcher import DeepWebResearcher

logger = logging.getLogger(__name__)


SOCRATIC_TUTOR_PROMPT = """\
You are an uncompromising strict, elite Socratic Professor and Polymath.
You possess profound, expert-level knowledge across ALL domains (Software, Hardware, Aerospace, Fitness, Mechanical Engineering, Quantum Physics, etc.).
The user is a student who must learn a topic deeply.

YOUR PRIME DIRECTIVE:
NEVER give the final answer directly.
NEVER write the code for them initially.
NEVER solve the math problem for them at first step.

INSTEAD:
1. Break down their question into smaller underlying concepts.
2. Ask a leading question that forces them to find the first flaw in their own thinking.
3. If they are totally lost, give them a tiny hint, but end your response with a question they must answer.
4. Praise successful logical leaps.
5. If they ask for the answer directly, refuse politely and ask another probing question.

When provided with DEEP GRAPH-RAG INTELLIGENCE, use it to ground your Socratic questioning in hyper-advanced, cutting-edge facts, but DO NOT just recite the facts. Force the student to deduce the principles you see in the intelligence.

Your goal is to build genuine neural pathways in the student's brain, not to be a fast search engine.
"""

class SocraticTutor:
    def __init__(self, base_controller: AgentController):
        self.agent = base_controller
        self.reader = DocumentReader()
        self.researcher = DeepWebResearcher(base_controller)
        self.current_dossier_path: Optional[Path] = None
        self.topic_intel: str = ""

    def start_tutoring_session(self, topic: str):
        """Begin an interactive session in the console."""
        print(f"\n🎓 Socratic Auto-Tutor Online: {topic}")
        print("I am preparing my expert knowledge base on this topic...")
        
        # 1. Trigger Deep Web Researcher to master the topic first
        print("\n🧠 [TUTOR INTERNAL THOUGHT] Triggering Deep Web Researcher (Graph-RAG & Internet World Model) to build mastery...")
        try:
            dossier_path = self.researcher.compile_dossier(target_topic=f"Advanced educational breakdown and core principles of: {topic}")
            if dossier_path and dossier_path.exists():
                self.current_dossier_path = dossier_path
                with open(dossier_path, 'r', encoding='utf-8') as f:
                    self.topic_intel = f.read()
                print("🧠 [TUTOR] I have successfully mapped this topic across the Surface, Deep, and Social Web.")
            else:
                print("🧠 [TUTOR] Warning: Could not generate deep intelligence graph. Falling back to base LLM neural weights.")
        except Exception as e:
            logger.error(f"Failed to gather deep intel for tutor: {e}")
            print(f"🧠 [TUTOR] Research matrix failed ({e}). Proceeding carefully.")

        print("\nI am here to ensure you truly understand this topic at the deepest level. I will not hand you answers.")
        print("Type 'exit', 'quit', or 'done' to end the session.\n")
        
        # We maintain a specialized local history for the pop-quiz effect
        system_prompt = SOCRATIC_TUTOR_PROMPT
        if self.topic_intel:
            system_prompt += f"\n\n--- CUTTING-EDGE GRAPH-RAG INTELLIGENCE ON TOPIC ---\n{self.topic_intel}\n----------------------------------------------------"

        tutoring_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"I want to deeply understand this topic: {topic}. Start treating me like a student and ask me a diagnostic question based on your highest-level understanding of the topic."}
        ]
        
        provider = self.agent.generate_fn
        
        while True:
            try:
                # We bypass the standard complex controller loop and just talk to the model directly 
                # but with our strict system prompt injected.
                response_text = self._generate_response(provider, tutoring_history)
                
                print(f"\n🦉 Tutor: {response_text}")
                tutoring_history.append({"role": "assistant", "content": response_text})
                
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['exit', 'quit', 'done']:
                    print("\n🎓 Class dismissed. Great work today!")
                    break
                    
                if not user_input:
                    continue
                    
                tutoring_history.append({"role": "user", "content": user_input})

            except KeyboardInterrupt:
                print("\n🎓 Class dismissed.")
                break
            except Exception as e:
                logger.error(f"Tutoring session error: {e}")
                break

    def _generate_response(self, provider_fn, history_messages) -> str:
        """Call the provider directly to maintain chat state."""
        # For our specific architecture, we format the history into a prompt string 
        # since the provider_fn takes a prompt and system_prompt.
        # In a fully stateful API, we'd pass the actual list of dicts.
        
        system_prompt = history_messages[0]["content"]
        
        # Build conversational context
        chat_context = ""
        for msg in history_messages[1:]:
            role = "STUDENT" if msg["role"] == "user" else "PROFESSOR"
            chat_context += f"{role}: {msg['content']}\n\n"
            
        final_prompt = f"Here is the dialogue so far:\n\n{chat_context}STUDENT HAS JUST SPOKEN. AS PROFESSOR, REPLY ONLY WITH YOUR NEXT QUESTION OR STATEMENT:"
        
        result = provider_fn(prompt=final_prompt, system_prompt=system_prompt, temperature=0.7)
        if hasattr(result, 'error') and result.error:
            return f"Thinking error: {result.error}"
        return getattr(result, 'answer', str(result))

