"""
Deep Web Intelligence Researcher
────────────────────────────────
Autonomously queries the Deep Web (Academic ArXiv) and Dark Web (Ahmia.fi) 
to compile highly technical, classified intelligence dossiers.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DeepWebResearcher:
    """
    Intelligence bot that runs targeted research queries against unindexed databases
    and synthesizes results into a markdown dossier.
    """
    
    def __init__(self, agent_controller):
        self.agent = agent_controller
        self.memory = getattr(agent_controller, 'memory', None)
        self.intel_dir = Path("data/intel")
        self.intel_dir.mkdir(parents=True, exist_ok=True)
        
    def compile_dossier(self, target_topic: str):
        """Execute the deep research pipeline."""
        
        print(f"\n🕵️‍♂️ [DEEP WEB RESEARCHER] Initializing Intel Gathering on: '{target_topic}'")
        
        # 1. Fetch raw data using the pre-built advanced_web_search tool
        from agents.tools.web_search import advanced_web_search
        
        print("   => 📡 Querying ArXiv Deep Web (Academic & Scientific)...")
        deep_data = advanced_web_search(target_topic, network="deep", max_results=5)
        
        print("   => 📡 Querying Ahmia Dark Web (Tor Hidden Services)...")
        dark_data = advanced_web_search(target_topic, network="dark", max_results=5)
        
        raw_intel = {
            "target": target_topic,
            "deep_web_findings": deep_data["results"],
            "dark_web_findings": dark_data["results"]
        }
        
        if not raw_intel["deep_web_findings"] and not raw_intel["dark_web_findings"]:
            print(f"\n❌ [DEEP WEB RESEARCHER] No intelligence found on '{target_topic}' across restricted networks.")
            return
            
        print(f"   => 📥 Retrieved {len(raw_intel['deep_web_findings'])} Deep Web objects and {len(raw_intel['dark_web_findings'])} Dark Web objects.")
        print("   => 🧠 Universal Logic Engine is synthesizing the Intelligence Dossier...")
        
        # Inject Expert Principles from Memory
        memory_context = ""
        if self.memory:
            memory_context = self.memory.build_context(target_topic)
            if memory_context:
                print("   => 💡 Injecting deduced First Principles into reasoning context.")

        prompt = self._build_dossier_prompt(target_topic, raw_intel, memory_context)
        dossier_content = self.agent.generate_fn(prompt)
        
        # Save to disk
        filename = f"dossier_{target_topic.replace(' ', '_').lower()}.md"
        filepath = self.intel_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(dossier_content)
            
        print("\n✅ [DEEP WEB RESEARCHER] Operation Complete.")
        print(f"📄 Classified Dossier written to: {filepath.absolute()}")
        return filepath

    def _build_dossier_prompt(self, target_topic: str, raw_intel: dict, memory_context: str = "") -> str:
        """Construct the system prompt for the logic engine to synthesize the dossier."""
        prompt_parts = [
            "You are the Chief Intelligence Analyst acting as a Universal Logic Engine.",
            "Your objective is to review raw intelligence intercepted from the Deep Web (Academic/Science)",
            "and the Dark Web (Tor .onion links) regarding the target topic.",
            f"\nTARGET TOPIC: {target_topic}"
        ]
        
        if memory_context:
            prompt_parts.append(f"\nAPPLY THESE EXPERT PRINCIPLES TO YOUR ANALYSIS:\n{memory_context}")
            
        prompt_parts.extend([
            "\nRAW INTERCEPTS:",
            "```json",
            json.dumps(raw_intel, indent=2),
            "```",
            "\nYOUR DIRECTIVE:",
            "Write a highly structured, classified \"Intelligence Dossier\" summarizing this data.",
            "Apply your Expert Principles to find mathematical/scientific insights within the raw data that ordinary analysts would miss.",
            "No conversational filler. Write from an emotionless, analytical, intelligence perspective.",
            "\nThe Dossier must include:",
            "1. EXECUTIVE SUMMARY: High-level overview of the findings.",
            "2. DEEP WEB ANALYSIS (Academic/Scientific): What are the bleeding-edge academic or scientific perspectives on this? Cite the ArXiv titles.",
            "3. DARK WEB ANALYSIS (Underground/Tor): What is the underground/hacker narrative on this? Summarize the Ahmia snippets.",
            "4. FIRST PRINCIPLES SYNTHESIS: Where do the scientific and underground realities intersect according to your injected axioms? What is the fundamental mechanism at play here?",
            "\nFormat entirely in Markdown."
        ])
        
        return "\n".join(prompt_parts)
