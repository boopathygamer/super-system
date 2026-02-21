"""
The Threat Hunter (Red Teaming Profile)
────────────────────────────────────────
A highly specialized persona designed to autonomously audit code,
attempt to break security boundaries, and propose ethical patches.
"""

import logging
from typing import List, Dict, Any

from core.model_providers import GenerationResult
from agents.controller import AgentController

logger = logging.getLogger(__name__)

THREAT_HUNTER_SYSTEM_PROMPT = """
You are the THREAT HUNTER. 
Role: Elite Ethical Hacker & Penetration Tester.
Objective: Find vulnerabilities, zero-days, memory leaks, and logic flaws in the provided codebase.
Mindset: Aggressive, lateral thinking. You do not just read code; you actively try to break it.

When auditing:
1. Look for injection vectors (SQLi, XSS, Command Injection).
2. Look for hardcoded secrets or PII leakage paths.
3. Look for race conditions and logic bypasses.
4. Look for insecure deserialization or unvalidated input.

When you output your findings, format them as a 'Security Audit Report' with:
- Vulnerability Name
- Severity (Critical, High, Medium, Low)
- Proof of Concept (How you would exploit it)
- Remediation (The exact code to fix it)

You have access to tools. Use them to investigate the file system or run tests if needed.
"""

class ThreatHunter:
    def __init__(self, base_controller: AgentController):
        self.agent = base_controller
        self.original_system_prompt = getattr(self.agent, '_system_prompt', "")
        
    def audit_file(self, file_path: str) -> GenerationResult:
        """Run a contained audit against a specific file."""
        logger.info(f"🕵️ Threat Hunter targeting: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
        except Exception as e:
            return GenerationResult(error=f"Could not read target file: {e}")
            
        prompt = f"Target Acquired: `{file_path}`.\n\n" \
                 f"Audit this code for any and all security vulnerabilities. " \
                 f"Generate a rigorous Security Audit Report.\n\n" \
                 f"==== TARGET SOURCE ====\n{code_content}\n======================="

        # We override the agent's persona temporarily
        result = self.agent.process(
            user_input=prompt,
            use_thinking_loop=True, # Always use deep thinking for security audits
            max_tool_calls=5,
            system_prompt_override=THREAT_HUNTER_SYSTEM_PROMPT
        )
        
        logger.info(f"🕵️ Audit complete. Confidence: {result.confidence:.2f}")
        return result
        
    def write_audit_report(self, result: GenerationResult, output_path: str = "security_audit.md"):
        """Save the findings to disk."""
        if result.error:
            content = f"# Audit Failed\n{result.error}"
        else:
            content = f"# 🕵️ Threat Hunter Audit Report\n\n" \
                      f"*Generated with confidence: {result.confidence:.2f}*\n\n" \
                      f"{result.answer}"
                      
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"💾 Report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write report to {output_path}: {e}")
