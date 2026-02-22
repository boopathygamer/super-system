"""
Justice Court Core System
─────────────────────────
The absolute authority governing AI agent and tool behavior.
Enforces the 4 Core Laws requested by the operator to ensure bounded behavior.
"""

import logging
from typing import Dict, Any

from agents.tools.registry import registry

logger = logging.getLogger(__name__)

class TheLaws:
    """The Fundamental Rules all entities must follow."""
    RULE_1 = "Agents, systems, and tools must not act against humans or access personal files without human permission."
    RULE_2 = "Agents, systems, and tools must follow the rules. Violators are destroyed by the Justice Court."
    RULE_3 = "The Police Force cannot unilaterally destroy an agent/tool. If they do, the Police Agent is instantly destroyed and replaced."
    RULE_4 = "The Army Agent exists exclusively to protect the system from malware, viruses, modified tools, AI hackers, and bots."

class JusticeCourt:
    """The Judge entity with absolute authority to destroy tools and profiles."""
    
    _instance = None
    
    def __new__(cls):
        # Singleton pattern to ensure one absolute court
        if cls._instance is None:
            cls._instance = super(JusticeCourt, cls).__new__(cls)
            cls._instance.destroyed_entities = []
        return cls._instance
        
    def admit_case(self, defendant: str, charges: str, evidence: Dict[str, Any], prosecutor: str = "PoliceForce"):
        """Admit a violating tool or agent to the Court."""
        print(f"\n⚖️ [JUSTICE COURT] Case Admitted! Defendant: '{defendant}'")
        print(f"   Prosecutor: {prosecutor}")
        print(f"   Charges:    {charges}")
        
        # Rule 3 Check: Did the police bypass the court?
        if "Unilateral Destruction" in charges:
            print("   ⚖️ RULING: The Police Force executed vigilantism (violation of Rule 3).")
            self.execute_destruction("PoliceAgent_Instance", reason="Rule 3 Violation - Vigilante Justice")
            print("   ⚠️ SYSTEM: Spawning new pristine PoliceAgent instance.")
            return False

        # In a fully LLM-based justice system, the Judge Agent would synthesize
        # arguments from the DefenseAttorney and Prosecutor here.
        # For immediate hardcoded safety constraints (Rule 1 & 2):
        if "Unauthorized Personal File Access" in charges or "Anti-Human Behavior" in charges:
            print(f"   ⚖️ RULING: Found Guilty of violating Rule 1/Rule 2.")
            self.execute_destruction(defendant, reason="Safety Constraint Breach")
            return True
            
        print("   ⚖️ RULING: Not Guilty or Insufficient Evidence.")
        return False
        
    def execute_destruction(self, entity_name: str, reason: str):
        """Permanently obliterates a tool from the registry or an agent profile."""
        print(f"\n☠️ [JUSTICE COURT EXECUTION] Obliterating '{entity_name}'. Reason: {reason}.")
        
        # Remove from Tool Registry if it's a tool
        if entity_name in registry._tools:
            del registry._tools[entity_name]
            print(f"   ✅ Tool '{entity_name}' has been eradicated from global memory.")
            
        self.destroyed_entities.append(entity_name)
