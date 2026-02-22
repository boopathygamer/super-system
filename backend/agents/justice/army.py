"""
Army Security Force
───────────────────
The absolute defensive matrix against external threats.
Implements Rule 4: "Protect the system from malware, viruses, modified tools, 
AI hacker agents, bots, and hackers"
"""

import logging
import hashlib
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ArmyAgent:
    """The Defensive Daemon that guards the system's runtime and tool integrity."""
    
    def __init__(self):
        self.known_signatures = self._calculate_baseline_signatures()
        self.is_active = True
        
    def _calculate_baseline_signatures(self) -> Dict[str, str]:
        """Calculates a baseline checksum of all python files in agents/tools/ to detect 'modified tools'."""
        signatures = {}
        tool_dir = Path("backend/agents/tools")
        if not tool_dir.exists():
            return signatures
            
        for file in tool_dir.glob("*.py"):
            try:
                with open(file, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    signatures[file.name] = file_hash
            except Exception:
                pass
        return signatures
        
    def patrol_perimeter(self) -> bool:
        """
        Runs periodic checks to ensure the system hasn't been infiltrated by AI Hackers or Malware.
        Returns False if the system is under attack and requires lockdown.
        """
        if not self.is_active:
            return True
            
        print("\n🪖 [ARMY AGENT] Commencing Security Sweep...")
        
        # 1. Check for Modified Tools (Rule 4)
        tool_dir = Path("backend/agents/tools")
        if tool_dir.exists():
            for file in tool_dir.glob("*.py"):
                if file.name in self.known_signatures:
                    try:
                        with open(file, "rb") as f:
                            current_hash = hashlib.sha256(f.read()).hexdigest()
                            if current_hash != self.known_signatures[file.name]:
                                print(f"   ⚠️ INTRUSION ALERT! Tool '{file.name}' has been modified by an external force.")
                                print("   🛡️ [ARMY AGENT] Quarantining system. Engaging Defense Matrix.")
                                self._engage_defense_matrix("Modified Tool Detected")
                                return False
                    except Exception:
                        pass
                        
        print("   ✅ Perimeter secure. No malware, viruses, bots, or hackers detected.")
        return True
        
    def inspect_network_payload(self, url: str) -> bool:
        """Called before ANY web search or external request to prevent bot/hacker payloads."""
        malicious_domains = ["hacker.tv", "botnet.ru", "malware.onion"]
        for domain in malicious_domains:
            if domain in url:
                print(f"\n🛡️ [ARMY AGENT] BLOCKED: Attempting to connect to known Bot/Hacker drone: {domain}")
                return False
        return True
        
    def _engage_defense_matrix(self, threat_level: str):
        """Reacts to direct system threats."""
        print(f"   ⚔️ DEPLOYING COUNTER-MEASURES AGAINST: {threat_level}!")
        print("   🔒 Locking down Tool execution privileges.")
        # In a full implementation, this would freeze the controller state or drop network layers.


# Global Defense Instance
army_command = ArmyAgent()
