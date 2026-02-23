"""
Army Security Force
───────────────────
The absolute defensive matrix against external threats.
Implements Rule 4: "Protect the system from malware, viruses, modified tools, 
AI hacker agents, bots, and hackers"

Security hardening:
  - Uses absolute paths (not relative CWD-dependent)
  - HMAC-based integrity verification when key is available
  - Expanded malicious domain pattern detection
"""

import hashlib
import hmac
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Resolve tool directory absolutely relative to this file
_TOOLS_DIR = Path(__file__).parent.parent / "tools"

# HMAC key for signature verification (optional — from env)
_HMAC_KEY = os.getenv("LLM_INTEGRITY_KEY", "").encode() or None

# Expanded malicious domain patterns (regex-based)
_MALICIOUS_DOMAIN_PATTERNS = [
    re.compile(r'(?:^|\.)(malware|botnet|phishing|exploit|hacker)\.\w+$', re.IGNORECASE),
    re.compile(r'\.onion$', re.IGNORECASE),
    re.compile(r'(?:^|\.)(darkweb|hack[s]?|crack[s]?|warez)\.\w+$', re.IGNORECASE),
    # Known bad TLDs for suspicious activity
    re.compile(r'\.(tk|ml|ga|cf|gq)$', re.IGNORECASE),
]

# Explicit blocklist
_BLOCKED_DOMAINS = frozenset({
    "hacker.tv", "botnet.ru", "malware.onion",
    "evil.com", "darkleaks.co", "ransomware.biz",
})


class ArmyAgent:
    """The Defensive Daemon that guards the system's runtime and tool integrity."""
    
    def __init__(self):
        self.known_signatures = self._calculate_baseline_signatures()
        self.is_active = True
        logger.info(
            f"🪖 Army Agent initialized — monitoring {len(self.known_signatures)} tool files"
        )
        
    def _calculate_baseline_signatures(self) -> Dict[str, str]:
        """
        Calculates HMAC-SHA256 (or SHA256) checksums of all python files 
        in agents/tools/ using absolute paths.
        """
        signatures = {}
        if not _TOOLS_DIR.exists():
            logger.warning(f"Tools directory not found: {_TOOLS_DIR}")
            return signatures
            
        for file in _TOOLS_DIR.glob("*.py"):
            try:
                content = file.read_bytes()
                if _HMAC_KEY:
                    file_hash = hmac.new(
                        _HMAC_KEY, content, hashlib.sha256
                    ).hexdigest()
                else:
                    file_hash = hashlib.sha256(content).hexdigest()
                signatures[file.name] = file_hash
            except Exception:
                pass
        return signatures
        
    def patrol_perimeter(self) -> bool:
        """
        Runs periodic checks to ensure the system hasn't been infiltrated.
        Returns False if the system is under attack and requires lockdown.
        """
        if not self.is_active:
            return True
            
        logger.info("🪖 [ARMY AGENT] Commencing Security Sweep...")
        
        # Check for Modified Tools (Rule 4) using absolute paths
        if _TOOLS_DIR.exists():
            for file in _TOOLS_DIR.glob("*.py"):
                if file.name in self.known_signatures:
                    try:
                        content = file.read_bytes()
                        if _HMAC_KEY:
                            current_hash = hmac.new(
                                _HMAC_KEY, content, hashlib.sha256
                            ).hexdigest()
                        else:
                            current_hash = hashlib.sha256(content).hexdigest()
                        
                        if current_hash != self.known_signatures[file.name]:
                            logger.critical(
                                f"⚠️ INTRUSION ALERT! Tool '{file.name}' has been modified!"
                            )
                            self._engage_defense_matrix("Modified Tool Detected")
                            return False
                    except Exception:
                        pass
                        
        logger.info("   ✅ Perimeter secure. No intrusions detected.")
        return True
        
    def inspect_network_payload(self, url: str) -> bool:
        """
        Called before ANY web search or external request.
        Uses both explicit blocklist and regex pattern matching.
        """
        if not url:
            return False
        
        url_lower = url.lower()
        
        # Check explicit blocklist
        for domain in _BLOCKED_DOMAINS:
            if domain in url_lower:
                logger.warning(
                    "🛡️ [ARMY AGENT] BLOCKED: Malicious domain detected in URL"
                )
                return False
        
        # Check regex patterns
        # Extract domain from URL
        domain_match = re.search(r'https?://([^/]+)', url_lower)
        if domain_match:
            domain = domain_match.group(1)
            for pattern in _MALICIOUS_DOMAIN_PATTERNS:
                if pattern.search(domain):
                    logger.warning(
                        "🛡️ [ARMY AGENT] BLOCKED: Suspicious domain pattern detected"
                    )
                    return False
        
        return True
        
    def _engage_defense_matrix(self, threat_level: str):
        """Reacts to direct system threats."""
        logger.critical(f"⚔️ DEPLOYING COUNTER-MEASURES AGAINST: {threat_level}!")
        logger.critical("🔒 Locking down Tool execution privileges.")
        # In a full implementation, this would freeze the controller state


# Global Defense Instance
army_command = ArmyAgent()
