"""
Threat Guard Tool — Agent-accessible threat scanning and remediation tools.
══════════════════════════════════════════════════════════════════════════════

Registers on the global tool registry so the agent (or controller pipeline)
can scan files, URLs, and content for viruses/malware, then quarantine or
destroy threats with user approval.

Tools:
  threat_scan_file       — Scan a file path for threats (read-only)
  threat_scan_url        — Scan a URL for phishing/malware
  threat_scan_content    — Scan text content for malicious patterns
  threat_quarantine_file — Quarantine a detected threat
  threat_destroy_file    — Permanently destroy a detected threat
"""

import logging
from pathlib import Path
from typing import Optional

from agents.tools.registry import registry, RiskLevel

logger = logging.getLogger(__name__)

# Lazy-init scanner singleton (avoid import-time side effects)
_scanner_instance = None


def _get_scanner():
    """Get or create the ThreatScanner singleton."""
    global _scanner_instance
    if _scanner_instance is None:
        from agents.safety.threat_scanner import ThreatScanner
        try:
            from config.settings import DATA_DIR
            quarantine_dir = str(DATA_DIR / "threat_quarantine")
        except ImportError:
            quarantine_dir = None
        _scanner_instance = ThreatScanner(quarantine_dir=quarantine_dir)
    return _scanner_instance


# ──────────────────────────────────────────────
# Active scan reports cache (for quarantine/destroy referencing)
# ──────────────────────────────────────────────
_active_reports = {}


@registry.register(
    name="threat_scan_file",
    description=(
        "Scan a file for viruses, malware, trojans, ransomware, and other threats. "
        "Uses 4-layer defense: signature matching, heuristic analysis, URL inspection, "
        "and behavioral analysis. Returns a detailed threat report."
    ),
    risk_level=RiskLevel.LOW,
    parameters={
        "file_path": "Absolute path to the file to scan",
    },
)
def threat_scan_file(file_path: str) -> dict:
    """Scan a file through the ThreatScanner engine."""
    scanner = _get_scanner()

    path = Path(file_path)
    if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    try:
        report = scanner.scan_file(file_path)

        # Cache for later quarantine/destroy
        _active_reports[report.scan_id] = report

        result = {
            "success": True,
            "scan_id": report.scan_id,
            "is_threat": report.is_threat,
            "summary": report.summary(),
        }

        if report.is_threat:
            result.update({
                "threat_type": report.threat_type.value if report.threat_type else None,
                "severity": report.severity.value if report.severity else None,
                "confidence": f"{report.confidence:.1%}",
                "recommended_action": report.recommended_action.value,
                "evidence_count": len(report.evidence),
                "detailed_report": report.detailed_report(),
                "alert": (
                    f"🚨 THREAT DETECTED: {report.threat_type.value.upper() if report.threat_type else 'UNKNOWN'} "
                    f"({report.severity.emoji if report.severity else '⚠️'} {report.severity.value.upper() if report.severity else 'UNKNOWN'}) — "
                    f"Awaiting user approval for remediation. "
                    f"Use scan_id '{report.scan_id}' to quarantine or destroy."
                ),
            })
        else:
            result["message"] = "✅ File is clean — no threats detected."

        return result

    except Exception as e:
        logger.error(f"threat_scan_file error: {e}")
        return {"success": False, "error": f"Scan failed: {e}"}


@registry.register(
    name="threat_scan_url",
    description=(
        "Scan a URL for phishing, malicious domains, IDN homograph attacks, "
        "suspicious TLDs, and other URL-based threats."
    ),
    risk_level=RiskLevel.LOW,
    parameters={
        "url": "The URL to scan for threats",
    },
)
def threat_scan_url(url: str) -> dict:
    """Scan a URL for phishing and malicious patterns."""
    scanner = _get_scanner()

    try:
        report = scanner.scan_url(url)
        _active_reports[report.scan_id] = report

        result = {
            "success": True,
            "scan_id": report.scan_id,
            "url": url,
            "is_threat": report.is_threat,
            "summary": report.summary(),
        }

        if report.is_threat:
            result.update({
                "threat_type": report.threat_type.value if report.threat_type else None,
                "severity": report.severity.value if report.severity else None,
                "confidence": f"{report.confidence:.1%}",
                "evidence_count": len(report.evidence),
                "detailed_report": report.detailed_report(),
                "alert": (
                    f"🚨 MALICIOUS URL DETECTED — "
                    f"DO NOT visit this URL. "
                    f"Confidence: {report.confidence:.1%}"
                ),
            })
        else:
            result["message"] = "✅ URL appears safe — no threats detected."

        return result

    except Exception as e:
        logger.error(f"threat_scan_url error: {e}")
        return {"success": False, "error": f"URL scan failed: {e}"}


@registry.register(
    name="threat_scan_content",
    description=(
        "Scan text content for malicious patterns like encoded payloads, "
        "shellcode, reverse shells, ransomware indicators, and more."
    ),
    risk_level=RiskLevel.LOW,
    parameters={
        "content": "The text content to scan for threats",
        "source": "Label describing the content source (optional)",
    },
)
def threat_scan_content(content: str, source: str = "inline") -> dict:
    """Scan text content for malicious patterns."""
    scanner = _get_scanner()

    try:
        report = scanner.scan_content(content, source=source)
        _active_reports[report.scan_id] = report

        result = {
            "success": True,
            "scan_id": report.scan_id,
            "is_threat": report.is_threat,
            "summary": report.summary(),
        }

        if report.is_threat:
            result.update({
                "threat_type": report.threat_type.value if report.threat_type else None,
                "severity": report.severity.value if report.severity else None,
                "confidence": f"{report.confidence:.1%}",
                "evidence_count": len(report.evidence),
                "detailed_report": report.detailed_report(),
            })
        else:
            result["message"] = "✅ Content is clean — no threats detected."

        return result

    except Exception as e:
        logger.error(f"threat_scan_content error: {e}")
        return {"success": False, "error": f"Content scan failed: {e}"}


@registry.register(
    name="threat_quarantine",
    description=(
        "Quarantine a detected threat — moves the file to a secure vault "
        "with full metadata preservation. Requires a scan_id from a prior scan."
    ),
    risk_level=RiskLevel.HIGH,
    parameters={
        "scan_id": "Scan ID from a prior threat_scan_file result",
    },
)
def threat_quarantine(scan_id: str) -> dict:
    """Quarantine a previously scanned threat."""
    scanner = _get_scanner()

    report = _active_reports.get(scan_id)
    if not report:
        return {"success": False, "error": f"Scan ID '{scan_id}' not found. Run a scan first."}

    if not report.is_threat:
        return {"success": False, "error": "File was not flagged as a threat — nothing to quarantine."}

    try:
        result = scanner.quarantine(report)
        if result["success"]:
            result["message"] = (
                f"🔒 File quarantined successfully.\n"
                f"  Original: {result['original_path']}\n"
                f"  Vault: {result['quarantine_path']}\n"
                f"  Use 'threat_destroy' with scan_id '{scan_id}' to permanently destroy."
            )
        return result

    except Exception as e:
        logger.error(f"threat_quarantine error: {e}")
        return {"success": False, "error": f"Quarantine failed: {e}"}


@registry.register(
    name="threat_destroy",
    description=(
        "Permanently destroy a detected threat using secure 3-pass overwrite "
        "(zeros → ones → random) with cryptographic proof of destruction. "
        "This action is IRREVERSIBLE. Requires a scan_id from a prior scan."
    ),
    risk_level=RiskLevel.CRITICAL,
    parameters={
        "scan_id": "Scan ID from a prior threat_scan_file result",
    },
)
def threat_destroy(scan_id: str) -> dict:
    """Permanently destroy a previously scanned threat."""
    scanner = _get_scanner()

    report = _active_reports.get(scan_id)
    if not report:
        return {"success": False, "error": f"Scan ID '{scan_id}' not found. Run a scan first."}

    if not report.is_threat:
        return {"success": False, "error": "File was not flagged as a threat — refusing to destroy."}

    try:
        result = scanner.destroy(report)
        if result["success"]:
            proof = result["destruction_proof"]
            result["message"] = (
                f"🔥 THREAT DESTROYED — 100% Verified\n"
                f"  File: {proof['original_path']}\n"
                f"  Hash (pre-destruction): {proof['pre_destruction_hash']}\n"
                f"  Overwrite: {proof['overwrite_passes']}-pass ({', '.join(proof['methods'])})\n"
                f"  Verified deleted: {proof['verified_deleted']}\n"
                f"  Proof hash: {proof['proof_hash']}\n"
                f"\n"
                f"  ✅ Cryptographic proof of destruction generated.\n"
                f"  🔐 This file has been irrecoverably destroyed."
            )
            # Generate and attach full proof document
            result["proof_document"] = scanner.generate_proof(report)

        return result

    except Exception as e:
        logger.error(f"threat_destroy error: {e}")
        return {"success": False, "error": f"Destruction failed: {e}"}
