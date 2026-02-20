"""
Session Store — JSONL-based transcript persistence.
────────────────────────────────────────────────────
Each session is persisted as a JSONL file:
  data/sessions/<agent_id>/<session_id>.jsonl

Each line is a JSON object with:
  - role: "user" | "assistant" | "system" | "tool"
  - content: message text
  - timestamp: ISO 8601
  - metadata: optional dict (tool_name, tool_args, etc.)
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@staticmethod
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """
    File-backed session transcript store.

    Stores conversations as append-only JSONL files for
    efficient streaming writes and line-by-line reads.
    """

    def __init__(self, base_dir: str = "data/sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, agent_id: str, session_id: str) -> Path:
        agent_dir = self.base_dir / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)
        return agent_dir / f"{session_id}.jsonl"

    def append(
        self,
        agent_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Append a message to the session transcript."""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            entry["metadata"] = metadata

        path = self._session_path(agent_id, session_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read(
        self,
        agent_id: str,
        session_id: str,
        limit: Optional[int] = None,
        include_tools: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Read session transcript.

        Args:
            limit: If set, return only the last N messages
            include_tools: If False, filter out tool messages
        """
        path = self._session_path(agent_id, session_id)
        if not path.exists():
            return []

        messages = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if not include_tools and entry.get("role") == "tool":
                        continue
                    messages.append(entry)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSONL line in {path}")
                    continue

        if limit:
            messages = messages[-limit:]
        return messages

    def list_sessions(self, agent_id: str) -> List[Dict[str, Any]]:
        """List all sessions for an agent with metadata."""
        agent_dir = self.base_dir / agent_id
        if not agent_dir.exists():
            return []

        sessions = []
        for path in agent_dir.glob("*.jsonl"):
            session_id = path.stem
            stat = path.stat()

            # Read first and last lines for metadata
            first_msg = None
            last_msg = None
            msg_count = 0

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    msg_count += 1
                    try:
                        msg = json.loads(line)
                        if first_msg is None:
                            first_msg = msg
                        last_msg = msg
                    except json.JSONDecodeError:
                        continue

            sessions.append({
                "session_id": session_id,
                "agent_id": agent_id,
                "message_count": msg_count,
                "size_bytes": stat.st_size,
                "created": first_msg.get("timestamp") if first_msg else None,
                "last_active": last_msg.get("timestamp") if last_msg else None,
            })

        # Sort by last active (most recent first)
        sessions.sort(
            key=lambda s: s.get("last_active") or "",
            reverse=True,
        )
        return sessions

    def delete_session(self, agent_id: str, session_id: str) -> bool:
        """Delete a session transcript."""
        path = self._session_path(agent_id, session_id)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted session: {agent_id}/{session_id}")
            return True
        return False

    def compact(
        self,
        agent_id: str,
        session_id: str,
        summary: str,
        keep_last: int = 5,
    ):
        """
        Compact a session: replace old messages with a summary,
        keeping the last N messages intact.
        """
        messages = self.read(agent_id, session_id)
        if len(messages) <= keep_last:
            return  # Nothing to compact

        # Keep last N messages
        kept = messages[-keep_last:]
        path = self._session_path(agent_id, session_id)

        # Rewrite the file
        with open(path, "w", encoding="utf-8") as f:
            # Write compaction summary as system message
            compaction_entry = {
                "role": "system",
                "content": f"[Session compacted] Summary of {len(messages) - keep_last} earlier messages:\n{summary}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {"compacted": True, "original_count": len(messages)},
            }
            f.write(json.dumps(compaction_entry, ensure_ascii=False) + "\n")

            # Write kept messages
            for msg in kept:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        logger.info(
            f"Compacted session {session_id}: "
            f"{len(messages)} → {keep_last + 1} messages"
        )
