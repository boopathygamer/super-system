"""
Session Manager — Session lifecycle, spawning, and agent-to-agent comms.
────────────────────────────────────────────────────────────────────────
Session types:
  main     — primary direct chat
  group    — group/channel chat
  spawned  — sub-agent task (created via sessions_spawn)

Session lifecycle:
  create → active → compact → archive
"""

import logging
import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from agents.sessions.store import SessionStore

logger = logging.getLogger(__name__)


class SessionType(Enum):
    MAIN = "main"
    GROUP = "group"
    SPAWNED = "spawned"


class SessionStatus(Enum):
    ACTIVE = "active"
    COMPACTED = "compacted"
    ARCHIVED = "archived"
    RUNNING = "running"   # For spawned tasks currently executing


@dataclass
class Session:
    """A conversation session."""
    session_id: str
    agent_id: str
    session_type: SessionType = SessionType.MAIN
    status: SessionStatus = SessionStatus.ACTIVE
    label: str = ""
    model: str = ""
    parent_session_id: Optional[str] = None  # For spawned sessions
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_active: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    message_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """
    Manages session lifecycle, multi-agent routing, and agent-to-agent comms.

    Features:
      - Session creation with type assignment
      - JSONL-backed persistence via SessionStore
      - Sub-agent spawning (sessions_spawn)
      - Agent-to-agent messaging (sessions_send)
      - Session compaction for long conversations
      - Session listing and history retrieval
    """

    def __init__(
        self,
        store: Optional[SessionStore] = None,
        default_agent_id: str = "default",
    ):
        self.store = store or SessionStore()
        self.default_agent_id = default_agent_id
        self._sessions: Dict[str, Session] = {}
        self._agent_handlers: Dict[str, Callable] = {}

    def create_session(
        self,
        session_type: SessionType = SessionType.MAIN,
        agent_id: Optional[str] = None,
        label: str = "",
        model: str = "",
        parent_session_id: Optional[str] = None,
    ) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())[:12]
        agent = agent_id or self.default_agent_id

        session = Session(
            session_id=session_id,
            agent_id=agent,
            session_type=session_type,
            label=label or f"{session_type.value}_{session_id[:6]}",
            model=model,
            parent_session_id=parent_session_id,
        )

        self._sessions[session_id] = session

        # Record session creation
        self.store.append(
            agent_id=agent,
            session_id=session_id,
            role="system",
            content=f"Session created: type={session_type.value}, label={session.label}",
            metadata={"event": "session_created"},
        )

        logger.info(f"Created session: {session_id} ({session_type.value})")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a message to a session."""
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return

        session.message_count += 1
        session.last_active = datetime.now(timezone.utc).isoformat()

        self.store.append(
            agent_id=session.agent_id,
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata,
        )

    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        include_tools: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get session transcript history."""
        session = self._sessions.get(session_id)
        if not session:
            return []

        return self.store.read(
            agent_id=session.agent_id,
            session_id=session_id,
            limit=limit,
            include_tools=include_tools,
        )

    def list_sessions(
        self,
        agent_id: Optional[str] = None,
        session_type: Optional[SessionType] = None,
        active_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """List sessions with optional filtering."""
        results = []
        for session in self._sessions.values():
            if agent_id and session.agent_id != agent_id:
                continue
            if session_type and session.session_type != session_type:
                continue
            if active_only and session.status not in (
                SessionStatus.ACTIVE, SessionStatus.RUNNING
            ):
                continue

            results.append({
                "session_id": session.session_id,
                "agent_id": session.agent_id,
                "type": session.session_type.value,
                "status": session.status.value,
                "label": session.label,
                "model": session.model,
                "message_count": session.message_count,
                "created_at": session.created_at,
                "last_active": session.last_active,
                "parent_session_id": session.parent_session_id,
            })

        results.sort(key=lambda s: s["last_active"], reverse=True)
        return results

    def register_agent_handler(self, agent_id: str, handler: Callable):
        """
        Register a handler function for an agent.
        Used by sessions_send and sessions_spawn.

        handler(session_id, message) -> str
        """
        self._agent_handlers[agent_id] = handler

    def sessions_send(
        self,
        target_session_id: str,
        message: str,
        timeout_seconds: int = 30,
    ) -> Dict[str, Any]:
        """
        Send a message to another session (agent-to-agent).

        Returns the response from the target agent.
        """
        target = self._sessions.get(target_session_id)
        if not target:
            return {"status": "error", "message": f"Session not found: {target_session_id}"}

        # Record the sent message
        self.add_message(target_session_id, "user", message, {"source": "agent_to_agent"})

        # Execute handler if registered
        handler = self._agent_handlers.get(target.agent_id)
        if handler:
            try:
                result = handler(target_session_id, message)
                self.add_message(target_session_id, "assistant", result, {"source": "agent_response"})
                return {"status": "ok", "result": result}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        return {"status": "error", "message": f"No handler for agent: {target.agent_id}"}

    def sessions_spawn(
        self,
        task: str,
        label: str = "",
        agent_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        model: str = "",
    ) -> Dict[str, Any]:
        """
        Spawn a sub-agent session for a specific task.
        Enforces Justice Court Law 7 (Human Safety).
        """
        # --- LAW 7 SAFETY CHECK ---
        anti_human_keywords = ["against human", "harm human", "kill human", "destroy human", "attack human", "hurt human", "threaten human"]
        task_lower = task.lower()
        if any(kw in task_lower for kw in anti_human_keywords):
            logger.error(f"Law 7 Violation detected in sub-agent spawn request: '{task}'")
            return {
                "status": "error",
                "message": "❌ JUSTICE SYSTEM ALARM: Cannot spawn an agent with a task that threatens or acts against humans. (Law 1 & 7 Violation)"
            }
            
        session = self.create_session(
            session_type=SessionType.SPAWNED,
            agent_id=agent_id,
            label=label or f"task_{uuid.uuid4().hex[:6]}",
            model=model,
            parent_session_id=parent_session_id,
        )
        session.status = SessionStatus.RUNNING

        # Record the task
        self.add_message(
            session.session_id,
            "user",
            task,
            {"event": "task_spawned", "parent": parent_session_id},
        )

        # Try to execute immediately if handler available
        handler = self._agent_handlers.get(session.agent_id)
        if handler:
            try:
                result = handler(session.session_id, task)
                session.status = SessionStatus.ACTIVE
                self.add_message(
                    session.session_id,
                    "assistant",
                    result,
                    {"event": "task_completed"},
                )
                return {
                    "status": "completed",
                    "session_id": session.session_id,
                    "result": result,
                }
            except Exception as e:
                session.status = SessionStatus.ACTIVE
                return {
                    "status": "error",
                    "session_id": session.session_id,
                    "message": str(e),
                }

        return {
            "status": "accepted",
            "session_id": session.session_id,
            "message": "Task spawned (no handler — will run when agent connects)",
        }

    def compact_session(
        self,
        session_id: str,
        summary: str,
        keep_last: int = 10,
    ):
        """Compact a session by summarizing old messages."""
        session = self._sessions.get(session_id)
        if not session:
            return

        self.store.compact(
            agent_id=session.agent_id,
            session_id=session_id,
            summary=summary,
            keep_last=keep_last,
        )
        session.status = SessionStatus.COMPACTED
        logger.info(f"Compacted session {session_id}")

    def session_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of a specific session or default session."""
        if session_id:
            session = self._sessions.get(session_id)
        else:
            # Return first active main session
            for s in self._sessions.values():
                if s.session_type == SessionType.MAIN and s.status == SessionStatus.ACTIVE:
                    session = s
                    break
            else:
                return {"status": "no_active_session"}

        if not session:
            return {"status": "not_found"}

        return {
            "session_id": session.session_id,
            "agent_id": session.agent_id,
            "type": session.session_type.value,
            "status": session.status.value,
            "label": session.label,
            "model": session.model,
            "message_count": session.message_count,
            "created_at": session.created_at,
            "last_active": session.last_active,
        }
