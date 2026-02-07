"""Data models for the agent dashboard."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal


class SessionStatus(str, Enum):
    """Status of an agent session."""

    ACTIVE = "active"  # Currently processing
    IDLE = "idle"  # Session exists but not currently active
    STOPPED = "stopped"  # Session ended


@dataclass
class SessionMessage:
    """A message in a session conversation."""

    role: Literal["user", "assistant", "system", "dashboard", "queued"]
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_id: str | None = None
    tool_calls: list[str] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "tool_calls": self.tool_calls,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMessage":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_id=data.get("message_id"),
            tool_calls=data.get("tool_calls"),
        )


@dataclass
class AgentSession:
    """Represents an active or recent agent session."""

    session_id: str
    conversation_id: str  # The actual Auggie session UUID
    workspace_root: str
    workspace_name: str
    status: SessionStatus = SessionStatus.IDLE
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_task: str | None = None
    messages: list[SessionMessage] = field(default_factory=list)
    pending_dashboard_messages: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    agent_pid: int | None = None  # PID of the running Auggie process
    # Quality loop settings
    loop_enabled: bool = False
    loop_count: int = 0
    loop_prompt_name: str | None = None  # Name of the selected loop prompt
    loop_started_at: datetime | None = None  # When the loop was enabled

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "workspace_root": self.workspace_root,
            "workspace_name": self.workspace_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "current_task": self.current_task,
            "messages": [m.to_dict() for m in self.messages],
            "pending_dashboard_messages": self.pending_dashboard_messages,
            "files_changed": self.files_changed,
            "tools_used": self.tools_used,
            "agent_pid": self.agent_pid,
            "loop_enabled": self.loop_enabled,
            "loop_count": self.loop_count,
            "loop_prompt_name": self.loop_prompt_name,
            "loop_started_at": self.loop_started_at.isoformat() if self.loop_started_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentSession":
        """Create from dictionary."""
        loop_started_at = data.get("loop_started_at")
        return cls(
            session_id=data["session_id"],
            conversation_id=data["conversation_id"],
            workspace_root=data["workspace_root"],
            workspace_name=data["workspace_name"],
            status=SessionStatus(data["status"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            current_task=data.get("current_task"),
            messages=[SessionMessage.from_dict(m) for m in data.get("messages", [])],
            pending_dashboard_messages=data.get("pending_dashboard_messages", []),
            files_changed=data.get("files_changed", []),
            tools_used=data.get("tools_used", []),
            agent_pid=data.get("agent_pid"),
            loop_enabled=data.get("loop_enabled", False),
            loop_count=data.get("loop_count", 0),
            loop_prompt_name=data.get("loop_prompt_name"),
            loop_started_at=datetime.fromisoformat(loop_started_at) if loop_started_at else None,
        )

    @property
    def message_count(self) -> int:
        """Number of messages in the session."""
        return len(self.messages)

    @property
    def last_message_preview(self) -> str | None:
        """Preview of the last message."""
        if not self.messages:
            return None
        last = self.messages[-1]
        content = last.content[:100]
        if len(last.content) > 100:
            content += "..."
        return content

