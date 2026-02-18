"""Data models for the agent dashboard."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .state_machine import SessionState


class SessionStatus(str, Enum):
    """Simple status of an agent session (for backwards compatibility).

    This is derived from the more detailed SessionState.
    """

    ACTIVE = "active"  # Currently processing
    IDLE = "idle"  # Session exists but not currently active
    STOPPED = "stopped"  # Session ended

    @classmethod
    def from_state(cls, state: "SessionState") -> "SessionStatus":
        """Convert a SessionState to a simple SessionStatus."""
        from .state_machine import SessionState

        if state.is_busy():
            return cls.ACTIVE
        elif state == SessionState.ERROR:
            return cls.STOPPED
        else:
            return cls.IDLE


@dataclass
class LoopConfig:
    """Configuration for a loop prompt.

    The prompt should give clear instructions to the LLM about what work to do
    and explain what the end condition is. The end_condition is a string that
    the LLM should include in its response when it believes the work is complete.

    Example:
        prompt: "Reach 100% test coverage. If you have truly reached 100% test
                 coverage of ALL code, say 'LOOP_COMPLETE: 100% coverage achieved.'"
        end_condition: "LOOP_COMPLETE: 100% coverage achieved."
    """

    prompt: str
    end_condition: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt": self.prompt,
            "end_condition": self.end_condition,
        }

    @classmethod
    def from_dict(cls, data: dict | str) -> "LoopConfig":
        """Create from dictionary or legacy string format.

        Supports backward compatibility with old configs that stored
        prompts as plain strings.
        """
        if isinstance(data, str):
            # Legacy format: just a string prompt, no end condition
            return cls(prompt=data, end_condition="")
        return cls(
            prompt=data.get("prompt", ""),
            end_condition=data.get("end_condition", ""),
        )


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
    # State machine state (detailed)
    _state: str = "idle"  # Stored as string, accessed via property
    # Legacy status field for backwards compatibility
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
    # Review settings
    review_enabled: bool = False
    review_constraints: str | None = None  # Instructions for the review agent
    review_iteration: int = 0  # Current review iteration count
    max_review_iterations: int = 3  # Max back-and-forth with reviewer
    in_review_cycle: bool = False  # Currently in a review cycle
    review_satisfied: bool = False  # Review agent satisfied with changes
    last_reviewed_files: list[str] = field(default_factory=list)

    @property
    def state(self) -> "SessionState":
        """Get the current state as a SessionState enum."""
        from .state_machine import SessionState

        return SessionState(self._state)

    @state.setter
    def state(self, value: "SessionState") -> None:
        """Set the state and update legacy status."""
        from .state_machine import SessionState

        if isinstance(value, SessionState):
            self._state = value.value
            self.status = SessionStatus.from_state(value)
        else:
            self._state = value
            self.status = SessionStatus.from_state(SessionState(value))

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "workspace_root": self.workspace_root,
            "workspace_name": self.workspace_name,
            "state": self._state,
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
            "loop_started_at": (
                self.loop_started_at.isoformat() if self.loop_started_at else None
            ),
            # Review fields
            "review_enabled": self.review_enabled,
            "review_constraints": self.review_constraints,
            "review_iteration": self.review_iteration,
            "max_review_iterations": self.max_review_iterations,
            "in_review_cycle": self.in_review_cycle,
            "review_satisfied": self.review_satisfied,
            "last_reviewed_files": self.last_reviewed_files,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentSession":
        """Create from dictionary."""
        loop_started_at = data.get("loop_started_at")
        # Handle state - default to deriving from status for backwards compat
        state = data.get("state")
        if state is None:
            # Backwards compatibility: derive state from status
            status_val = data.get("status", "idle")
            state = status_val  # IDLE and ACTIVE map directly
        return cls(
            session_id=data["session_id"],
            conversation_id=data["conversation_id"],
            workspace_root=data["workspace_root"],
            workspace_name=data["workspace_name"],
            _state=state,
            status=SessionStatus(data.get("status", "idle")),
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
            loop_started_at=(
                datetime.fromisoformat(loop_started_at) if loop_started_at else None
            ),
            # Review fields
            review_enabled=data.get("review_enabled", False),
            review_constraints=data.get("review_constraints"),
            review_iteration=data.get("review_iteration", 0),
            max_review_iterations=data.get("max_review_iterations", 3),
            in_review_cycle=data.get("in_review_cycle", False),
            review_satisfied=data.get("review_satisfied", False),
            last_reviewed_files=data.get("last_reviewed_files", []),
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

