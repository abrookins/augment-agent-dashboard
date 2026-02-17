"""Data models for federation support."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RemoteDashboard:
    """Configuration for a remote dashboard server."""

    url: str  # Base URL like "http://machine-b.local:8080"
    name: str  # Human-friendly name like "Work Laptop"
    api_key: str | None = None  # Optional API key for authentication
    last_seen: datetime | None = None  # Last successful connection
    is_healthy: bool = True  # Whether the remote is currently reachable

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "url": self.url,
            "name": self.name,
            "api_key": self.api_key,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "is_healthy": self.is_healthy,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RemoteDashboard":
        """Create from dictionary."""
        last_seen = data.get("last_seen")
        return cls(
            url=data["url"],
            name=data["name"],
            api_key=data.get("api_key"),
            last_seen=datetime.fromisoformat(last_seen) if last_seen else None,
            is_healthy=data.get("is_healthy", True),
        )


@dataclass
class FederationConfig:
    """Configuration for dashboard federation."""

    enabled: bool = True  # Whether federation is enabled
    share_locally: bool = True  # Allow others to fetch our sessions
    api_key: str | None = None  # API key required to access this dashboard
    remote_dashboards: list[RemoteDashboard] = field(default_factory=list)
    this_machine_name: str = "This Machine"  # Display name for local machine

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "enabled": self.enabled,
            "share_locally": self.share_locally,
            "api_key": self.api_key,
            "remote_dashboards": [r.to_dict() for r in self.remote_dashboards],
            "this_machine_name": self.this_machine_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FederationConfig":
        """Create from dictionary."""
        remotes = data.get("remote_dashboards", [])
        return cls(
            enabled=data.get("enabled", True),
            share_locally=data.get("share_locally", True),
            api_key=data.get("api_key"),
            remote_dashboards=[RemoteDashboard.from_dict(r) for r in remotes],
            this_machine_name=data.get("this_machine_name", "This Machine"),
        )


@dataclass
class RemoteSession:
    """A session from a remote dashboard, with origin tracking."""

    # All original session fields
    session_id: str
    conversation_id: str
    workspace_root: str
    workspace_name: str
    status: str  # Keep as string since it comes from remote
    started_at: str  # ISO format string
    last_activity: str  # ISO format string
    current_task: str | None
    message_count: int
    last_message_preview: str | None

    # Federation-specific fields
    origin_url: str  # URL of the dashboard this session came from
    origin_name: str  # Human-friendly name of the origin
    remote_session_id: str  # Original session ID on the remote

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "workspace_root": self.workspace_root,
            "workspace_name": self.workspace_name,
            "status": self.status,
            "started_at": self.started_at,
            "last_activity": self.last_activity,
            "current_task": self.current_task,
            "message_count": self.message_count,
            "last_message_preview": self.last_message_preview,
            "origin_url": self.origin_url,
            "origin_name": self.origin_name,
            "remote_session_id": self.remote_session_id,
        }

