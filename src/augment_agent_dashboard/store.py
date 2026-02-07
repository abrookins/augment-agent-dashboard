"""File-based session store with locking for concurrent access."""

import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .models import AgentSession, SessionMessage, SessionStatus


def get_dashboard_dir() -> Path:
    """Get the dashboard data directory."""
    home = Path.home()
    dashboard_dir = home / ".augment" / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    return dashboard_dir


def get_sessions_file() -> Path:
    """Get the sessions data file path."""
    return get_dashboard_dir() / "sessions.json"


def get_lock_file() -> Path:
    """Get the lock file path."""
    return get_dashboard_dir() / "sessions.lock"


class SessionStore:
    """Thread-safe file-based session store using file locking."""

    def __init__(self, sessions_file: Path | None = None):
        """Initialize the store.

        Args:
            sessions_file: Path to sessions JSON file.
                Defaults to ~/.augment/dashboard/sessions.json
        """
        self.sessions_file = sessions_file or get_sessions_file()
        self.lock_file = self.sessions_file.parent / f"{self.sessions_file.stem}.lock"
        # Ensure parent directory exists
        self.sessions_file.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _file_lock(self, exclusive: bool = True) -> Iterator[None]:
        """Acquire a file lock for safe concurrent access.

        Args:
            exclusive: If True, acquire exclusive (write) lock.
                If False, shared (read) lock.
        """
        # Create lock file if it doesn't exist
        self.lock_file.touch(exist_ok=True)

        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fd = os.open(str(self.lock_file), os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, lock_type)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _read_sessions(self) -> dict[str, AgentSession]:
        """Read all sessions from file. Must be called within a lock."""
        if not self.sessions_file.exists():
            return {}

        try:
            with open(self.sessions_file) as f:
                data = json.load(f)
            return {sid: AgentSession.from_dict(s) for sid, s in data.items()}
        except (json.JSONDecodeError, KeyError):
            return {}

    def _write_sessions(self, sessions: dict[str, AgentSession]) -> None:
        """Write all sessions to file. Must be called within a lock."""
        data = {sid: s.to_dict() for sid, s in sessions.items()}
        # Write atomically using temp file
        temp_file = self.sessions_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.sessions_file)

    def get_session(self, session_id: str) -> AgentSession | None:
        """Get a session by ID."""
        with self._file_lock(exclusive=False):
            sessions = self._read_sessions()
            return sessions.get(session_id)

    def get_all_sessions(self) -> list[AgentSession]:
        """Get all sessions, sorted by last activity (most recent first)."""
        with self._file_lock(exclusive=False):
            sessions = self._read_sessions()
            return sorted(
                sessions.values(), key=lambda s: s.last_activity, reverse=True
            )

    def get_active_sessions(self) -> list[AgentSession]:
        """Get only active/idle sessions (not stopped)."""
        sessions = self.get_all_sessions()
        return [s for s in sessions if s.status != SessionStatus.STOPPED]

    def upsert_session(self, session: AgentSession) -> None:
        """Create or update a session."""
        with self._file_lock(exclusive=True):
            sessions = self._read_sessions()
            sessions[session.session_id] = session
            self._write_sessions(sessions)

    def update_session_status(
        self,
        session_id: str,
        status: SessionStatus,
        current_task: str | None = None,
    ) -> AgentSession | None:
        """Update session status and optionally current task."""
        with self._file_lock(exclusive=True):
            sessions = self._read_sessions()
            if session_id not in sessions:
                return None
            session = sessions[session_id]
            session.status = status
            session.last_activity = datetime.now(timezone.utc)
            if current_task is not None:
                session.current_task = current_task
            self._write_sessions(sessions)
            return session

    def add_message(self, session_id: str, message: SessionMessage) -> bool:
        """Add a message to a session. Returns True if successful."""
        with self._file_lock(exclusive=True):
            sessions = self._read_sessions()
            if session_id not in sessions:
                return False
            sessions[session_id].messages.append(message)
            sessions[session_id].last_activity = datetime.now(timezone.utc)
            self._write_sessions(sessions)
            return True

    def add_dashboard_message(self, session_id: str, message: str) -> bool:
        """Add a message from the dashboard for the agent to see."""
        with self._file_lock(exclusive=True):
            sessions = self._read_sessions()
            if session_id not in sessions:
                return False
            sessions[session_id].pending_dashboard_messages.append(message)
            self._write_sessions(sessions)
            return True

    def get_and_clear_dashboard_messages(self, session_id: str) -> list[str]:
        """Get pending dashboard messages and clear them."""
        with self._file_lock(exclusive=True):
            sessions = self._read_sessions()
            if session_id not in sessions:
                return []
            messages = sessions[session_id].pending_dashboard_messages.copy()
            sessions[session_id].pending_dashboard_messages = []
            self._write_sessions(sessions)
            return messages

    def update_session_pid(self, session_id: str, pid: int) -> bool:
        """Update the agent PID for a session. Returns True if successful."""
        with self._file_lock(exclusive=True):
            sessions = self._read_sessions()
            if session_id not in sessions:
                return False
            sessions[session_id].agent_pid = pid
            sessions[session_id].last_activity = datetime.now(timezone.utc)
            self._write_sessions(sessions)
            return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if successful."""
        with self._file_lock(exclusive=True):
            sessions = self._read_sessions()
            if session_id not in sessions:
                return False
            del sessions[session_id]
            self._write_sessions(sessions)
            return True

