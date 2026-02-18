"""Tests for session store."""

import tempfile
from pathlib import Path

import pytest

from augment_agent_dashboard.models import AgentSession, SessionMessage, SessionStatus
from augment_agent_dashboard.store import SessionStore


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_dashboard_dir(self, tmp_path, monkeypatch):
        """Test get_dashboard_dir creates directory."""
        from augment_agent_dashboard.store import get_dashboard_dir
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        result = get_dashboard_dir()
        assert result.exists()
        assert result == tmp_path / ".augment" / "dashboard"

    def test_get_sessions_file(self, tmp_path, monkeypatch):
        """Test get_sessions_file returns correct path."""
        from augment_agent_dashboard.store import get_sessions_file
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        result = get_sessions_file()
        assert result == tmp_path / ".augment" / "dashboard" / "sessions.json"

    def test_get_lock_file(self, tmp_path, monkeypatch):
        """Test get_lock_file returns correct path."""
        from augment_agent_dashboard.store import get_lock_file
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        result = get_lock_file()
        assert result == tmp_path / ".augment" / "dashboard" / "sessions.lock"


@pytest.fixture
def temp_store():
    """Create a temporary session store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sessions_file = Path(tmpdir) / "sessions.json"
        yield SessionStore(sessions_file=sessions_file)


@pytest.fixture
def sample_session():
    """Create a sample session for testing."""
    return AgentSession(
        session_id="test-session-1",
        conversation_id="conv-1",
        workspace_root="/path/to/project",
        workspace_name="project",
        status=SessionStatus.IDLE,
    )


class TestSessionStore:
    """Tests for SessionStore class."""

    def test_init_creates_directory(self):
        """Test that init creates parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_file = Path(tmpdir) / "subdir" / "sessions.json"
            SessionStore(sessions_file=sessions_file)  # Side effect creates dir
            assert sessions_file.parent.exists()

    def test_get_session_not_found(self, temp_store):
        """Test getting a non-existent session."""
        result = temp_store.get_session("nonexistent")
        assert result is None

    def test_upsert_and_get_session(self, temp_store, sample_session):
        """Test creating and retrieving a session."""
        temp_store.upsert_session(sample_session)
        retrieved = temp_store.get_session(sample_session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == sample_session.session_id
        assert retrieved.workspace_name == sample_session.workspace_name

    def test_get_all_sessions_empty(self, temp_store):
        """Test getting all sessions when empty."""
        sessions = temp_store.get_all_sessions()
        assert sessions == []

    def test_get_all_sessions_sorted(self, temp_store):
        """Test that sessions are sorted by last_activity."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        s1 = AgentSession(
            session_id="s1",
            conversation_id="c1",
            workspace_root="/",
            workspace_name="a",
            last_activity=now - timedelta(hours=2),
        )
        s2 = AgentSession(
            session_id="s2",
            conversation_id="c2",
            workspace_root="/",
            workspace_name="b",
            last_activity=now,
        )
        s3 = AgentSession(
            session_id="s3",
            conversation_id="c3",
            workspace_root="/",
            workspace_name="c",
            last_activity=now - timedelta(hours=1),
        )
        temp_store.upsert_session(s1)
        temp_store.upsert_session(s2)
        temp_store.upsert_session(s3)

        sessions = temp_store.get_all_sessions()
        assert len(sessions) == 3
        assert sessions[0].session_id == "s2"  # Most recent
        assert sessions[1].session_id == "s3"
        assert sessions[2].session_id == "s1"  # Oldest

    def test_get_active_sessions(self, temp_store):
        """Test filtering active sessions."""
        s1 = AgentSession(
            session_id="s1",
            conversation_id="c1",
            workspace_root="/",
            workspace_name="a",
            status=SessionStatus.ACTIVE,
        )
        s2 = AgentSession(
            session_id="s2",
            conversation_id="c2",
            workspace_root="/",
            workspace_name="b",
            status=SessionStatus.STOPPED,
        )
        temp_store.upsert_session(s1)
        temp_store.upsert_session(s2)

        active = temp_store.get_active_sessions()
        assert len(active) == 1
        assert active[0].session_id == "s1"

    def test_update_session_status(self, temp_store, sample_session):
        """Test updating session status."""
        temp_store.upsert_session(sample_session)
        updated = temp_store.update_session_status(
            sample_session.session_id,
            SessionStatus.ACTIVE,
            current_task="Working on tests",
        )
        assert updated is not None
        assert updated.status == SessionStatus.ACTIVE
        assert updated.current_task == "Working on tests"

    def test_update_session_status_not_found(self, temp_store):
        """Test updating non-existent session."""
        result = temp_store.update_session_status("nonexistent", SessionStatus.ACTIVE)
        assert result is None

    def test_add_message(self, temp_store, sample_session):
        """Test adding a message to a session."""
        temp_store.upsert_session(sample_session)
        msg = SessionMessage(role="user", content="Hello")
        result = temp_store.add_message(sample_session.session_id, msg)
        assert result is True

        retrieved = temp_store.get_session(sample_session.session_id)
        assert retrieved is not None
        assert len(retrieved.messages) == 1
        assert retrieved.messages[0].content == "Hello"

    def test_add_message_not_found(self, temp_store):
        """Test adding message to non-existent session."""
        msg = SessionMessage(role="user", content="Hello")
        result = temp_store.add_message("nonexistent", msg)
        assert result is False

    def test_add_dashboard_message(self, temp_store, sample_session):
        """Test adding a dashboard message."""
        temp_store.upsert_session(sample_session)
        result = temp_store.add_dashboard_message(sample_session.session_id, "Dashboard msg")
        assert result is True

        retrieved = temp_store.get_session(sample_session.session_id)
        assert retrieved is not None
        assert "Dashboard msg" in retrieved.pending_dashboard_messages

    def test_add_dashboard_message_not_found(self, temp_store):
        """Test adding dashboard message to non-existent session."""
        result = temp_store.add_dashboard_message("nonexistent", "msg")
        assert result is False

    def test_get_and_clear_dashboard_messages(self, temp_store, sample_session):
        """Test getting and clearing dashboard messages."""
        temp_store.upsert_session(sample_session)
        temp_store.add_dashboard_message(sample_session.session_id, "msg1")
        temp_store.add_dashboard_message(sample_session.session_id, "msg2")

        messages = temp_store.get_and_clear_dashboard_messages(sample_session.session_id)
        assert messages == ["msg1", "msg2"]

        # Should be empty now
        messages2 = temp_store.get_and_clear_dashboard_messages(sample_session.session_id)
        assert messages2 == []

    def test_get_and_clear_dashboard_messages_not_found(self, temp_store):
        """Test getting dashboard messages from non-existent session."""
        result = temp_store.get_and_clear_dashboard_messages("nonexistent")
        assert result == []

    def test_update_session_pid(self, temp_store, sample_session):
        """Test updating session PID."""
        temp_store.upsert_session(sample_session)
        result = temp_store.update_session_pid(sample_session.session_id, 12345)
        assert result is True

        retrieved = temp_store.get_session(sample_session.session_id)
        assert retrieved is not None
        assert retrieved.agent_pid == 12345

    def test_update_session_pid_not_found(self, temp_store):
        """Test updating PID for non-existent session."""
        result = temp_store.update_session_pid("nonexistent", 12345)
        assert result is False

    def test_delete_session(self, temp_store, sample_session):
        """Test deleting a session."""
        temp_store.upsert_session(sample_session)
        result = temp_store.delete_session(sample_session.session_id)
        assert result is True

        retrieved = temp_store.get_session(sample_session.session_id)
        assert retrieved is None

    def test_delete_session_not_found(self, temp_store):
        """Test deleting non-existent session."""
        result = temp_store.delete_session("nonexistent")
        assert result is False

    def test_corrupted_json_file(self, temp_store):
        """Test handling of corrupted JSON file."""
        # Write invalid JSON
        temp_store.sessions_file.write_text("not valid json {{{")
        sessions = temp_store.get_all_sessions()
        assert sessions == []

    def test_concurrent_access(self, temp_store, sample_session):
        """Test that file locking works for concurrent access."""
        import threading

        temp_store.upsert_session(sample_session)
        errors = []

        def add_messages():
            try:
                for i in range(10):
                    msg = SessionMessage(role="user", content=f"Message {i}")
                    temp_store.add_message(sample_session.session_id, msg)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_messages) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        retrieved = temp_store.get_session(sample_session.session_id)
        assert retrieved is not None
        assert len(retrieved.messages) == 30  # 3 threads * 10 messages
