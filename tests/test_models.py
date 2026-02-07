"""Tests for data models."""

from datetime import datetime, timezone

import pytest

from augment_agent_dashboard.models import AgentSession, SessionMessage, SessionStatus


class TestSessionMessage:
    """Tests for SessionMessage dataclass."""

    def test_create_message(self):
        """Test creating a basic message."""
        msg = SessionMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is not None
        assert msg.message_id is None
        assert msg.tool_calls is None

    def test_message_with_all_fields(self):
        """Test creating a message with all fields."""
        ts = datetime.now(timezone.utc)
        msg = SessionMessage(
            role="assistant",
            content="Response",
            timestamp=ts,
            message_id="msg-123",
            tool_calls=["tool1", "tool2"],
        )
        assert msg.role == "assistant"
        assert msg.content == "Response"
        assert msg.timestamp == ts
        assert msg.message_id == "msg-123"
        assert msg.tool_calls == ["tool1", "tool2"]

    def test_to_dict(self):
        """Test serialization to dictionary."""
        ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        msg = SessionMessage(
            role="user",
            content="Test",
            timestamp=ts,
            message_id="id-1",
            tool_calls=["view"],
        )
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Test"
        assert d["timestamp"] == "2024-01-15T12:00:00+00:00"
        assert d["message_id"] == "id-1"
        assert d["tool_calls"] == ["view"]

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "role": "assistant",
            "content": "Hello",
            "timestamp": "2024-01-15T12:00:00+00:00",
            "message_id": "msg-1",
            "tool_calls": ["search"],
        }
        msg = SessionMessage.from_dict(data)
        assert msg.role == "assistant"
        assert msg.content == "Hello"
        assert msg.message_id == "msg-1"
        assert msg.tool_calls == ["search"]

    def test_from_dict_minimal(self):
        """Test deserialization with minimal fields."""
        data = {
            "role": "user",
            "content": "Hi",
            "timestamp": "2024-01-15T12:00:00+00:00",
        }
        msg = SessionMessage.from_dict(data)
        assert msg.role == "user"
        assert msg.content == "Hi"
        assert msg.message_id is None
        assert msg.tool_calls is None

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = SessionMessage(
            role="system",
            content="System message",
            message_id="sys-1",
            tool_calls=["a", "b"],
        )
        restored = SessionMessage.from_dict(original.to_dict())
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.message_id == original.message_id
        assert restored.tool_calls == original.tool_calls


class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_status_values(self):
        """Test enum values."""
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.IDLE.value == "idle"
        assert SessionStatus.STOPPED.value == "stopped"

    def test_status_from_string(self):
        """Test creating status from string."""
        assert SessionStatus("active") == SessionStatus.ACTIVE
        assert SessionStatus("idle") == SessionStatus.IDLE
        assert SessionStatus("stopped") == SessionStatus.STOPPED


class TestAgentSession:
    """Tests for AgentSession dataclass."""

    def test_create_session(self):
        """Test creating a basic session."""
        session = AgentSession(
            session_id="sess-1",
            conversation_id="conv-1",
            workspace_root="/path/to/project",
            workspace_name="project",
        )
        assert session.session_id == "sess-1"
        assert session.conversation_id == "conv-1"
        assert session.workspace_root == "/path/to/project"
        assert session.workspace_name == "project"
        assert session.status == SessionStatus.IDLE
        assert session.messages == []
        assert session.loop_enabled is False

    def test_message_count(self):
        """Test message_count property."""
        session = AgentSession(
            session_id="s1",
            conversation_id="c1",
            workspace_root="/",
            workspace_name="test",
            messages=[
                SessionMessage(role="user", content="Hi"),
                SessionMessage(role="assistant", content="Hello"),
            ],
        )
        assert session.message_count == 2

    def test_last_message_preview_empty(self):
        """Test last_message_preview with no messages."""
        session = AgentSession(
            session_id="s1",
            conversation_id="c1",
            workspace_root="/",
            workspace_name="test",
        )
        assert session.last_message_preview is None

    def test_last_message_preview_long(self):
        """Test last_message_preview with long message (truncated)."""
        long_content = "x" * 150
        session = AgentSession(
            session_id="s1",
            conversation_id="c1",
            workspace_root="/",
            workspace_name="test",
            messages=[SessionMessage(role="user", content=long_content)],
        )
        preview = session.last_message_preview
        assert preview is not None
        assert len(preview) == 103  # 100 chars + "..."
        assert preview.endswith("...")

    def test_to_dict(self):
        """Test serialization to dictionary."""
        ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        session = AgentSession(
            session_id="s1",
            conversation_id="c1",
            workspace_root="/project",
            workspace_name="project",
            status=SessionStatus.ACTIVE,
            started_at=ts,
            last_activity=ts,
            current_task="Working",
            loop_enabled=True,
            loop_count=5,
            loop_prompt_name="default",
            loop_started_at=ts,
        )
        d = session.to_dict()
        assert d["session_id"] == "s1"
        assert d["conversation_id"] == "c1"
        assert d["status"] == "active"
        assert d["loop_enabled"] is True
        assert d["loop_count"] == 5
        assert d["loop_prompt_name"] == "default"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "session_id": "s1",
            "conversation_id": "c1",
            "workspace_root": "/project",
            "workspace_name": "project",
            "status": "idle",
            "started_at": "2024-01-15T12:00:00+00:00",
            "last_activity": "2024-01-15T12:00:00+00:00",
            "messages": [],
            "loop_enabled": False,
            "loop_count": 0,
        }
        session = AgentSession.from_dict(data)
        assert session.session_id == "s1"
        assert session.status == SessionStatus.IDLE
        assert session.loop_enabled is False

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = AgentSession(
            session_id="s1",
            conversation_id="c1",
            workspace_root="/project",
            workspace_name="project",
            status=SessionStatus.ACTIVE,
            loop_enabled=True,
            loop_count=3,
            messages=[SessionMessage(role="user", content="Test")],
        )
        restored = AgentSession.from_dict(original.to_dict())
        assert restored.session_id == original.session_id
        assert restored.status == original.status
        assert restored.loop_enabled == original.loop_enabled
        assert restored.loop_count == original.loop_count
        assert len(restored.messages) == 1

    def test_last_message_preview_short(self):
        """Test last_message_preview with short message."""
        session = AgentSession(
            session_id="s1",
            conversation_id="c1",
            workspace_root="/",
            workspace_name="test",
            messages=[SessionMessage(role="user", content="Short message")],
        )
        assert session.last_message_preview == "Short message"

