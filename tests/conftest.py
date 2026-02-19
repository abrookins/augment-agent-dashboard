"""Shared pytest fixtures and configuration."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from augment_agent_dashboard.models import AgentSession, SessionMessage, SessionStatus
from augment_agent_dashboard.store import SessionStore


# Note: We don't use autouse for disabling notifications because some tests
# specifically test the notification functions. Tests that call run_hook()
# already mock send_notification via @patch decorators.


# Note: We don't use an autouse fixture for isolating session store
# because many tests already set up their own tmp_path with monkeypatch.
# The key is that tests use the temp_store fixture or their own mocks.


@pytest.fixture
def temp_store(tmp_path):
    """Create a temporary session store for tests that need explicit access."""
    sessions_file = tmp_path / "sessions.json"
    store = SessionStore(sessions_file=sessions_file)
    yield store


@pytest.fixture
def sample_session():
    """Create a sample session for testing."""
    return AgentSession(
        session_id="test-session-1",
        conversation_id="conv-1",
        workspace_root="/path/to/project",
        workspace_name="project",
        status=SessionStatus.IDLE,
        messages=[
            SessionMessage(role="user", content="Hello"),
            SessionMessage(role="assistant", content="Hi there!"),
        ],
    )


@pytest.fixture
def sample_active_session():
    """Create an active session for testing."""
    return AgentSession(
        session_id="test-session-active",
        conversation_id="conv-active",
        workspace_root="/path/to/project",
        workspace_name="project",
        status=SessionStatus.ACTIVE,
        messages=[
            SessionMessage(role="user", content="Do something"),
        ],
    )


@pytest.fixture
def sample_session_with_queued():
    """Create a session with queued messages."""
    return AgentSession(
        session_id="test-session-queued",
        conversation_id="conv-queued",
        workspace_root="/path/to/project",
        workspace_name="project",
        status=SessionStatus.IDLE,
        messages=[
            SessionMessage(role="user", content="First message"),
            SessionMessage(role="queued", content="Queued message 1"),
            SessionMessage(role="queued", content="Queued message 2"),
        ],
    )

