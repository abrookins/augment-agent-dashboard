"""Tests for the FastAPI server."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from augment_agent_dashboard.models import AgentSession, SessionMessage, SessionStatus
from augment_agent_dashboard.server import app
from augment_agent_dashboard.store import SessionStore


@pytest.fixture
def temp_store():
    """Create a temporary session store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sessions_file = Path(tmpdir) / "sessions.json"
        store = SessionStore(sessions_file=sessions_file)
        yield store


@pytest.fixture
def sample_session():
    """Create a sample session."""
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
async def client(temp_store):
    """Create an async test client with mocked store."""
    with patch("augment_agent_dashboard.server.get_store", return_value=temp_store):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, temp_store


class TestAPIEndpoints:
    """Tests for API endpoints."""

    @pytest.mark.asyncio
    async def test_get_sessions_empty(self, client):
        """Test getting sessions when empty."""
        ac, store = client
        response = await ac.get("/api/sessions")
        assert response.status_code == 200
        assert response.json()["sessions"] == []

    @pytest.mark.asyncio
    async def test_get_sessions_with_data(self, client, sample_session):
        """Test getting sessions with data."""
        ac, store = client
        store.upsert_session(sample_session)
        response = await ac.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["session_id"] == sample_session.session_id

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, client):
        """Test getting a non-existent session."""
        ac, store = client
        response = await ac.get("/api/sessions/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_session_found(self, client, sample_session):
        """Test getting an existing session."""
        ac, store = client
        store.upsert_session(sample_session)
        response = await ac.get(f"/api/sessions/{sample_session.session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == sample_session.session_id
        assert len(data["messages"]) == 2

    @pytest.mark.asyncio
    async def test_delete_session(self, client, sample_session):
        """Test deleting a session."""
        ac, store = client
        store.upsert_session(sample_session)
        response = await ac.post(f"/session/{sample_session.session_id}/delete")
        assert response.status_code == 303  # Redirect after delete
        # Verify it's gone
        assert store.get_session(sample_session.session_id) is None

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self, client):
        """Test deleting a non-existent session."""
        ac, store = client
        response = await ac.post("/session/nonexistent/delete")
        assert response.status_code == 404


class TestHTMLPages:
    """Tests for HTML page endpoints."""

    @pytest.mark.asyncio
    async def test_index_page(self, client):
        """Test the index page loads."""
        ac, store = client
        response = await ac.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Agent Dashboard" in response.text

    @pytest.mark.asyncio
    async def test_session_page_not_found(self, client):
        """Test session page for non-existent session."""
        ac, store = client
        response = await ac.get("/session/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_session_page_found(self, client, sample_session):
        """Test session page for existing session."""
        ac, store = client
        store.upsert_session(sample_session)
        response = await ac.get(f"/session/{sample_session.session_id}")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    @pytest.mark.asyncio
    async def test_config_page(self, client):
        """Test the config page loads."""
        ac, store = client
        response = await ac.get("/config")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestNotifications:
    """Tests for notification endpoints."""

    @pytest.mark.asyncio
    async def test_send_notification(self, client):
        """Test sending a browser notification."""
        ac, store = client
        response = await ac.post(
            "/api/notifications/send",
            data={"title": "Test", "body": "Test message", "url": "http://test"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "queued"

    @pytest.mark.asyncio
    async def test_poll_notifications_empty(self, client):
        """Test polling notifications when empty."""
        ac, store = client
        response = await ac.get("/api/notifications/poll")
        assert response.status_code == 200
        assert response.json()["notifications"] == []

    @pytest.mark.asyncio
    async def test_manifest_json(self, client):
        """Test the manifest.json endpoint."""
        ac, store = client
        response = await ac.get("/manifest.json")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data

    @pytest.mark.asyncio
    async def test_service_worker(self, client):
        """Test the service worker endpoint."""
        ac, store = client
        response = await ac.get("/sw.js")
        assert response.status_code == 200
        assert "javascript" in response.headers["content-type"]


class TestLoopEndpoints:
    """Tests for quality loop endpoints."""

    @pytest.mark.asyncio
    async def test_enable_loop(self, client, sample_session):
        """Test enabling the quality loop."""
        ac, store = client
        store.upsert_session(sample_session)
        response = await ac.post(
            f"/session/{sample_session.session_id}/loop/enable",
            data={"prompt_name": "default"},
        )
        assert response.status_code == 303  # Redirect after enable
        # Verify loop is enabled
        session = store.get_session(sample_session.session_id)
        assert session is not None
        assert session.loop_enabled is True

    @pytest.mark.asyncio
    async def test_disable_loop(self, client, sample_session):
        """Test disabling (pausing) the quality loop."""
        ac, store = client
        sample_session.loop_enabled = True
        store.upsert_session(sample_session)
        response = await ac.post(
            f"/session/{sample_session.session_id}/loop/pause",
        )
        assert response.status_code == 303  # Redirect after pause
        session = store.get_session(sample_session.session_id)
        assert session is not None
        assert session.loop_enabled is False


class TestMessageEndpoints:
    """Tests for message queue endpoints."""

    @pytest.mark.asyncio
    async def test_queue_message(self, client, sample_session):
        """Test queuing a message for a session."""
        ac, store = client
        store.upsert_session(sample_session)
        response = await ac.post(
            f"/session/{sample_session.session_id}/queue",
            data={"message": "Test queued message"},
        )
        assert response.status_code == 303  # Redirect

    @pytest.mark.asyncio
    async def test_queue_message_not_found(self, client):
        """Test queuing a message for non-existent session."""
        ac, store = client
        response = await ac.post(
            "/session/nonexistent/queue",
            data={"message": "Test message"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_clear_queue(self, client, sample_session):
        """Test clearing queued messages."""
        ac, store = client
        store.upsert_session(sample_session)
        response = await ac.post(f"/session/{sample_session.session_id}/queue/clear")
        assert response.status_code == 303  # Redirect


class TestLoopResetEndpoint:
    """Tests for loop reset endpoint."""

    @pytest.mark.asyncio
    async def test_reset_loop(self, client, sample_session):
        """Test resetting the loop counter."""
        ac, store = client
        sample_session.loop_enabled = True
        sample_session.loop_count = 5
        store.upsert_session(sample_session)
        response = await ac.post(f"/session/{sample_session.session_id}/loop/reset")
        assert response.status_code == 303  # Redirect
        session = store.get_session(sample_session.session_id)
        assert session is not None
        assert session.loop_count == 0


class TestAPIFiltering:
    """Tests for API filtering and pagination."""

    @pytest.mark.asyncio
    async def test_get_sessions_with_status_filter(self, client, sample_session):
        """Test filtering sessions by status."""
        ac, store = client
        sample_session.status = SessionStatus.ACTIVE
        store.upsert_session(sample_session)
        response = await ac.get("/api/sessions?status=active")
        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) == 1

    @pytest.mark.asyncio
    async def test_get_sessions_with_invalid_status(self, client, sample_session):
        """Test filtering with invalid status returns empty."""
        ac, store = client
        store.upsert_session(sample_session)
        response = await ac.get("/api/sessions?status=invalid")
        assert response.status_code == 200
        # Invalid status should be ignored, returning all sessions
        data = response.json()
        assert len(data["sessions"]) == 1
