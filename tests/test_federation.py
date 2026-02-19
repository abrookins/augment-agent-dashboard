"""Tests for federation client and routes."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from augment_agent_dashboard.federation.client import (
    RemoteDashboardClient,
    _generate_federated_session_id,
    find_remote_by_hash,
    is_remote_session_id,
    parse_remote_session_id,
)
from augment_agent_dashboard.federation.models import RemoteDashboard
from augment_agent_dashboard.federation.routes import router
from augment_agent_dashboard.models import AgentSession, SessionStatus
from augment_agent_dashboard.store import SessionStore


class TestFederationClientHelpers:
    """Tests for federation client helper functions."""

    def test_generate_federated_session_id(self):
        """Test federated session ID generation."""
        session_id = _generate_federated_session_id("http://localhost:9000", "sess-123")
        assert session_id.startswith("remote-")
        assert "sess-123" in session_id

    def test_is_remote_session_id(self):
        """Test remote session ID detection."""
        assert is_remote_session_id("remote-abc12345-sess-123") is True
        assert is_remote_session_id("local-session-id") is False

    def test_parse_remote_session_id_valid(self):
        """Test parsing valid remote session ID."""
        result = parse_remote_session_id("remote-abc12345-original-id")
        assert result is not None
        assert result[0] == "abc12345"
        assert result[1] == "original-id"

    def test_parse_remote_session_id_invalid(self):
        """Test parsing invalid session IDs."""
        assert parse_remote_session_id("not-remote") is None
        assert parse_remote_session_id("remote-onlyonepart") is None

    def test_find_remote_by_hash(self):
        """Test finding remote by URL hash."""
        import hashlib
        remote = RemoteDashboard(name="test", url="http://localhost:9000")
        url_hash = hashlib.md5(remote.url.encode()).hexdigest()[:8]
        assert find_remote_by_hash([remote], url_hash) == remote
        assert find_remote_by_hash([remote], "wronghash") is None


class TestRemoteDashboardClient:
    """Tests for RemoteDashboardClient."""

    @pytest.fixture
    def remote(self):
        return RemoteDashboard(name="test-remote", url="http://localhost:9001", api_key="key")

    @pytest.fixture
    def client(self, remote):
        return RemoteDashboardClient(remote)

    def test_get_headers_with_api_key(self, client):
        headers = client._get_headers()
        assert headers["X-Dashboard-Api-Key"] == "key"

    def test_get_headers_without_api_key(self):
        remote = RemoteDashboard(name="test", url="http://localhost:9001")
        client = RemoteDashboardClient(remote)
        assert "X-Dashboard-Api-Key" not in client._get_headers()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.get")
    async def test_health_check_success(self, mock_get, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        assert await client.health_check() is True

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.get")
    async def test_health_check_failure(self, mock_get, client):
        mock_get.side_effect = Exception("error")
        assert await client.health_check() is False

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.get")
    async def test_fetch_sessions_success(self, mock_get, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"sessions": [{"session_id": "s1", "status": "active",
            "started_at": "", "last_activity": "", "message_count": 0}]}
        mock_get.return_value = mock_response
        sessions = await client.fetch_sessions()
        assert len(sessions) == 1

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.get")
    async def test_fetch_sessions_auth_failure(self, mock_get, client):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        assert await client.fetch_sessions() == []

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.get")
    async def test_fetch_sessions_timeout(self, mock_get, client):
        import httpx
        mock_get.side_effect = httpx.TimeoutException("timeout")
        assert await client.fetch_sessions() == []

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_send_message_success(self, mock_post, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        assert await client.send_message("sess-1", "Hello") is True

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_send_message_failure(self, mock_post, client):
        mock_post.side_effect = Exception("error")
        assert await client.send_message("sess-1", "Hello") is False

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.get")
    async def test_fetch_session_detail_success(self, mock_get, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"session_id": "s1"}
        mock_get.return_value = mock_response
        assert (await client.fetch_session_detail("s1")) is not None

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.get")
    async def test_fetch_session_detail_not_found(self, mock_get, client):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        assert (await client.fetch_session_detail("s1")) is None

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_create_session_success(self, mock_post, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_post.return_value = mock_response
        assert (await client.create_session("/ws", "prompt")) is not None

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.delete")
    async def test_delete_session_success(self, mock_delete, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_delete.return_value = mock_response
        assert await client.delete_session("s1") is True

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.delete")
    async def test_delete_session_error(self, mock_delete, client):
        mock_delete.side_effect = Exception("error")
        assert await client.delete_session("s1") is False


class TestFederationRoutes:
    """Tests for federation API routes."""

    @pytest.fixture
    def temp_store(self, tmp_path):
        sessions_file = tmp_path / "sessions.json"
        return SessionStore(sessions_file=sessions_file)

    @pytest.fixture
    def sample_session(self):
        return AgentSession(
            session_id="fed-test-1",
            conversation_id="conv-1",
            workspace_root="/path/to/project",
            workspace_name="project",
            status=SessionStatus.IDLE,
        )

    @pytest.fixture
    async def federation_client(self, temp_store, tmp_path, monkeypatch):
        from fastapi import FastAPI
        from augment_agent_dashboard.federation.routes import get_store

        app = FastAPI()
        app.include_router(router)

        # Use FastAPI's dependency override mechanism
        app.dependency_overrides[get_store] = lambda: temp_store

        config_path = tmp_path / ".augment" / "dashboard" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({"federation": {"share_locally": True}}))
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, temp_store

        # Clean up overrides
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_health_check(self, federation_client):
        ac, _ = federation_client
        response = await ac.get("/api/federation/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, federation_client):
        ac, _ = federation_client
        response = await ac.get("/api/federation/sessions")
        assert response.status_code == 200
        assert response.json()["sessions"] == []

    @pytest.mark.asyncio
    async def test_list_sessions_with_data(self, federation_client, sample_session):
        ac, store = federation_client
        store.upsert_session(sample_session)
        response = await ac.get("/api/federation/sessions")
        assert len(response.json()["sessions"]) == 1

    @pytest.mark.asyncio
    async def test_get_session(self, federation_client, sample_session):
        ac, store = federation_client
        store.upsert_session(sample_session)
        response = await ac.get(f"/api/federation/sessions/{sample_session.session_id}")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, federation_client):
        ac, _ = federation_client
        response = await ac.get("/api/federation/sessions/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_session(self, federation_client, sample_session):
        ac, store = federation_client
        store.upsert_session(sample_session)
        response = await ac.delete(f"/api/federation/sessions/{sample_session.session_id}")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self, federation_client):
        ac, _ = federation_client
        response = await ac.delete("/api/federation/sessions/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_send_message_to_session(self, federation_client, sample_session):
        """Test sending a message to a session."""
        ac, store = federation_client
        # Need a conversation_id and workspace_root for send_message to work
        sample_session.conversation_id = "conv-123"
        sample_session.workspace_root = "/tmp/project"
        store.upsert_session(sample_session)

        with patch("augment_agent_dashboard.server.spawn_auggie_message"):
            response = await ac.post(
                f"/api/federation/sessions/{sample_session.session_id}/message",
                json={"message": "Hello"}
            )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_send_message_session_not_found(self, federation_client):
        """Test sending message to non-existent session."""
        ac, _ = federation_client
        response = await ac.post(
            "/api/federation/sessions/nonexistent/message",
            json={"message": "Hello"}
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_send_message_no_conversation_id(self, federation_client, sample_session):
        """Test sending message to session without conversation_id."""
        ac, store = federation_client
        sample_session.conversation_id = "unknown"
        store.upsert_session(sample_session)
        response = await ac.post(
            f"/api/federation/sessions/{sample_session.session_id}/message",
            json={"message": "Hello"}
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_send_message_no_workspace(self, federation_client, sample_session):
        """Test sending message to session without workspace_root."""
        ac, store = federation_client
        sample_session.conversation_id = "conv-123"
        sample_session.workspace_root = ""
        store.upsert_session(sample_session)
        response = await ac.post(
            f"/api/federation/sessions/{sample_session.session_id}/message",
            json={"message": "Hello"}
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_create_session_success(self, federation_client, tmp_path):
        """Test creating a new session."""
        ac, _ = federation_client
        # Create a real directory for the workspace
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with patch("augment_agent_dashboard.server.spawn_new_session"):
            response = await ac.post(
                "/api/federation/sessions/new",
                json={"workspace_root": str(workspace), "prompt": "Do something"}
            )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_create_session_invalid_workspace(self, federation_client):
        """Test creating session with invalid workspace."""
        ac, _ = federation_client
        response = await ac.post(
            "/api/federation/sessions/new",
            json={"workspace_root": "/nonexistent/path/xyz123", "prompt": "Do something"}
        )
        assert response.status_code == 400


class TestVerifyApiKey:
    """Tests for API key verification."""

    @pytest.fixture
    def config_path(self, tmp_path, monkeypatch):
        """Set up a config path."""
        config_path = tmp_path / ".augment" / "dashboard" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        return config_path

    def test_verify_api_key_no_config(self, config_path):
        """Test when no config file exists - should allow access."""
        from augment_agent_dashboard.federation.routes import verify_api_key
        # No config file exists
        assert verify_api_key(None) is True

    def test_verify_api_key_sharing_disabled(self, config_path):
        """Test when sharing is disabled."""
        from augment_agent_dashboard.federation.routes import verify_api_key
        config_path.write_text(json.dumps({"federation": {"share_locally": False}}))
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(None)
        assert exc_info.value.status_code == 403

    def test_verify_api_key_required_missing(self, config_path):
        """Test when API key is required but not provided."""
        from augment_agent_dashboard.federation.routes import verify_api_key
        config_path.write_text(json.dumps({"federation": {"api_key": "secret123"}}))
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(None)
        assert exc_info.value.status_code == 401

    def test_verify_api_key_required_wrong(self, config_path):
        """Test when API key is required but wrong key provided."""
        from augment_agent_dashboard.federation.routes import verify_api_key
        config_path.write_text(json.dumps({"federation": {"api_key": "secret123"}}))
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key("wrongkey")
        assert exc_info.value.status_code == 401

    def test_verify_api_key_required_correct(self, config_path):
        """Test when correct API key is provided."""
        from augment_agent_dashboard.federation.routes import verify_api_key
        config_path.write_text(json.dumps({"federation": {"api_key": "secret123"}}))
        assert verify_api_key("secret123") is True

