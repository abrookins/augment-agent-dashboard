"""Tests for the FastAPI server."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

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


class TestDashboardSorting:
    """Tests for dashboard sorting."""

    @pytest.mark.asyncio
    async def test_dashboard_sort_by_name(self, client, sample_session):
        """Test sorting dashboard by name."""
        ac, store = client
        store.upsert_session(sample_session)
        response = await ac.get("/?sort=name")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_dashboard_dark_mode(self, client, sample_session):
        """Test dashboard with dark mode param."""
        ac, store = client
        store.upsert_session(sample_session)
        response = await ac.get("/?dark=true")
        assert response.status_code == 200


class TestPostMessage:
    """Tests for posting messages to sessions."""

    @pytest.mark.asyncio
    async def test_post_message_session_not_found(self, client):
        """Test posting message to non-existent session."""
        ac, store = client
        response = await ac.post(
            "/session/nonexistent/message",
            data={"message": "Hello"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_post_message_no_conversation_id(self, client):
        """Test posting message to session without conversation ID."""
        ac, store = client
        session = AgentSession(
            session_id="test-1",
            conversation_id="unknown",
            workspace_root="/test",
            workspace_name="test",
        )
        store.upsert_session(session)
        response = await ac.post(
            "/session/test-1/message",
            data={"message": "Hello"},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_post_message_no_workspace(self, client):
        """Test posting message to session without workspace root."""
        ac, store = client
        session = AgentSession(
            session_id="test-1",
            conversation_id="conv-123",
            workspace_root="",
            workspace_name="test",
        )
        store.upsert_session(session)
        response = await ac.post(
            "/session/test-1/message",
            data={"message": "Hello"},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_post_message_success(self, client, sample_session):
        """Test successful message posting."""
        ac, store = client
        sample_session.conversation_id = "conv-valid"
        sample_session.workspace_root = "/valid/workspace"
        store.upsert_session(sample_session)

        with patch("augment_agent_dashboard.server.spawn_auggie_message"):
            response = await ac.post(
                f"/session/{sample_session.session_id}/message",
                data={"message": "Hello agent"},
            )
        assert response.status_code == 303

    @pytest.mark.asyncio
    async def test_queue_message_empty(self, client, sample_session):
        """Test queuing an empty message returns redirect without adding."""
        ac, store = client
        store.upsert_session(sample_session)
        response = await ac.post(
            f"/session/{sample_session.session_id}/queue",
            data={"message": "   "},  # whitespace only
        )
        assert response.status_code == 303


class TestLoopNotFound:
    """Tests for loop endpoints with non-existent sessions."""

    @pytest.mark.asyncio
    async def test_enable_loop_not_found(self, client):
        """Test enabling loop for non-existent session."""
        ac, store = client
        response = await ac.post(
            "/session/nonexistent/loop/enable",
            data={"prompt_name": "default"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_pause_loop_not_found(self, client):
        """Test pausing loop for non-existent session."""
        ac, store = client
        response = await ac.post("/session/nonexistent/loop/pause")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_reset_loop_not_found(self, client):
        """Test resetting loop for non-existent session."""
        ac, store = client
        response = await ac.post("/session/nonexistent/loop/reset")
        assert response.status_code == 404


class TestConfigPrompts:
    """Tests for config prompt management endpoints."""

    @pytest.mark.asyncio
    async def test_add_prompt(self, client, tmp_path, monkeypatch):
        """Test adding a new prompt."""
        ac, store = client
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        (tmp_path / ".augment" / "dashboard").mkdir(parents=True)

        response = await ac.post(
            "/config/prompts/add",
            data={"name": "test_prompt", "prompt": "Test prompt text"},
        )
        assert response.status_code == 303

    @pytest.mark.asyncio
    async def test_delete_prompt(self, client, tmp_path, monkeypatch):
        """Test deleting a prompt."""
        ac, store = client
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        config_dir = tmp_path / ".augment" / "dashboard"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text('{"loop_prompts": {"test": "value"}}')

        response = await ac.post(
            "/config/prompts/delete",
            data={"name": "test"},
        )
        assert response.status_code == 303

    @pytest.mark.asyncio
    async def test_edit_prompt(self, client, tmp_path, monkeypatch):
        """Test editing a prompt."""
        ac, store = client
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        config_dir = tmp_path / ".augment" / "dashboard"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text('{"loop_prompts": {"test": "old"}}')

        response = await ac.post(
            "/config/prompts/edit",
            data={"name": "test", "prompt": "new value"},
        )
        assert response.status_code == 303


class TestClearQueueNotFound:
    """Tests for clear queue with non-existent session."""

    @pytest.mark.asyncio
    async def test_clear_queue_not_found(self, client):
        """Test clearing queue for non-existent session."""
        ac, store = client
        response = await ac.post("/session/nonexistent/queue/clear")
        assert response.status_code == 404


class TestSpawnAuggieMessage:
    """Tests for spawn_auggie_message function."""

    @pytest.mark.asyncio
    async def test_spawn_auggie_no_auggie_found(self):
        """Test when auggie is not in PATH."""
        from augment_agent_dashboard.server import spawn_auggie_message
        with patch("shutil.which", return_value=None):
            result = await spawn_auggie_message("conv-123", "/workspace", "test message")
            assert result is False

    @pytest.mark.asyncio
    async def test_spawn_auggie_success(self):
        """Test successful auggie spawn."""
        from augment_agent_dashboard.server import spawn_auggie_message
        with patch("shutil.which", return_value="/usr/local/bin/auggie"), \
             patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b"output", b""))
            mock_exec.return_value = mock_process

            result = await spawn_auggie_message("conv-123", "/workspace", "test message")
            assert result is True
            mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_spawn_auggie_failure(self):
        """Test when auggie returns non-zero."""
        from augment_agent_dashboard.server import spawn_auggie_message
        with patch("shutil.which", return_value="/usr/local/bin/auggie"), \
             patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = MagicMock()
            mock_process.returncode = 1
            mock_process.communicate = AsyncMock(return_value=(b"", b"error"))
            mock_exec.return_value = mock_process

            result = await spawn_auggie_message("conv-123", "/workspace", "test message")
            assert result is False

    @pytest.mark.asyncio
    async def test_spawn_auggie_exception(self):
        """Test when an exception occurs."""
        from augment_agent_dashboard.server import spawn_auggie_message
        with patch("shutil.which", return_value="/usr/local/bin/auggie"), \
             patch("asyncio.create_subprocess_exec", side_effect=Exception("Test error")):
            result = await spawn_auggie_message("conv-123", "/workspace", "test message")
            assert result is False


class TestGetLoopPrompts:
    """Tests for _get_loop_prompts function."""

    def test_get_loop_prompts_no_file(self, tmp_path, monkeypatch):
        """Test when config file doesn't exist."""
        from augment_agent_dashboard.server import _get_loop_prompts
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = _get_loop_prompts()
        assert isinstance(result, dict)

    def test_get_loop_prompts_with_file(self, tmp_path, monkeypatch):
        """Test when config file exists with new format."""
        from augment_agent_dashboard.server import _get_loop_prompts
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        config_dir = tmp_path / ".augment" / "dashboard"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text('{"loop_prompts": {"test": {"prompt": "test prompt", "end_condition": "DONE"}}}')

        result = _get_loop_prompts()
        assert "test" in result
        assert result["test"]["prompt"] == "test prompt"
        assert result["test"]["end_condition"] == "DONE"

    def test_get_loop_prompts_legacy_format(self, tmp_path, monkeypatch):
        """Test backward compatibility with old string-only format."""
        from augment_agent_dashboard.server import _get_loop_prompts
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        config_dir = tmp_path / ".augment" / "dashboard"
        config_dir.mkdir(parents=True)
        # Old format: prompts stored as plain strings
        (config_dir / "config.json").write_text('{"loop_prompts": {"test": "test prompt"}}')

        result = _get_loop_prompts()
        assert "test" in result
        # Should be normalized to new format with empty end_condition
        assert result["test"]["prompt"] == "test prompt"
        assert result["test"]["end_condition"] == ""

    def test_get_loop_prompts_invalid_json(self, tmp_path, monkeypatch):
        """Test when config file has invalid JSON."""
        from augment_agent_dashboard.server import _get_loop_prompts
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        config_dir = tmp_path / ".augment" / "dashboard"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text('not valid json')

        result = _get_loop_prompts()
        # Should return defaults on error
        assert isinstance(result, dict)


class TestNotificationOverflow:
    """Tests for notification queue overflow."""

    @pytest.mark.asyncio
    async def test_notification_queue_max_size(self, client):
        """Test that notification queue doesn't exceed 50."""
        ac, store = client
        # Send more than 50 notifications
        for i in range(55):
            response = await ac.post(
                "/api/notifications/send",
                data={"title": f"Test {i}", "body": "body", "url": ""},
            )
            assert response.status_code == 200


class TestNotificationPolling:
    """Tests for notification polling with timestamp."""

    @pytest.mark.asyncio
    async def test_poll_with_timestamp(self, client):
        """Test polling with a timestamp filter."""
        ac, store = client
        # Send a notification
        await ac.post(
            "/api/notifications/send",
            data={"title": "Test", "body": "body", "url": ""},
        )
        # Poll with a timestamp
        response = await ac.get("/api/notifications/poll?since=2020-01-01T00:00:00")
        assert response.status_code == 200
        data = response.json()
        assert "notifications" in data


class TestIconEndpoints:
    """Tests for icon endpoints."""

    @pytest.mark.asyncio
    async def test_icon_192(self, client):
        """Test 192x192 icon."""
        ac, store = client
        response = await ac.get("/icon-192.png")
        assert response.status_code == 200
        assert "svg" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_icon_512(self, client):
        """Test 512x512 icon."""
        ac, store = client
        response = await ac.get("/icon-512.png")
        assert response.status_code == 200
        assert "svg" in response.headers.get("content-type", "")


class TestFormatTimeAgo:
    """Tests for format_time_ago function."""

    def test_format_time_ago_just_now(self):
        """Test time less than a minute ago."""
        from datetime import datetime, timezone, timedelta
        from augment_agent_dashboard.server import format_time_ago

        now = datetime.now(timezone.utc)
        result = format_time_ago(now - timedelta(seconds=30))
        assert result == "just now"

    def test_format_time_ago_minutes(self):
        """Test time minutes ago."""
        from datetime import datetime, timezone, timedelta
        from augment_agent_dashboard.server import format_time_ago

        now = datetime.now(timezone.utc)
        result = format_time_ago(now - timedelta(minutes=5))
        assert "m ago" in result

    def test_format_time_ago_hours(self):
        """Test time hours ago."""
        from datetime import datetime, timezone, timedelta
        from augment_agent_dashboard.server import format_time_ago

        now = datetime.now(timezone.utc)
        result = format_time_ago(now - timedelta(hours=3))
        assert "h ago" in result

    def test_format_time_ago_days(self):
        """Test time days ago."""
        from datetime import datetime, timezone, timedelta
        from augment_agent_dashboard.server import format_time_ago

        now = datetime.now(timezone.utc)
        result = format_time_ago(now - timedelta(days=2))
        assert "d ago" in result

    def test_format_time_ago_naive_datetime(self):
        """Test with naive datetime (no timezone)."""
        from datetime import datetime, timedelta, timezone
        from augment_agent_dashboard.server import format_time_ago

        # Create a naive datetime that's 10 minutes before UTC now
        # The function adds UTC timezone to naive datetimes
        utc_now = datetime.now(timezone.utc)
        naive_dt = utc_now.replace(tzinfo=None) - timedelta(minutes=10)
        result = format_time_ago(naive_dt)
        assert "m ago" in result


class TestGetBaseStyles:
    """Tests for get_base_styles function."""

    def test_get_base_styles_dark(self):
        """Test dark mode styles."""
        from augment_agent_dashboard.server import get_base_styles

        result = get_base_styles("true")
        assert "1a1a2e" in result  # Dark background color

    def test_get_base_styles_light(self):
        """Test light mode styles."""
        from augment_agent_dashboard.server import get_base_styles

        result = get_base_styles("false")
        assert "ffffff" in result  # Light background color

    def test_get_base_styles_auto(self):
        """Test auto mode styles."""
        from augment_agent_dashboard.server import get_base_styles

        result = get_base_styles(None)
        assert "prefers-color-scheme" in result  # Should have media query


class TestGetFullConfig:
    """Tests for _get_full_config function."""

    def test_get_full_config_no_file(self, tmp_path, monkeypatch):
        """Test when config file doesn't exist."""
        from augment_agent_dashboard.server import _get_full_config
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = _get_full_config()
        assert result == {}

    def test_get_full_config_with_file(self, tmp_path, monkeypatch):
        """Test when config file exists."""
        from augment_agent_dashboard.server import _get_full_config
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        config_dir = tmp_path / ".augment" / "dashboard"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text('{"key": "value"}')

        result = _get_full_config()
        assert result == {"key": "value"}

    def test_get_full_config_invalid_json(self, tmp_path, monkeypatch):
        """Test when config file has invalid JSON."""
        from augment_agent_dashboard.server import _get_full_config
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        config_dir = tmp_path / ".augment" / "dashboard"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text('not valid json')

        result = _get_full_config()
        assert result == {}


class TestLoadLoopPrompts:
    """Tests for load_loop_prompts function."""

    def test_load_loop_prompts_no_file(self):
        """Test with no file specified."""
        from augment_agent_dashboard.server import load_loop_prompts, DEFAULT_LOOP_PROMPTS
        result = load_loop_prompts(None)
        assert result == DEFAULT_LOOP_PROMPTS

    def test_load_loop_prompts_with_file(self, tmp_path):
        """Test with valid file - legacy string format is normalized to dict."""
        from augment_agent_dashboard.server import load_loop_prompts
        prompts_file = tmp_path / "prompts.json"
        prompts_file.write_text('{"custom": "prompt"}')

        result = load_loop_prompts(str(prompts_file))
        # Legacy string format is normalized to new dict format with end_condition
        assert result == {"custom": {"prompt": "prompt", "end_condition": ""}}

    def test_load_loop_prompts_invalid_file(self, tmp_path):
        """Test with invalid file."""
        from augment_agent_dashboard.server import load_loop_prompts, DEFAULT_LOOP_PROMPTS
        prompts_file = tmp_path / "prompts.json"
        prompts_file.write_text('not valid json')

        result = load_loop_prompts(str(prompts_file))
        assert result == DEFAULT_LOOP_PROMPTS

    def test_load_loop_prompts_missing_file(self):
        """Test with missing file."""
        from augment_agent_dashboard.server import load_loop_prompts, DEFAULT_LOOP_PROMPTS
        result = load_loop_prompts("/nonexistent/file.json")
        assert result == DEFAULT_LOOP_PROMPTS


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_config(self, tmp_path, monkeypatch):
        """Test saving config."""
        import json
        from augment_agent_dashboard.server import save_config
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        save_config(9000, True, {"test": "prompt"}, 100)

        config_path = tmp_path / ".augment" / "dashboard" / "config.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert config["port"] == 9000
        assert config["notification_sound"] is True
        assert config["loop_prompts"] == {"test": "prompt"}
        assert config["max_loop_iterations"] == 100


class TestFormatElapsedTime:
    """Tests for _format_elapsed_time function."""

    def test_format_elapsed_time_none(self):
        """Test with None input."""
        from augment_agent_dashboard.server import _format_elapsed_time
        result = _format_elapsed_time(None)
        assert result == ""

    def test_format_elapsed_time_seconds(self):
        """Test with seconds only."""
        from datetime import datetime, timezone, timedelta
        from augment_agent_dashboard.server import _format_elapsed_time

        started = datetime.now(timezone.utc) - timedelta(seconds=30)
        result = _format_elapsed_time(started)
        assert "s" in result

    def test_format_elapsed_time_minutes(self):
        """Test with minutes."""
        from datetime import datetime, timezone, timedelta
        from augment_agent_dashboard.server import _format_elapsed_time

        started = datetime.now(timezone.utc) - timedelta(minutes=5, seconds=30)
        result = _format_elapsed_time(started)
        assert "m" in result and "s" in result

    def test_format_elapsed_time_hours(self):
        """Test with hours."""
        from datetime import datetime, timezone, timedelta
        from augment_agent_dashboard.server import _format_elapsed_time

        started = datetime.now(timezone.utc) - timedelta(hours=2, minutes=30)
        result = _format_elapsed_time(started)
        assert "h" in result and "m" in result

    def test_format_elapsed_time_naive_datetime(self):
        """Test with naive datetime."""
        from datetime import datetime, timezone, timedelta
        from augment_agent_dashboard.server import _format_elapsed_time

        utc_now = datetime.now(timezone.utc)
        started = utc_now.replace(tzinfo=None) - timedelta(minutes=5)
        result = _format_elapsed_time(started)
        assert "m" in result


class TestRenderMessageForm:
    """Tests for _render_message_form function."""

    def test_render_message_form_active(self, sample_session):
        """Test rendering form for active session."""
        from augment_agent_dashboard.server import _render_message_form
        from augment_agent_dashboard.models import SessionStatus

        sample_session.status = SessionStatus.ACTIVE
        result = _render_message_form(sample_session)
        assert "queue" in result.lower()

    def test_render_message_form_idle(self, sample_session):
        """Test rendering form for idle session."""
        from augment_agent_dashboard.server import _render_message_form
        from augment_agent_dashboard.models import SessionStatus

        sample_session.status = SessionStatus.IDLE
        result = _render_message_form(sample_session)
        assert "message" in result.lower()

    def test_render_message_form_with_queued(self, sample_session):
        """Test rendering form with queued messages."""
        from augment_agent_dashboard.server import _render_message_form
        from augment_agent_dashboard.models import SessionStatus, SessionMessage

        sample_session.status = SessionStatus.ACTIVE
        sample_session.messages = [
            SessionMessage(role="queued", content="test1"),
            SessionMessage(role="queued", content="test2"),
        ]
        result = _render_message_form(sample_session)
        assert "2 queued" in result


class TestRenderSessionDetail:
    """Tests for render_session_detail function."""

    def test_render_session_detail_empty_messages(self, sample_session):
        """Test rendering session with empty messages."""
        from augment_agent_dashboard.server import render_session_detail

        sample_session.messages = []
        result = render_session_detail(sample_session, None, {})
        assert "No messages" in result

    def test_render_session_detail_with_messages(self, sample_session):
        """Test rendering session with messages."""
        from augment_agent_dashboard.server import render_session_detail
        from augment_agent_dashboard.models import SessionMessage

        sample_session.messages = [
            SessionMessage(role="user", content="Hello"),
            SessionMessage(role="assistant", content="Hi there"),
        ]
        result = render_session_detail(sample_session, None, {})
        assert "Hello" in result
        assert "Hi there" in result

    def test_render_session_detail_with_queued(self, sample_session):
        """Test rendering session with queued messages."""
        from augment_agent_dashboard.server import render_session_detail
        from augment_agent_dashboard.models import SessionMessage

        sample_session.messages = [
            SessionMessage(role="queued", content="Queued message"),
        ]
        result = render_session_detail(sample_session, None, {})
        assert "Queued" in result
        assert "Clear Queue" in result


class TestGetStore:
    """Tests for get_store function."""

    def test_get_store(self):
        """Test get_store returns a SessionStore."""
        from augment_agent_dashboard.server import get_store
        from augment_agent_dashboard.store import SessionStore

        store = get_store()
        assert isinstance(store, SessionStore)


class TestRenderLoopControls:
    """Tests for _render_loop_controls function."""

    def test_render_loop_controls_enabled(self, sample_session):
        """Test rendering loop controls when enabled."""
        from datetime import datetime, timezone
        from augment_agent_dashboard.server import _render_loop_controls

        sample_session.loop_enabled = True
        sample_session.loop_count = 5
        sample_session.loop_prompt_name = "Test Prompt"
        sample_session.loop_started_at = datetime.now(timezone.utc)

        result = _render_loop_controls(sample_session, {"Test Prompt": "prompt text"})
        assert "Test Prompt" in result
        assert "5 iterations" in result

    def test_render_loop_controls_disabled(self, sample_session):
        """Test rendering loop controls when disabled."""
        from augment_agent_dashboard.server import _render_loop_controls

        sample_session.loop_enabled = False
        result = _render_loop_controls(sample_session, {"Default": "prompt"})
        assert "Loop Paused" in result or "loop-controls" in result


class TestMain:
    """Tests for main entry point."""

    def test_main_default_args(self, tmp_path, monkeypatch):
        """Test main with default arguments."""
        import sys
        from augment_agent_dashboard import server

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setattr(sys, "argv", ["augment-dashboard"])

        # Mock uvicorn.run to prevent actually starting the server
        uvicorn_called = {}
        def mock_uvicorn_run(app, host, port):
            uvicorn_called["app"] = app
            uvicorn_called["host"] = host
            uvicorn_called["port"] = port

        with patch("uvicorn.run", mock_uvicorn_run):
            server.main()

        assert uvicorn_called["port"] == 8080
        assert uvicorn_called["host"] == "0.0.0.0"

    def test_main_custom_port(self, tmp_path, monkeypatch):
        """Test main with custom port."""
        import sys
        from augment_agent_dashboard import server

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setattr(sys, "argv", ["augment-dashboard", "--port", "9000"])

        uvicorn_called = {}
        def mock_uvicorn_run(app, host, port):
            uvicorn_called["port"] = port

        with patch("uvicorn.run", mock_uvicorn_run):
            server.main()

        assert uvicorn_called["port"] == 9000

    def test_main_no_sound(self, tmp_path, monkeypatch):
        """Test main with --no-sound flag."""
        import sys
        import json
        from augment_agent_dashboard import server

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setattr(sys, "argv", ["augment-dashboard", "--no-sound"])

        with patch("uvicorn.run"):
            server.main()

        # Check config was saved with sound disabled
        config_path = tmp_path / ".augment" / "dashboard" / "config.json"
        config = json.loads(config_path.read_text())
        assert config["notification_sound"] is False

    def test_main_with_loop_prompts_file(self, tmp_path, monkeypatch):
        """Test main with custom loop prompts file."""
        import sys
        import json
        from augment_agent_dashboard import server

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Create custom prompts file
        prompts_file = tmp_path / "prompts.json"
        prompts_file.write_text('{"Custom": "custom prompt"}')

        monkeypatch.setattr(sys, "argv", ["augment-dashboard", "--loop-prompts-file", str(prompts_file)])

        with patch("uvicorn.run"):
            server.main()

        # Check config was saved with custom prompts
        config_path = tmp_path / ".augment" / "dashboard" / "config.json"
        config = json.loads(config_path.read_text())
        assert "Custom" in config["loop_prompts"]

    def test_main_max_loop_iterations(self, tmp_path, monkeypatch):
        """Test main with custom max loop iterations."""
        import sys
        import json
        from augment_agent_dashboard import server

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setattr(sys, "argv", ["augment-dashboard", "--max-loop-iterations", "100"])

        with patch("uvicorn.run"):
            server.main()

        config_path = tmp_path / ".augment" / "dashboard" / "config.json"
        config = json.loads(config_path.read_text())
        assert config["max_loop_iterations"] == 100

    def test_server_as_main(self, tmp_path, monkeypatch):
        """Test running server as __main__."""
        import runpy
        import sys

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setattr(sys, "argv", ["augment-dashboard"])

        with patch("uvicorn.run"):
            runpy.run_module("augment_agent_dashboard.server", run_name="__main__")


class TestStaticFileMount:
    """Tests for static file mounting."""

    def test_static_mount_when_dir_exists(self):
        """Test that static files are mounted when directory exists."""
        from pathlib import Path

        from augment_agent_dashboard import server
        static_dir = Path(server.__file__).parent / "static"

        # The static directory should exist (created with .gitkeep)
        assert static_dir.exists(), "Static directory should exist"

        # Create a test file
        test_file = static_dir / "test.txt"
        test_file.write_text("test content")
        try:
            # Reload the module to trigger the static mount
            import importlib
            importlib.reload(server)

            from fastapi.testclient import TestClient
            client = TestClient(server.app)

            # Try to access the static file
            response = client.get("/static/test.txt")
            assert response.status_code == 200
            assert response.text == "test content"
        finally:
            # Clean up
            if test_file.exists():
                test_file.unlink()
