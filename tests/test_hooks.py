"""Tests for Augment dashboard hooks."""

import io
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from augment_agent_dashboard.hooks import session_start, stop, tool_use
from augment_agent_dashboard.models import AgentSession, SessionMessage, SessionStatus
from augment_agent_dashboard.store import SessionStore


class TestSessionStartHelpers:
    """Tests for session_start helper functions."""

    def test_get_workspace_root_empty(self):
        assert session_start.get_workspace_root([]) is None

    def test_get_workspace_root_with_roots(self):
        assert session_start.get_workspace_root(["/path/to/project", "/other"]) == "/path/to/project"

    def test_get_workspace_name_none(self):
        assert session_start.get_workspace_name(None) == "unknown"

    def test_get_workspace_name_with_path(self):
        assert session_start.get_workspace_name("/path/to/myproject") == "myproject"

    def test_get_session_id(self):
        assert session_start.get_session_id("abc-123") == "abc-123"


class TestStopHelpers:
    """Tests for stop hook helper functions."""

    def test_get_workspace_root_empty(self):
        assert stop.get_workspace_root([]) is None

    def test_get_workspace_root_with_roots(self):
        assert stop.get_workspace_root(["/path/to/project"]) == "/path/to/project"

    def test_get_workspace_name_none(self):
        assert stop.get_workspace_name(None) == "unknown"

    def test_get_workspace_name_with_path(self):
        assert stop.get_workspace_name("/home/user/code") == "code"

    def test_get_session_id(self):
        assert stop.get_session_id("xyz-789") == "xyz-789"

    def test_load_config_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(stop, "CONFIG_PATH", tmp_path / "nonexistent.json")
        assert stop.load_config() == {}

    def test_load_config_with_file(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.json"
        config_path.write_text('{"port": 8080}')
        monkeypatch.setattr(stop, "CONFIG_PATH", config_path)
        assert stop.load_config() == {"port": 8080}

    def test_load_config_invalid_json(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.json"
        config_path.write_text('invalid json')
        monkeypatch.setattr(stop, "CONFIG_PATH", config_path)
        assert stop.load_config() == {}


class TestToolUseHelpers:
    """Tests for tool_use hook helper functions."""

    def test_get_workspace_root_empty(self):
        assert tool_use.get_workspace_root([]) is None

    def test_get_workspace_root_with_roots(self):
        assert tool_use.get_workspace_root(["/some/path"]) == "/some/path"

    def test_get_workspace_name_none(self):
        assert tool_use.get_workspace_name(None) == "unknown"

    def test_get_workspace_name_with_path(self):
        assert tool_use.get_workspace_name("/projects/myapp") == "myapp"

    def test_get_session_id(self):
        assert tool_use.get_session_id("session-456") == "session-456"


class TestSessionStartHook:
    """Tests for session_start.run_hook."""

    @patch("augment_agent_dashboard.hooks.session_start.SessionStore")
    @patch("sys.stdin", new_callable=io.StringIO)
    @patch("psutil.Process")
    def test_run_hook_new_session(self, mock_process, mock_stdin, mock_store_class, tmp_path, monkeypatch, capsys):
        # Setup mocks
        mock_store = MagicMock()
        mock_store.get_session.return_value = None
        mock_store_class.return_value = mock_store

        # Mock psutil.Process
        mock_parent = MagicMock()
        mock_parent.name.return_value = "python"
        mock_parent.cwd.return_value = "/test"
        mock_process.return_value = mock_parent

        # Set up log path
        log_dir = tmp_path / ".augment" / "dashboard"
        log_dir.mkdir(parents=True)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Provide input
        mock_stdin.write(json.dumps({
            "workspace_roots": ["/path/to/project"],
            "conversation_id": "conv-123"
        }))
        mock_stdin.seek(0)

        session_start.run_hook()

        # Verify session was created
        mock_store.upsert_session.assert_called_once()
        session = mock_store.upsert_session.call_args[0][0]
        assert session.session_id == "conv-123"
        assert session.workspace_name == "project"

        captured = capsys.readouterr()
        assert "{}" in captured.out  # Empty JSON output


class TestToolUseHook:
    """Tests for tool_use.run_hook."""

    @patch("augment_agent_dashboard.hooks.tool_use.SessionStore")
    @patch("sys.stdin", new_callable=io.StringIO)
    def test_run_hook_post_tool_use(self, mock_stdin, mock_store_class):
        mock_store = MagicMock()
        mock_session = AgentSession(
            session_id="sess-1",
            conversation_id="conv-1",
            workspace_root="/test",
            workspace_name="test",
        )
        mock_store.get_session.return_value = mock_session
        mock_store_class.return_value = mock_store

        mock_stdin.write(json.dumps({
            "conversation_id": "sess-1",
            "toolUse": {"name": "view", "input": {"path": "file.py"}}
        }))
        mock_stdin.seek(0)

        tool_use.run_hook("PostToolUse")

        # Verify tool was added and message was created
        mock_store.add_message.assert_called_once()
        mock_store.upsert_session.assert_called_once()


class TestStopHook:
    """Tests for stop.run_hook."""

    @patch("augment_agent_dashboard.hooks.stop.send_notification")
    @patch("augment_agent_dashboard.hooks.stop.load_config")
    @patch("augment_agent_dashboard.hooks.stop.SessionStore")
    @patch("sys.stdin", new_callable=io.StringIO)
    def test_run_hook_new_session(self, mock_stdin, mock_store_class, mock_config, mock_notify, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        (tmp_path / ".augment" / "dashboard").mkdir(parents=True)

        mock_store = MagicMock()
        mock_store.get_session.return_value = None
        mock_store_class.return_value = mock_store
        mock_config.return_value = {"port": 9000}

        mock_stdin.write(json.dumps({
            "workspace_roots": ["/path/to/project"],
            "conversation_id": "conv-123",
            "conversation": {
                "userPrompt": "Hello",
                "agentTextResponse": "Hi there!"
            }
        }))
        mock_stdin.seek(0)

        stop.run_hook()

        # Verify session was created
        mock_store.upsert_session.assert_called()
        mock_notify.assert_called()

        captured = capsys.readouterr()
        assert "{}" in captured.out

    @patch("augment_agent_dashboard.hooks.stop.send_notification")
    @patch("augment_agent_dashboard.hooks.stop.load_config")
    @patch("augment_agent_dashboard.hooks.stop.SessionStore")
    @patch("sys.stdin", new_callable=io.StringIO)
    def test_run_hook_existing_session(self, mock_stdin, mock_store_class, mock_config, mock_notify, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        (tmp_path / ".augment" / "dashboard").mkdir(parents=True)

        mock_store = MagicMock()
        existing_session = AgentSession(
            session_id="conv-123",
            conversation_id="conv-123",
            workspace_root="/path/to/project",
            workspace_name="project",
        )
        mock_store.get_session.return_value = existing_session
        mock_store_class.return_value = mock_store
        mock_config.return_value = {}

        mock_stdin.write(json.dumps({
            "workspace_roots": ["/path/to/project"],
            "conversation_id": "conv-123",
            "conversation": {
                "userPrompt": "New message",
                "agentTextResponse": "Response"
            }
        }))
        mock_stdin.seek(0)

        stop.run_hook()

        # Verify messages were added
        assert mock_store.add_message.call_count == 2  # user + assistant
        mock_store.update_session_status.assert_called()

