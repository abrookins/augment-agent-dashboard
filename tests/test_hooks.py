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

    @patch("augment_agent_dashboard.hooks.session_start.SessionStore")
    @patch("sys.stdin", new_callable=io.StringIO)
    @patch("psutil.Process")
    def test_run_hook_existing_session_with_messages(self, mock_process, mock_stdin, mock_store_class, tmp_path, monkeypatch, capsys):
        """Test that existing session with dashboard messages outputs them."""
        mock_store = MagicMock()
        existing_session = AgentSession(
            session_id="conv-123",
            conversation_id="conv-123",
            workspace_root="/path/to/project",
            workspace_name="project",
        )
        mock_store.get_session.return_value = existing_session
        mock_store.get_and_clear_dashboard_messages.return_value = ["Message 1", "Message 2"]
        mock_store_class.return_value = mock_store

        mock_parent = MagicMock()
        mock_parent.name.return_value = "python"
        mock_parent.cwd.return_value = "/test"
        mock_process.return_value = mock_parent

        log_dir = tmp_path / ".augment" / "dashboard"
        log_dir.mkdir(parents=True)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        mock_stdin.write(json.dumps({
            "workspace_roots": ["/path/to/project"],
            "conversation_id": "conv-123"
        }))
        mock_stdin.seek(0)

        session_start.run_hook()

        captured = capsys.readouterr()
        assert "Messages from Dashboard" in captured.out
        assert "Message 1" in captured.out

    @patch("augment_agent_dashboard.hooks.session_start.SessionStore")
    @patch("sys.stdin", new_callable=io.StringIO)
    @patch("psutil.Process")
    def test_run_hook_no_json_input(self, mock_process, mock_stdin, mock_store_class, tmp_path, monkeypatch, capsys):
        """Test handling of invalid JSON input."""
        mock_store = MagicMock()
        mock_store.get_session.return_value = None
        mock_store_class.return_value = mock_store

        mock_parent = MagicMock()
        mock_parent.name.return_value = "python"
        mock_parent.cwd.return_value = "/test"
        mock_process.return_value = mock_parent

        log_dir = tmp_path / ".augment" / "dashboard"
        log_dir.mkdir(parents=True)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        mock_stdin.write("not valid json")
        mock_stdin.seek(0)

        session_start.run_hook()

        captured = capsys.readouterr()
        assert "{}" in captured.out

    @patch("augment_agent_dashboard.hooks.session_start.SessionStore")
    @patch("sys.stdin", new_callable=io.StringIO)
    @patch("psutil.Process")
    def test_run_hook_parent_process_error(self, mock_process, mock_stdin, mock_store_class, tmp_path, monkeypatch, capsys):
        """Test handling of parent process error."""
        mock_store = MagicMock()
        mock_store.get_session.return_value = None
        mock_store_class.return_value = mock_store

        mock_process.side_effect = Exception("No such process")

        log_dir = tmp_path / ".augment" / "dashboard"
        log_dir.mkdir(parents=True)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        mock_stdin.write(json.dumps({
            "workspace_roots": ["/path/to/project"],
            "conversation_id": "conv-123"
        }))
        mock_stdin.seek(0)

        session_start.run_hook()

        captured = capsys.readouterr()
        assert "{}" in captured.out

    @patch("augment_agent_dashboard.hooks.session_start.SessionStore")
    @patch("sys.stdin", new_callable=io.StringIO)
    @patch("psutil.Process")
    def test_run_hook_store_error(self, mock_process, mock_stdin, mock_store_class, tmp_path, monkeypatch, capsys):
        """Test handling of store errors."""
        mock_store = MagicMock()
        mock_store.get_session.side_effect = Exception("Store error")
        mock_store_class.return_value = mock_store

        mock_parent = MagicMock()
        mock_parent.name.return_value = "python"
        mock_parent.cwd.return_value = "/test"
        mock_process.return_value = mock_parent

        log_dir = tmp_path / ".augment" / "dashboard"
        log_dir.mkdir(parents=True)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        mock_stdin.write(json.dumps({
            "workspace_roots": ["/path/to/project"],
            "conversation_id": "conv-123"
        }))
        mock_stdin.seek(0)

        session_start.run_hook()

        captured = capsys.readouterr()
        assert "{}" in captured.out


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

    @patch("augment_agent_dashboard.hooks.tool_use.SessionStore")
    @patch("sys.stdin", new_callable=io.StringIO)
    def test_run_hook_invalid_json(self, mock_stdin, mock_store_class):
        """Test handling of invalid JSON input."""
        mock_store = MagicMock()
        mock_store.get_session.return_value = None
        mock_store_class.return_value = mock_store

        mock_stdin.write("not valid json")
        mock_stdin.seek(0)

        tool_use.run_hook("PostToolUse")
        # Should complete without error

    @patch("augment_agent_dashboard.hooks.tool_use.SessionStore")
    @patch("sys.stdin", new_callable=io.StringIO)
    def test_run_hook_long_input(self, mock_stdin, mock_store_class):
        """Test that long input is truncated."""
        mock_store = MagicMock()
        mock_session = AgentSession(
            session_id="sess-1",
            conversation_id="conv-1",
            workspace_root="/test",
            workspace_name="test",
        )
        mock_store.get_session.return_value = mock_session
        mock_store_class.return_value = mock_store

        # Create input with very long content
        long_content = "x" * 500
        mock_stdin.write(json.dumps({
            "conversation_id": "sess-1",
            "toolUse": {"name": "view", "input": {"path": long_content}}
        }))
        mock_stdin.seek(0)

        tool_use.run_hook("PostToolUse")

        mock_store.add_message.assert_called_once()

    @patch("augment_agent_dashboard.hooks.tool_use.SessionStore")
    @patch("sys.stdin", new_callable=io.StringIO)
    def test_run_hook_store_error(self, mock_stdin, mock_store_class, capsys):
        """Test handling of store errors."""
        mock_store = MagicMock()
        mock_store.get_session.side_effect = Exception("Store error")
        mock_store_class.return_value = mock_store

        mock_stdin.write(json.dumps({
            "conversation_id": "sess-1",
            "toolUse": {"name": "view", "input": {}}
        }))
        mock_stdin.seek(0)

        tool_use.run_hook("PostToolUse")

        captured = capsys.readouterr()
        assert "Dashboard tool hook error" in captured.err

    @patch("augment_agent_dashboard.hooks.tool_use.SessionStore")
    @patch("sys.stdin", new_callable=io.StringIO)
    def test_run_pre_tool_use(self, mock_stdin, mock_store_class):
        """Test PreToolUse hook."""
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
            "toolUse": {"name": "view", "input": {}}
        }))
        mock_stdin.seek(0)

        tool_use.run_pre_tool_use()
        # PreToolUse doesn't add messages, just updates tools_used

    @patch("augment_agent_dashboard.hooks.tool_use.SessionStore")
    @patch("sys.stdin", new_callable=io.StringIO)
    def test_run_hook_session_not_found(self, mock_stdin, mock_store_class):
        """Test when session is not found."""
        mock_store = MagicMock()
        mock_store.get_session.return_value = None
        mock_store_class.return_value = mock_store

        mock_stdin.write(json.dumps({
            "conversation_id": "unknown-sess",
            "toolUse": {"name": "view", "input": {}}
        }))
        mock_stdin.seek(0)

        tool_use.run_hook("PostToolUse")
        # Should complete without error, no session to update


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


class TestSendNotification:
    """Tests for send_notification function."""

    @patch("augment_agent_dashboard.hooks.stop.send_browser_notification")
    @patch("shutil.which")
    def test_send_notification_no_terminal_notifier(self, mock_which, mock_browser, capsys):
        mock_which.return_value = None
        stop.send_notification("Title", "Message", "workspace", "sess-1")
        mock_browser.assert_called_once()
        captured = capsys.readouterr()
        assert "terminal-notifier not found" in captured.err

    @patch("augment_agent_dashboard.hooks.stop.send_browser_notification")
    @patch("subprocess.run")
    @patch("shutil.which")
    def test_send_notification_with_terminal_notifier(self, mock_which, mock_run, mock_browser):
        mock_which.return_value = "/usr/local/bin/terminal-notifier"
        mock_run.return_value = MagicMock(returncode=0)
        stop.send_notification("Title", "Message", "workspace", "sess-1", sound=True)
        mock_run.assert_called_once()
        assert "-sound" in mock_run.call_args[0][0]

    @patch("augment_agent_dashboard.hooks.stop.send_browser_notification")
    @patch("subprocess.run")
    @patch("shutil.which")
    def test_send_notification_exception(self, mock_which, mock_run, mock_browser, capsys):
        mock_which.return_value = "/usr/local/bin/terminal-notifier"
        mock_run.side_effect = Exception("Test error")
        stop.send_notification("Title", "Message", "workspace", "sess-1")
        captured = capsys.readouterr()
        assert "Notification error" in captured.err


class TestSendBrowserNotification:
    """Tests for send_browser_notification function."""

    @patch("urllib.request.urlopen")
    def test_send_browser_notification_success(self, mock_urlopen):
        mock_urlopen.return_value.__enter__ = MagicMock()
        mock_urlopen.return_value.__exit__ = MagicMock()
        stop.send_browser_notification("Title", "Body", "http://url", 9000)
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_send_browser_notification_error(self, mock_urlopen, capsys):
        mock_urlopen.side_effect = Exception("Connection error")
        stop.send_browser_notification("Title", "Body", "http://url", 9000)
        captured = capsys.readouterr()
        assert "Browser notification error" in captured.err


class TestSpawnLoopMessage:
    """Tests for spawn_loop_message function."""

    @patch("shutil.which")
    def test_spawn_loop_message_no_auggie(self, mock_which, capsys):
        mock_which.return_value = None
        stop.spawn_loop_message("conv-1", "/workspace", "prompt")
        captured = capsys.readouterr()
        assert "auggie not found" in captured.err

    @patch("shutil.which")
    def test_spawn_loop_message_no_workspace(self, mock_which, capsys):
        mock_which.return_value = "/usr/local/bin/auggie"
        stop.spawn_loop_message("conv-1", None, "prompt")
        captured = capsys.readouterr()
        assert "No workspace root" in captured.err

    @patch("subprocess.Popen")
    @patch("shutil.which")
    def test_spawn_loop_message_success(self, mock_which, mock_popen):
        mock_which.return_value = "/usr/local/bin/auggie"
        stop.spawn_loop_message("conv-1", "/workspace", "prompt")
        mock_popen.assert_called_once()

    @patch("subprocess.Popen")
    @patch("shutil.which")
    def test_spawn_loop_message_exception(self, mock_which, mock_popen, capsys):
        mock_which.return_value = "/usr/local/bin/auggie"
        mock_popen.side_effect = Exception("Spawn error")
        stop.spawn_loop_message("conv-1", "/workspace", "prompt")
        captured = capsys.readouterr()
        assert "Failed to spawn loop message" in captured.err


class TestStopHookLoopLogic:
    """Tests for loop and queue logic in stop hook."""

    @patch("augment_agent_dashboard.hooks.stop.spawn_loop_message")
    @patch("augment_agent_dashboard.hooks.stop.send_notification")
    @patch("augment_agent_dashboard.hooks.stop.load_config")
    @patch("augment_agent_dashboard.hooks.stop.SessionStore")
    @patch("sys.stdin", new_callable=io.StringIO)
    def test_loop_enabled_increments_count(self, mock_stdin, mock_store_class, mock_config, mock_notify, mock_spawn, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        (tmp_path / ".augment" / "dashboard").mkdir(parents=True)

        mock_store = MagicMock()
        session = AgentSession(
            session_id="conv-123",
            conversation_id="conv-123",
            workspace_root="/path/to/project",
            workspace_name="project",
            loop_enabled=True,
            loop_count=0,
        )
        mock_store.get_session.return_value = session
        mock_store_class.return_value = mock_store
        mock_config.return_value = {"max_loop_iterations": 10}

        mock_stdin.write(json.dumps({
            "workspace_roots": ["/path/to/project"],
            "conversation_id": "conv-123",
            "conversation": {"agentTextResponse": "Done"}
        }))
        mock_stdin.seek(0)

        stop.run_hook()

        mock_spawn.assert_called_once()
        mock_store.upsert_session.assert_called()

    @patch("augment_agent_dashboard.hooks.stop.spawn_loop_message")
    @patch("augment_agent_dashboard.hooks.stop.send_notification")
    @patch("augment_agent_dashboard.hooks.stop.load_config")
    @patch("augment_agent_dashboard.hooks.stop.SessionStore")
    @patch("sys.stdin", new_callable=io.StringIO)
    def test_loop_max_iterations_reached(self, mock_stdin, mock_store_class, mock_config, mock_notify, mock_spawn, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        (tmp_path / ".augment" / "dashboard").mkdir(parents=True)

        mock_store = MagicMock()
        session = AgentSession(
            session_id="conv-123",
            conversation_id="conv-123",
            workspace_root="/path/to/project",
            workspace_name="project",
            loop_enabled=True,
            loop_count=50,  # At max
        )
        mock_store.get_session.return_value = session
        mock_store_class.return_value = mock_store
        mock_config.return_value = {"max_loop_iterations": 50}

        mock_stdin.write(json.dumps({
            "workspace_roots": ["/path/to/project"],
            "conversation_id": "conv-123",
            "conversation": {"agentTextResponse": "Done"}
        }))
        mock_stdin.seek(0)

        stop.run_hook()

        # spawn_loop_message should NOT be called, but notification should
        mock_spawn.assert_not_called()
        # Notification called twice: once for turn complete, once for loop complete
        assert mock_notify.call_count == 2

    @patch("augment_agent_dashboard.hooks.stop.spawn_loop_message")
    @patch("augment_agent_dashboard.hooks.stop.send_notification")
    @patch("augment_agent_dashboard.hooks.stop.load_config")
    @patch("augment_agent_dashboard.hooks.stop.SessionStore")
    @patch("sys.stdin", new_callable=io.StringIO)
    def test_queued_message_processing(self, mock_stdin, mock_store_class, mock_config, mock_notify, mock_spawn, tmp_path, monkeypatch):
        from augment_agent_dashboard.models import SessionMessage
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        (tmp_path / ".augment" / "dashboard").mkdir(parents=True)

        mock_store = MagicMock()
        queued_msg = SessionMessage(role="queued", content="Do this next")
        session = AgentSession(
            session_id="conv-123",
            conversation_id="conv-123",
            workspace_root="/path/to/project",
            workspace_name="project",
            loop_enabled=False,
            messages=[queued_msg],
        )
        mock_store.get_session.return_value = session
        mock_store_class.return_value = mock_store
        mock_config.return_value = {}

        mock_stdin.write(json.dumps({
            "workspace_roots": ["/path/to/project"],
            "conversation_id": "conv-123",
            "conversation": {"agentTextResponse": "Done"}
        }))
        mock_stdin.seek(0)

        stop.run_hook()

        # Should spawn with the queued message content
        mock_spawn.assert_called_once()
        assert "Do this next" in mock_spawn.call_args[0][2]

