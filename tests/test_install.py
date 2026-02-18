"""Tests for install module."""

import json
import stat
from unittest.mock import patch

import pytest

from augment_agent_dashboard import install


class TestHelperFunctions:
    """Tests for install helper functions."""

    def test_get_settings_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = install.get_settings_file()
        assert result == tmp_path / ".augment" / "settings.json"

    def test_get_hooks_scripts_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = install.get_hooks_scripts_dir()
        assert result == tmp_path / ".augment" / "dashboard" / "hooks"

    def test_find_command_path_exists(self):
        # 'python' should exist on any test system
        result = install.find_command_path("python")
        assert result is not None
        assert "python" in result

    def test_find_command_path_not_exists(self):
        result = install.find_command_path("nonexistent-command-xyz")
        assert result is None

    def test_create_wrapper_script(self, tmp_path):
        script_path = tmp_path / "test.sh"
        install.create_wrapper_script(script_path, "/usr/bin/test-command")

        assert script_path.exists()
        content = script_path.read_text()
        assert "#!/usr/bin/env bash" in content
        assert "exec /usr/bin/test-command" in content

        # Check executable permission
        mode = script_path.stat().st_mode
        assert mode & stat.S_IXUSR


class TestInstallHooks:
    """Tests for install_hooks function."""

    @patch("augment_agent_dashboard.install.find_command_path")
    def test_install_hooks_missing_session_start(self, mock_find, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        mock_find.return_value = None

        with pytest.raises(SystemExit) as exc_info:
            install.install_hooks()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "augment-dashboard-session-start not found" in captured.out

    @patch("augment_agent_dashboard.install.find_command_path")
    def test_install_hooks_missing_stop(self, mock_find, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        def find_path(cmd):
            if cmd == "augment-dashboard-session-start":
                return "/fake/session-start"
            return None

        mock_find.side_effect = find_path

        with pytest.raises(SystemExit) as exc_info:
            install.install_hooks()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "augment-dashboard-stop not found" in captured.out

    @patch("augment_agent_dashboard.install.find_command_path")
    def test_install_hooks_success(self, mock_find, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        def find_path(cmd):
            return f"/fake/{cmd}"

        mock_find.side_effect = find_path

        # Create augment dir
        augment_dir = tmp_path / ".augment"
        augment_dir.mkdir()

        install.install_hooks()

        # Verify settings file was created
        settings_file = augment_dir / "settings.json"
        assert settings_file.exists()

        settings = json.loads(settings_file.read_text())
        assert "hooks" in settings
        assert "SessionStart" in settings["hooks"]
        assert "Stop" in settings["hooks"]

        captured = capsys.readouterr()
        assert "Dashboard hooks installed successfully" in captured.out

    @patch("augment_agent_dashboard.install.find_command_path")
    def test_install_hooks_preserves_existing(self, mock_find, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        def find_path(cmd):
            return f"/fake/{cmd}"

        mock_find.side_effect = find_path

        # Create existing settings with other hooks
        augment_dir = tmp_path / ".augment"
        augment_dir.mkdir()
        settings_file = augment_dir / "settings.json"
        settings_file.write_text(json.dumps({
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": "/other/plugin/hook.sh"}]}
                ]
            }
        }))

        install.install_hooks()

        # Verify other hooks were preserved
        settings = json.loads(settings_file.read_text())
        session_start_hooks = settings["hooks"]["SessionStart"]

        # Should have both the original and the dashboard hook
        assert len(session_start_hooks) == 2
        commands = []
        for entry in session_start_hooks:
            for hook in entry.get("hooks", []):
                commands.append(hook.get("command", ""))

        assert any("/other/plugin/hook.sh" in c for c in commands)
        assert any("/dashboard/hooks/" in c for c in commands)

    @patch("augment_agent_dashboard.install.find_command_path")
    def test_install_hooks_updates_existing_dashboard_hook(
        self, mock_find, tmp_path, monkeypatch, capsys
    ):
        """Test that existing dashboard hooks are updated, not duplicated."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        def find_path(cmd):
            return f"/fake/{cmd}"

        mock_find.side_effect = find_path

        # Create existing settings with dashboard hook already present
        augment_dir = tmp_path / ".augment"
        augment_dir.mkdir()
        settings_file = augment_dir / "settings.json"
        old_cmd = "/old/dashboard/hooks/session-start.sh"
        settings_file.write_text(json.dumps({
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": old_cmd}]}
                ]
            }
        }))

        install.install_hooks()

        # Verify dashboard hook was updated, not duplicated
        settings = json.loads(settings_file.read_text())
        session_start_hooks = settings["hooks"]["SessionStart"]

        # Should have only one hook (the updated dashboard hook)
        assert len(session_start_hooks) == 1
        # The command should be the new one
        hook_cmd = session_start_hooks[0]["hooks"][0]["command"]
        assert "/dashboard/hooks/" in hook_cmd

    @patch("augment_agent_dashboard.install.find_command_path")
    def test_install_hooks_cleans_old_hook_files(self, mock_find, tmp_path, monkeypatch, capsys):
        """Test that old hook config files are cleaned up."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        def find_path(cmd):
            return f"/fake/{cmd}"

        mock_find.side_effect = find_path

        # Create augment dir and old hooks dir with old files
        augment_dir = tmp_path / ".augment"
        augment_dir.mkdir()
        old_hooks_dir = augment_dir / "hooks"
        old_hooks_dir.mkdir()

        old_session_start = old_hooks_dir / "dashboard-session-start.json"
        old_stop = old_hooks_dir / "dashboard-stop.json"
        old_session_start.write_text("{}")
        old_stop.write_text("{}")

        install.install_hooks()

        # Verify old files were removed
        assert not old_session_start.exists()
        assert not old_stop.exists()

        captured = capsys.readouterr()
        assert "Removed old config" in captured.out


class TestInstallMemoryHooks:
    """Tests for install_memory_hooks function."""

    def test_install_memory_hooks_success(self, tmp_path, monkeypatch, capsys):
        """Test successful memory hooks installation."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Create augment dir and settings file
        augment_dir = tmp_path / ".augment"
        augment_dir.mkdir()
        settings_file = augment_dir / "settings.json"
        settings_file.write_text("{}")

        result = install.install_memory_hooks(enable_tool_tracking=False)

        assert result is True
        captured = capsys.readouterr()
        assert "Installing Agent Memory hooks" in captured.out
        assert "Memory hooks installed successfully" in captured.out

        # Verify hooks dir was created
        memory_hooks_dir = augment_dir / "memory-hooks"
        assert memory_hooks_dir.exists()

    def test_install_memory_hooks_with_tool_tracking(self, tmp_path, monkeypatch, capsys):
        """Test memory hooks installation with tool tracking enabled."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Create augment dir and settings file
        augment_dir = tmp_path / ".augment"
        augment_dir.mkdir()
        settings_file = augment_dir / "settings.json"
        settings_file.write_text("{}")

        result = install.install_memory_hooks(enable_tool_tracking=True)

        assert result is True

        # Verify PostToolUse hook is in settings when tool tracking enabled
        import json
        settings = json.loads(settings_file.read_text())
        assert "hooks" in settings
        assert "PostToolUse" in settings["hooks"]

    def test_install_memory_hooks_import_error(self, capsys):
        """Test graceful handling when augment-agent-memory is not installed."""
        import sys

        # Save the original module if it exists
        original_module = sys.modules.get("augment_agent_memory.install")

        # Remove the module from sys.modules to simulate it not being installed
        if "augment_agent_memory.install" in sys.modules:
            del sys.modules["augment_agent_memory.install"]
        if "augment_agent_memory" in sys.modules:
            original_parent = sys.modules.get("augment_agent_memory")
            del sys.modules["augment_agent_memory"]
        else:
            original_parent = None

        # Make the import fail
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("augment_agent_memory"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = mock_import

            # Now call the function - it should return False gracefully
            result = install.install_memory_hooks()

            assert result is False
            captured = capsys.readouterr()
            assert "augment-agent-memory not found" in captured.out
        finally:
            # Restore original import
            builtins.__import__ = original_import
            # Restore original modules
            if original_module:
                sys.modules["augment_agent_memory.install"] = original_module
            if original_parent:
                sys.modules["augment_agent_memory"] = original_parent

    def test_install_memory_hooks_creates_scripts(self, tmp_path, monkeypatch):
        """Test that memory hook scripts are created."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Create augment dir and settings file
        augment_dir = tmp_path / ".augment"
        augment_dir.mkdir()
        settings_file = augment_dir / "settings.json"
        settings_file.write_text("{}")

        install.install_memory_hooks()

        # Verify shell scripts were created
        memory_hooks_dir = augment_dir / "memory-hooks"
        assert (memory_hooks_dir / "session_start.sh").exists()
        assert (memory_hooks_dir / "stop.sh").exists()
        assert (memory_hooks_dir / "post_tool_use.sh").exists()

    def test_install_memory_hooks_updates_settings(self, tmp_path, monkeypatch):
        """Test that settings.json is updated with memory hooks."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Create augment dir and settings file
        augment_dir = tmp_path / ".augment"
        augment_dir.mkdir()
        settings_file = augment_dir / "settings.json"
        settings_file.write_text("{}")

        install.install_memory_hooks()

        # Verify settings were updated
        import json
        settings = json.loads(settings_file.read_text())
        assert "hooks" in settings
        assert "SessionStart" in settings["hooks"]
        assert "Stop" in settings["hooks"]

        # Verify memory hooks are present (contain memory-hooks path)
        session_hooks = settings["hooks"]["SessionStart"]
        assert any(
            "memory-hooks" in str(entry)
            for entry in session_hooks
        )

    def test_install_memory_hooks_preserves_dashboard_hooks(self, tmp_path, monkeypatch):
        """Test that memory hooks don't overwrite dashboard hooks."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Create augment dir with existing dashboard hooks
        augment_dir = tmp_path / ".augment"
        augment_dir.mkdir()
        settings_file = augment_dir / "settings.json"
        settings_file.write_text(json.dumps({
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": "/dashboard/hooks/session-start.sh"}]}
                ]
            }
        }))

        install.install_memory_hooks()

        # Verify both hooks are present
        settings = json.loads(settings_file.read_text())
        session_hooks = settings["hooks"]["SessionStart"]

        # Should have both dashboard and memory hooks
        assert len(session_hooks) == 2
        commands = []
        for entry in session_hooks:
            for hook in entry.get("hooks", []):
                commands.append(hook.get("command", ""))

        assert any("/dashboard/hooks/" in c for c in commands)
        assert any("memory-hooks" in c for c in commands)


class TestMain:
    """Tests for main entry point."""

    @patch("augment_agent_dashboard.install.install_hooks")
    @patch("augment_agent_dashboard.install.install_memory_hooks")
    def test_main(self, mock_memory_install, mock_install, monkeypatch):
        """Test main calls install_hooks and install_memory_hooks."""
        import sys

        # Mock sys.argv to avoid argparse picking up pytest args
        monkeypatch.setattr(sys, "argv", ["augment-dashboard-install"])
        install.main()
        mock_install.assert_called_once()
        mock_memory_install.assert_called_once()

    @patch("augment_agent_dashboard.install.install_hooks")
    @patch("augment_agent_dashboard.install.install_memory_hooks")
    def test_main_skip_memory(self, mock_memory_install, mock_install, monkeypatch):
        """Test main with --skip-memory flag."""
        import sys

        monkeypatch.setattr(sys, "argv", ["augment-dashboard-install", "--skip-memory"])
        install.main()
        mock_install.assert_called_once()
        mock_memory_install.assert_not_called()

    def test_install_as_main(self, tmp_path, monkeypatch):
        """Test running install as __main__."""
        import runpy
        import sys
        import warnings

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setattr(sys, "argv", ["augment-dashboard-install"])

        with patch("augment_agent_dashboard.install.install_hooks"):
            with patch("augment_agent_dashboard.install.install_memory_hooks"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    runpy.run_module(
                        "augment_agent_dashboard.install",
                        run_name="__main__"
                    )

