"""Install Augment hooks for the dashboard plugin."""

import json
import os
import shutil
import stat
import sys
from pathlib import Path


def get_settings_file() -> Path:
    """Get the Augment settings file path."""
    return Path.home() / ".augment" / "settings.json"


def get_hooks_scripts_dir() -> Path:
    """Get the directory for hook shell scripts."""
    return Path.home() / ".augment" / "dashboard" / "hooks"


def find_command_path(command: str) -> str | None:
    """Find the full path to a command."""
    return shutil.which(command)


def create_wrapper_script(script_path: Path, python_command: str) -> None:
    """Create a shell wrapper script that calls a Python command."""
    script_content = f"""#!/usr/bin/env bash
# Wrapper script for Augment hook - calls Python command
exec {python_command} "$@"
"""
    script_path.write_text(script_content)
    # Make executable
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def install_hooks() -> None:
    """Install the dashboard hooks into Augment."""
    settings_file = get_settings_file()
    scripts_dir = get_hooks_scripts_dir()
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Find paths to our hook commands
    session_start_path = find_command_path("augment-dashboard-session-start")
    stop_path = find_command_path("augment-dashboard-stop")
    pre_tool_path = find_command_path("augment-dashboard-pre-tool")
    post_tool_path = find_command_path("augment-dashboard-post-tool")

    if not session_start_path:
        print("Warning: augment-dashboard-session-start not found in PATH")
        print("Make sure augment-agent-dashboard is installed")
        sys.exit(1)

    if not stop_path:
        print("Warning: augment-dashboard-stop not found in PATH")
        print("Make sure augment-agent-dashboard is installed")
        sys.exit(1)

    # Create wrapper shell scripts (Augment requires .sh files)
    session_start_script = scripts_dir / "session-start.sh"
    stop_script = scripts_dir / "stop.sh"
    pre_tool_script = scripts_dir / "pre-tool.sh"
    post_tool_script = scripts_dir / "post-tool.sh"

    create_wrapper_script(session_start_script, session_start_path)
    print(f"Created wrapper: {session_start_script}")

    create_wrapper_script(stop_script, stop_path)
    print(f"Created wrapper: {stop_script}")

    if pre_tool_path:
        create_wrapper_script(pre_tool_script, pre_tool_path)
        print(f"Created wrapper: {pre_tool_script}")

    if post_tool_path:
        create_wrapper_script(post_tool_script, post_tool_path)
        print(f"Created wrapper: {post_tool_script}")

    # Load existing settings or create new
    if settings_file.exists():
        with open(settings_file) as f:
            settings = json.load(f)
    else:
        settings = {}

    # Ensure hooks structure exists
    if "hooks" not in settings:
        settings["hooks"] = {}

    # Dashboard hook identifier - used to find and update our hooks
    DASHBOARD_MARKER = "/dashboard/hooks/"

    def add_or_update_hook(hook_type: str, script_path: str, timeout: int = 5000, metadata: dict | None = None):
        """Add or update a dashboard hook without removing other hooks."""
        if hook_type not in settings["hooks"]:
            settings["hooks"][hook_type] = []

        hook_list = settings["hooks"][hook_type]

        # Find existing dashboard hook entry
        dashboard_entry_idx = None
        for i, entry in enumerate(hook_list):
            if "hooks" in entry:
                for hook in entry["hooks"]:
                    if hook.get("type") == "command" and DASHBOARD_MARKER in hook.get("command", ""):
                        dashboard_entry_idx = i
                        break
            if dashboard_entry_idx is not None:
                break

        # Create the new hook entry
        new_entry = {
            "hooks": [
                {
                    "type": "command",
                    "command": script_path,
                    "timeout": timeout,
                }
            ]
        }
        if metadata:
            new_entry["metadata"] = metadata

        if dashboard_entry_idx is not None:
            # Update existing dashboard hook
            hook_list[dashboard_entry_idx] = new_entry
        else:
            # Add new dashboard hook
            hook_list.append(new_entry)

    # Add/update SessionStart hook
    add_or_update_hook("SessionStart", str(session_start_script))

    # Add/update Stop hook with conversation data
    add_or_update_hook("Stop", str(stop_script), metadata={"includeConversationData": True})

    # Add/update PreToolUse hook (optional)
    if pre_tool_path:
        add_or_update_hook("PreToolUse", str(pre_tool_script))

    # Add/update PostToolUse hook (optional)
    if post_tool_path:
        add_or_update_hook("PostToolUse", str(post_tool_script))

    # Write settings
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_file, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"Updated: {settings_file}")

    # Clean up old hook files if they exist
    old_hooks_dir = Path.home() / ".augment" / "hooks"
    for old_file in ["dashboard-session-start.json", "dashboard-stop.json"]:
        old_path = old_hooks_dir / old_file
        if old_path.exists():
            old_path.unlink()
            print(f"Removed old config: {old_path}")

    print("\nDashboard hooks installed successfully!")
    print("\nTo start the dashboard:")
    print("  augment-dashboard")
    print("\nThen open http://localhost:8080 in your browser.")


def main():
    """Main entry point."""
    install_hooks()


if __name__ == "__main__":
    main()

