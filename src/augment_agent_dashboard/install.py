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

    create_wrapper_script(session_start_script, session_start_path)
    print(f"Created wrapper: {session_start_script}")

    create_wrapper_script(stop_script, stop_path)
    print(f"Created wrapper: {stop_script}")

    # Load existing settings or create new
    if settings_file.exists():
        with open(settings_file) as f:
            settings = json.load(f)
    else:
        settings = {}

    # Ensure hooks structure exists
    if "hooks" not in settings:
        settings["hooks"] = {}

    # Add SessionStart hook
    settings["hooks"]["SessionStart"] = [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": str(session_start_script),
                    "timeout": 5000,
                }
            ]
        }
    ]

    # Add Stop hook with conversation data
    settings["hooks"]["Stop"] = [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": str(stop_script),
                    "timeout": 5000,
                }
            ],
            "metadata": {
                "includeConversationData": True,
            },
        }
    ]

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

