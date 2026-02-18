"""Tool use hooks - captures PreToolUse and PostToolUse events."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..models import SessionMessage
from ..store import SessionStore

# Config path
CONFIG_PATH = Path.home() / ".augment" / "dashboard" / "config.json"


def get_workspace_root(workspace_roots: list) -> str | None:
    """Get the primary workspace root from the list."""
    if not workspace_roots:
        return None
    return workspace_roots[0]


def get_workspace_name(workspace_root: str | None) -> str:
    """Extract workspace name from root path."""
    if not workspace_root:
        return "unknown"
    return Path(workspace_root).name


def get_session_id(conversation_id: str) -> str:
    """Use conversation_id as session ID for consistency with auggie --resume."""
    return conversation_id


def run_hook(hook_type: str = "PreToolUse"):
    """Entry point for tool use hooks."""
    hook_input = {}

    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        pass

    # Extract data from hook input
    workspace_roots = hook_input.get("workspace_roots", [])
    conversation_id = hook_input.get("conversation_id", "unknown")
    tool_use = hook_input.get("toolUse", {})

    get_workspace_root(workspace_roots)  # Currently unused, but may be needed for validation
    session_id = get_session_id(conversation_id)

    tool_name = tool_use.get("name", "unknown")
    tool_input = tool_use.get("input", {})

    # Debug logging disabled to reduce noise
    # sys.stderr.write(f"Dashboard {hook_type}: session={session_id}, tool={tool_name}\n")

    try:
        store = SessionStore()
        session = store.get_session(session_id)

        if session:
            # Add tool to tools_used list if not already there
            if tool_name not in session.tools_used:
                session.tools_used.append(tool_name)

            # For PostToolUse, we could also capture the result
            if hook_type == "PostToolUse":
                # Note: tool_output could be used to show results in the future
                # tool_output = tool_use.get("output", "")
                # Add a system message showing tool execution
                tool_msg = f"🔧 **{tool_name}**"
                if tool_input:
                    # Show abbreviated input
                    input_preview = json.dumps(tool_input)[:200]
                    if len(json.dumps(tool_input)) > 200:
                        input_preview += "..."
                    tool_msg += f"\n```json\n{input_preview}\n```"

                store.add_message(
                    session_id,
                    SessionMessage(role="system", content=tool_msg),
                )

            # Update last activity
            session.last_activity = datetime.now(timezone.utc)
            store.upsert_session(session)

    except Exception as e:
        sys.stderr.write(f"Dashboard tool hook error: {e}\n")


def run_pre_tool_use():
    """Entry point for PreToolUse hook."""
    run_hook("PreToolUse")


def run_post_tool_use():
    """Entry point for PostToolUse hook."""
    run_hook("PostToolUse")


def main():
    """Main entry point - defaults to PostToolUse."""
    run_post_tool_use()


if __name__ == "__main__":
    main()

