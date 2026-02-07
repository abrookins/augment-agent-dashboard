"""Stop hook - updates session with conversation messages."""

import json
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path

from ..models import AgentSession, SessionMessage, SessionStatus
from ..store import SessionStore

# Config file location
CONFIG_PATH = Path.home() / ".augment" / "dashboard" / "config.json"


def load_config() -> dict:
    """Load dashboard config."""
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {}


def send_notification(title: str, message: str, workspace_name: str, session_id: str, port: int = 9000, sound: bool = True) -> None:
    """Send a desktop notification using terminal-notifier if available."""
    notifier = shutil.which("terminal-notifier")
    if not notifier:
        sys.stderr.write("terminal-notifier not found\n")
        return

    # URL to open the session in the dashboard
    session_url = f"http://localhost:{port}/session/{urllib.parse.quote(session_id, safe='')}"

    sys.stderr.write(f"Sending notification: title={title}, message={message[:50]}, url={session_url}\n")

    # Clean up message - remove newlines and extra whitespace
    clean_message = " ".join(message.split())[:100] if message else "Turn complete"

    cmd = [
        notifier,
        "-title", title,
        "-subtitle", workspace_name,
        "-message", clean_message,
        "-group", f"augment-{workspace_name}",
        "-open", session_url,
    ]

    if sound:
        cmd.extend(["-sound", "default"])

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=5)
        sys.stderr.write(f"Notification result: rc={result.returncode}, stderr={result.stderr.decode()}\n")
    except Exception as e:
        sys.stderr.write(f"Notification error: {e}\n")


def get_workspace_root(workspace_roots: list[str]) -> str | None:
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


def run_hook() -> None:
    """Entry point for Stop hook."""
    # Read hook input from stdin
    hook_input = {}
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        pass

    # Extract workspace and conversation info
    workspace_roots = hook_input.get("workspace_roots", [])
    conversation_id = hook_input.get("conversation_id", "unknown")
    conversation = hook_input.get("conversation", {})
    workspace_root = get_workspace_root(workspace_roots)
    workspace_name = get_workspace_name(workspace_root)
    session_id = get_session_id(conversation_id)

    sys.stderr.write(
        f"Dashboard Stop: session={session_id}, workspace={workspace_root}\n"
    )

    try:
        store = SessionStore()

        # Extract messages from conversation
        user_prompt = conversation.get("userPrompt", "")
        agent_text = conversation.get("agentTextResponse", "")
        agent_code = conversation.get("agentCodeResponse", [])

        # Track files changed
        files_changed = []
        if isinstance(agent_code, list):
            for change in agent_code:
                if isinstance(change, dict) and "path" in change:
                    files_changed.append(change["path"])

        # Get or create session
        existing = store.get_session(session_id)
        if existing:
            # Add messages to existing session
            if user_prompt:
                store.add_message(
                    session_id,
                    SessionMessage(role="user", content=user_prompt),
                )
            if agent_text:
                store.add_message(
                    session_id,
                    SessionMessage(role="assistant", content=agent_text),
                )
            # Update status to idle (turn complete)
            store.update_session_status(
                session_id,
                SessionStatus.IDLE,
                current_task=user_prompt[:100] if user_prompt else None,
            )
        else:
            # Create new session with messages
            messages = []
            if user_prompt:
                messages.append(SessionMessage(role="user", content=user_prompt))
            if agent_text:
                messages.append(SessionMessage(role="assistant", content=agent_text))

            session = AgentSession(
                session_id=session_id,
                conversation_id=conversation_id,
                workspace_root=workspace_root or "",
                workspace_name=workspace_name,
                status=SessionStatus.IDLE,
                current_task=user_prompt[:100] if user_prompt else None,
                messages=messages,
                files_changed=files_changed,
            )
            store.upsert_session(session)

        sys.stderr.write(f"Updated dashboard session: {session_id}\n")

        # Send desktop notification with link to session
        config = load_config()
        preview = agent_text[:80] if agent_text else "Turn complete"
        send_notification(
            "Agent Turn Complete",
            preview,
            workspace_name,
            session_id,
            port=config.get("port", 9000),
            sound=config.get("notification_sound", True),
        )

    except Exception as e:
        sys.stderr.write(f"Dashboard store error: {e}\n")

    # Always output empty JSON
    print(json.dumps({}))


def main():
    """Main entry point."""
    run_hook()


if __name__ == "__main__":
    main()

