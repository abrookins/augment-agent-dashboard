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
    # URL to open the session in the dashboard
    session_url = f"http://localhost:{port}/session/{urllib.parse.quote(session_id, safe='')}"

    # Clean up message - remove newlines and extra whitespace
    clean_message = " ".join(message.split())[:100] if message else "Turn complete"

    # Send browser notification via dashboard API
    send_browser_notification(title, f"{workspace_name}: {clean_message}", session_url, port)

    # Also send macOS notification via terminal-notifier
    notifier = shutil.which("terminal-notifier")
    if not notifier:
        sys.stderr.write("terminal-notifier not found\n")
        return

    sys.stderr.write(f"Sending notification: title={title}, message={message[:50]}, url={session_url}\n")

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


def send_browser_notification(title: str, body: str, url: str, port: int = 9000) -> None:
    """Send a browser notification via the dashboard API."""
    import urllib.request

    try:
        data = urllib.parse.urlencode({
            "title": title,
            "body": body,
            "url": url,
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{port}/api/notifications/send",
            data=data,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            sys.stderr.write(f"Browser notification sent: {resp.status}\n")
    except Exception as e:
        sys.stderr.write(f"Browser notification error: {e}\n")


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


def spawn_loop_message(conversation_id: str, workspace_root: str | None, message: str) -> None:
    """Spawn auggie subprocess to send the loop prompt."""
    auggie_path = shutil.which("auggie")
    if not auggie_path:
        sys.stderr.write("auggie not found, cannot spawn loop message\n")
        return

    if not workspace_root:
        sys.stderr.write("No workspace root, cannot spawn loop message\n")
        return

    try:
        # Spawn auggie in background - don't wait for it
        subprocess.Popen(
            [auggie_path, "--resume", conversation_id, "--print", message],
            cwd=workspace_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # Detach from parent process
        )
        sys.stderr.write(f"Spawned loop message for {conversation_id}\n")
    except Exception as e:
        sys.stderr.write(f"Failed to spawn loop message: {e}\n")


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

        # Load config for notifications and loop settings
        config = load_config()

        # Send desktop notification with link to session
        preview = agent_text[:80] if agent_text else "Turn complete"
        send_notification(
            "Agent Turn Complete",
            preview,
            workspace_name,
            session_id,
            port=config.get("port", 9000),
            sound=config.get("notification_sound", True),
        )

        # Check if quality loop is enabled for this session
        session = store.get_session(session_id)
        if session and session.loop_enabled:
            max_iterations = config.get("max_loop_iterations", 50)
            if session.loop_count < max_iterations:
                # Increment loop count
                session.loop_count += 1
                store.upsert_session(session)

                # Get loop prompt from config using the session's selected prompt name
                loop_prompts = config.get("loop_prompts", {})
                prompt_name = session.loop_prompt_name
                default_prompt = "Did you use TDD, reach 100% test coverage, and verify quality at .8 or above with mfcqi? If not, continue working. If choices must be made, choose wisely."
                loop_prompt = loop_prompts.get(prompt_name, default_prompt) if prompt_name else default_prompt

                sys.stderr.write(f"Quality loop iteration {session.loop_count}/{max_iterations} using '{prompt_name}'\n")

                # Spawn auggie with the loop prompt
                spawn_loop_message(conversation_id, workspace_root, loop_prompt)
            else:
                # Max iterations reached, disable loop
                session.loop_enabled = False
                store.upsert_session(session)
                sys.stderr.write(f"Quality loop reached max iterations ({max_iterations}), disabling\n")
                send_notification(
                    "Quality Loop Complete",
                    f"Reached {max_iterations} iterations",
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

