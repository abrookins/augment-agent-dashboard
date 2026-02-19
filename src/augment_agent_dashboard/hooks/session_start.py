"""SessionStart hook - registers session and delivers pending messages."""

import json
import sys
from pathlib import Path

from ..models import AgentSession
from ..state_machine import get_state_machine
from ..store import SessionStore


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
    """Use conversation_id as session ID for consistency with auggie --resume.

    Always use the raw conversation_id (UUID) as the session_id.
    This ensures that when we resume a session with `auggie --resume <conversation_id>`,
    the hooks will find/update the correct session.
    """
    return conversation_id


def run_hook() -> None:
    """Entry point for SessionStart hook."""
    import os
    from datetime import datetime

    import psutil

    # Write to a log file silently (no stderr output to avoid TUI issues)
    log_file = Path.home() / ".augment" / "dashboard" / "hook_debug.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        with open(log_file, "a") as f:
            f.write(f"{datetime.now().isoformat()} {msg}\n")

    # Log process info for debugging
    my_pid = os.getpid()
    parent_pid = os.getppid()
    log(f" SessionStart hook - PID: {my_pid}, Parent PID: {parent_pid}")

    try:
        parent = psutil.Process(parent_pid)
        log(f"  Parent: {parent.name()} cwd={parent.cwd()}")
    except Exception as e:
        log(f"  Error getting parent info: {e}")

    # Read hook input from stdin
    hook_input = {}
    try:
        hook_input = json.load(sys.stdin)
        log(f"  Hook input: {json.dumps(hook_input)}")
    except json.JSONDecodeError:
        log("  No JSON input received")
        pass

    # Extract workspace and conversation info
    workspace_roots = hook_input.get("workspace_roots", [])
    conversation_id = hook_input.get("conversation_id", "unknown")
    workspace_root = get_workspace_root(workspace_roots)
    workspace_name = get_workspace_name(workspace_root)
    session_id = get_session_id(conversation_id)

    log(f"  Session: {session_id}, workspace: {workspace_root}")

    try:
        store = SessionStore()
        state_machine = get_state_machine()

        # Check if session exists
        existing = store.get_session(session_id)
        dashboard_messages: list[str] = []

        if existing:
            # Use state machine to transition to ACTIVE
            result = state_machine.process_event(existing, "session_start")
            log(f"  State transition: {result.old_state} -> {result.new_state}")

            # Update PID and save session
            existing.agent_pid = parent_pid
            store.upsert_session(existing)

            # Get and clear any pending dashboard messages
            dashboard_messages = store.get_and_clear_dashboard_messages(session_id)
            if dashboard_messages:
                log(f"  Got {len(dashboard_messages)} pending dashboard messages")
        else:
            # Create new session with parent PID
            session = AgentSession(
                session_id=session_id,
                conversation_id=conversation_id,
                workspace_root=workspace_root or "",
                workspace_name=workspace_name,
                agent_pid=parent_pid,
            )
            # Use state machine to set initial state
            result = state_machine.process_event(session, "session_start")
            log(f"  New session state: {result.new_state}")

            # Check for pending initial prompt from dashboard
            # Import here to avoid circular imports
            from ..server import get_and_clear_pending_prompt
            from ..models import SessionMessage

            if workspace_root:
                pending_prompt = get_and_clear_pending_prompt(workspace_root)
                if pending_prompt:
                    # Add the initial user message to the session
                    session.messages.append(SessionMessage(
                        role="user",
                        content=pending_prompt,
                    ))
                    log(f"  Added initial prompt as user message")

            store.upsert_session(session)
            log(f"  Registered new session: {session_id}")

        # Output dashboard messages to inject into context
        if dashboard_messages:
            context_parts = ["## Messages from Dashboard"]
            for msg in dashboard_messages:
                context_parts.append(f"- {msg}")
            print("\n".join(context_parts))
        else:
            print(json.dumps({}))

    except Exception as e:
        log(f"  Dashboard store error: {e}")
        print(json.dumps({}))


def main():
    """Main entry point."""
    run_hook()


if __name__ == "__main__":
    main()

