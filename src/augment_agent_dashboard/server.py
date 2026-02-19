"""FastAPI dashboard server for monitoring Augment agent sessions."""

import asyncio
import html
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import markdown
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .federation.models import FederationConfig, RemoteDashboard
from .federation.routes import router as federation_router
from .models import SessionStatus
from .store import SessionStore


def render_markdown(text: str) -> str:
    """Render markdown text to HTML."""
    return markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "nl2br"],
    )

app = FastAPI(title="Augment Agent Dashboard", version="0.1.0")

# Include federation routes
app.include_router(federation_router)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


def get_store() -> SessionStore:
    """Get the session store instance."""
    return SessionStore()


def _get_loop_prompts() -> dict[str, dict[str, str]]:
    """Get loop prompts from config file.

    Returns a dict mapping prompt names to their config (prompt and end_condition).
    Handles backward compatibility with old configs that stored prompts as strings.
    """
    import json
    config_path = Path.home() / ".augment" / "dashboard" / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            raw_prompts = config.get("loop_prompts", DEFAULT_LOOP_PROMPTS)
            # Handle backward compatibility: convert string prompts to dict format
            normalized: dict[str, dict[str, str]] = {}
            for name, value in raw_prompts.items():
                if isinstance(value, str):
                    # Legacy format: just a string prompt
                    normalized[name] = {"prompt": value, "end_condition": ""}
                else:
                    normalized[name] = value
            return normalized
        except Exception:
            pass
    return DEFAULT_LOOP_PROMPTS.copy()


def _get_quick_replies() -> dict[str, str]:
    """Get quick replies from config file.

    Returns a dict mapping reply names to their message content.
    Starts empty - users add their own quick replies via config.
    """
    config = _get_full_config()
    return config.get("quick_replies", {})


def _get_agent_timeout_minutes() -> int:
    """Get the agent timeout threshold in minutes from config.

    If a session is in a busy state and hasn't had activity for this long,
    it will be considered timed out (agent may have crashed).

    Default: 15 minutes.
    """
    config = _get_full_config()
    return config.get("agent_timeout_minutes", 15)


def check_and_reset_timed_out_sessions() -> list[str]:
    """Check for sessions that have timed out and reset them.

    A session is considered timed out if:
    - It's in a busy state (ACTIVE, UNDER_REVIEW, LOOP_PROMPTING)
    - last_activity is older than agent_timeout_minutes

    Returns a list of session IDs that were reset.

    Note: This only resets the sessions. To also process queued messages,
    use check_timeouts_and_process_queues() instead.
    """
    from .models import SessionMessage
    from .state_machine import SessionState

    store = get_store()
    timeout_minutes = _get_agent_timeout_minutes()
    now = datetime.now(timezone.utc)

    reset_session_ids = []

    for session in store.get_all_sessions():
        # Only check sessions in busy states
        if not session.state.is_busy():
            continue

        # Calculate time since last activity
        time_since_activity = now - session.last_activity
        minutes_inactive = time_since_activity.total_seconds() / 60

        if minutes_inactive >= timeout_minutes:
            # Session has timed out - reset it
            old_state = session.state
            session.state = SessionState.IDLE
            session.loop_enabled = False

            # Add a system message about the timeout
            session.messages.append(
                SessionMessage(
                    role="system",
                    content=(
                        f"⚠️ **Agent timed out** - no activity for {int(minutes_inactive)} minutes "
                        f"(was in state: {old_state.value}). Session reset to idle."
                    )
                )
            )
            session.last_activity = now
            store.upsert_session(session)
            reset_session_ids.append(session.session_id)

    return reset_session_ids


async def check_timeouts_and_process_queues() -> list[str]:
    """Check for timed out sessions and process any queued messages.

    This is the main entry point for timeout checking. It:
    1. Resets any timed out sessions to IDLE
    2. Processes queued messages for those sessions

    Returns list of session IDs that were reset.
    """
    reset_session_ids = check_and_reset_timed_out_sessions()

    # Process queued messages for each reset session
    for session_id in reset_session_ids:
        await process_queued_messages(session_id)

    return reset_session_ids


async def spawn_auggie_message(conversation_id: str, workspace_root: str, message: str) -> bool:
    """Spawn auggie subprocess to inject a message into a session.

    Returns True if successful, False otherwise.

    Note: This spawns a NEW auggie process with --resume. If another auggie
    process is already running for this conversation, behavior may vary.
    """
    import logging
    logger = logging.getLogger(__name__)

    auggie_path = shutil.which("auggie")
    if not auggie_path:
        logger.warning("auggie not found in PATH")
        return False

    logger.info(f"Spawning auggie --resume {conversation_id} in {workspace_root}")

    try:
        process = await asyncio.create_subprocess_exec(
            auggie_path,
            "--resume", conversation_id,
            "--print", message,
            cwd=workspace_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"auggie failed: rc={process.returncode}, stderr={stderr.decode()[:500]}")
            return False

        logger.info(f"auggie completed successfully for {conversation_id}")
        return True
    except Exception as e:
        logger.exception(f"Error spawning auggie: {e}")
        return False


async def process_queued_messages(session_id: str) -> bool:
    """Process queued messages for an idle session.

    If the session has queued messages and is idle, sends the first one.
    Returns True if a message was sent, False otherwise.
    """
    import logging
    logger = logging.getLogger(__name__)

    store = get_store()
    session = store.get_session(session_id)

    if not session:
        return False

    # Only process if session is idle
    from .state_machine import SessionState
    if session.state != SessionState.IDLE:
        return False

    # Check for queued messages
    queued_messages = [m for m in session.messages if m.role == "queued"]
    if not queued_messages:
        return False

    # Get the first queued message
    next_msg = queued_messages[0]
    message_content = next_msg.content

    # Convert queued message to user message
    next_msg.role = "user"
    store.upsert_session(session)

    logger.info(f"Processing queued message for session {session_id}")

    # Send the message
    if session.workspace_root and session.conversation_id:
        return await spawn_auggie_message(
            session.conversation_id,
            session.workspace_root,
            message_content,
        )

    return False


async def spawn_new_session(workspace_root: str, prompt: str) -> bool:
    """Spawn auggie subprocess to start a new session.

    Returns True if the process was successfully started, False otherwise.

    Note: This spawns a NEW auggie process without --resume, starting a fresh session.
    The process runs in the background (detached).
    """
    import logging
    logger = logging.getLogger(__name__)

    auggie_path = shutil.which("auggie")
    if not auggie_path:
        logger.warning("auggie not found in PATH")
        return False

    logger.info(f"Spawning new auggie session in {workspace_root}")

    # Save the prompt so SessionStart hook can add it as initial user message
    _save_pending_prompt(workspace_root, prompt)

    try:
        # Start auggie as a detached background process
        process = await asyncio.create_subprocess_exec(
            auggie_path,
            "--print", prompt,
            cwd=workspace_root,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,  # Detach from parent process
        )
        logger.info(f"New auggie session started with PID {process.pid}")
        return True
    except Exception as e:
        logger.exception(f"Error spawning new auggie session: {e}")
        return False


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard view showing all sessions."""
    from .federation.client import RemoteDashboardClient

    # Check for timed out sessions and process any queued messages
    await check_timeouts_and_process_queues()

    store = get_store()
    local_sessions = store.get_all_sessions()

    # Determine dark mode from query param or default to system preference
    dark_mode = request.query_params.get("dark", None)
    sort_by = request.query_params.get("sort", "recent")

    # Sort local sessions
    if sort_by == "name":
        local_sessions = sorted(local_sessions, key=lambda s: s.workspace_name.lower())

    # Get federation config
    fed_config = _get_federation_config()

    # If federation is enabled and we have remotes, fetch their sessions
    remote_sessions_by_origin: dict[str, list] = {}
    if fed_config.enabled and fed_config.remote_dashboards:
        import asyncio

        async def fetch_remote(remote: RemoteDashboard):
            client = RemoteDashboardClient(remote)
            sessions = await client.fetch_sessions()
            return (remote, sessions)

        # Fetch from all remotes in parallel
        tasks = [fetch_remote(r) for r in fed_config.remote_dashboards]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                continue
            remote, sessions = result
            if sort_by == "name":
                sessions = sorted(sessions, key=lambda s: s.workspace_name.lower())
            remote_sessions_by_origin[remote.url] = {
                "remote": remote,
                "sessions": sessions,
            }

    # Render with swim lanes if we have remotes configured
    if fed_config.remote_dashboards:
        page_html = render_dashboard_swimlanes(
            local_sessions,
            remote_sessions_by_origin,
            fed_config,
            dark_mode,
            sort_by,
        )
    else:
        # Single machine mode - no swim lanes needed
        page_html = render_dashboard(local_sessions, dark_mode, sort_by)

    return HTMLResponse(content=page_html)


@app.post("/session/new")
async def create_new_session(
    working_directory: Annotated[str, Form()],
    prompt: Annotated[str, Form()],
    background_tasks: BackgroundTasks,
):
    """Start a new auggie session in the specified directory with an initial prompt."""
    import logging
    import os
    logger = logging.getLogger(__name__)

    # Validate working directory
    if not working_directory or not working_directory.strip():
        raise HTTPException(status_code=400, detail="Working directory is required")

    working_directory = os.path.expanduser(working_directory.strip())

    if not os.path.isdir(working_directory):
        raise HTTPException(
            status_code=400, detail=f"Directory does not exist: {working_directory}"
        )

    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Track the working directory for recent directories feature
    _add_recent_working_directory(working_directory)

    # Spawn auggie in background
    background_tasks.add_task(spawn_new_session, working_directory, prompt.strip())

    logger.info(f"Starting new session in {working_directory}")

    # Redirect back to dashboard
    return RedirectResponse(url="/", status_code=303)


@app.get("/session/{session_id}", response_class=HTMLResponse)
async def session_detail(session_id: str, request: Request):
    """Session detail view showing conversation history."""
    from .federation.client import (
        RemoteDashboardClient,
        find_remote_by_hash,
        is_remote_session_id,
        parse_remote_session_id,
    )

    # Check for timed out sessions and process any queued messages
    await check_timeouts_and_process_queues()

    dark_mode = request.query_params.get("dark", None)
    loop_prompts = _get_loop_prompts()

    # Check if this is a remote session
    if is_remote_session_id(session_id):
        parsed = parse_remote_session_id(session_id)
        if not parsed:
            raise HTTPException(status_code=400, detail="Invalid remote session ID format")

        url_hash, remote_session_id = parsed
        fed_config = _get_federation_config()

        # Find the remote dashboard by hash
        remote = find_remote_by_hash(fed_config.remote_dashboards, url_hash)
        if not remote:
            raise HTTPException(
                status_code=404,
                detail="Remote dashboard not found - it may have been removed from config"
            )

        # Fetch the session from the remote
        client = RemoteDashboardClient(remote)
        session_data = await client.fetch_session_detail(remote_session_id)

        if not session_data:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found on remote dashboard '{remote.name}'"
            )

        # Render remote session detail
        page_html = render_remote_session_detail(
            session_data,
            remote,
            session_id,  # federated session ID for links
            dark_mode,
        )
        return HTMLResponse(content=page_html)

    # Local session
    store = get_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get local machine name from federation config
    fed_config = _get_federation_config()
    machine_name = fed_config.this_machine_name

    page_html = render_session_detail(session, dark_mode, loop_prompts, machine_name)
    return HTMLResponse(content=page_html)


@app.post("/session/{session_id}/message")
async def post_message(
    session_id: str,
    message: Annotated[str, Form()],
    background_tasks: BackgroundTasks,
):
    """Post a message to a session - injects it in real-time via auggie subprocess."""
    from augment_agent_dashboard.models import SessionMessage, SessionStatus

    store = get_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.conversation_id or session.conversation_id == "unknown":
        raise HTTPException(status_code=400, detail="Session has no conversation ID for resuming")

    if not session.workspace_root:
        raise HTTPException(status_code=400, detail="Session has no workspace root")

    # Immediately add the user message to the session so it shows in UI
    user_msg = SessionMessage(role="user", content=message.strip())
    store.add_message(session_id, user_msg)

    # Mark session as active
    store.update_session_status(session_id, SessionStatus.ACTIVE)

    # Spawn auggie in background to inject the message
    background_tasks.add_task(
        spawn_auggie_message,
        session.conversation_id,
        session.workspace_root,
        message,
    )

    return RedirectResponse(url=f"/session/{session_id}", status_code=303)


@app.post("/session/{session_id}/queue")
async def queue_message(
    session_id: str,
    message: Annotated[str, Form()],
):
    """Queue a message to be sent when the agent is ready."""
    from augment_agent_dashboard.models import SessionMessage

    store = get_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not message.strip():
        return RedirectResponse(url=f"/session/{session_id}", status_code=303)

    # Add as a queued message
    queued_msg = SessionMessage(role="queued", content=message.strip())
    store.add_message(session_id, queued_msg)

    return RedirectResponse(url=f"/session/{session_id}", status_code=303)


@app.post("/session/{session_id}/queue/clear")
async def clear_queue(session_id: str):
    """Clear all queued messages for a session."""
    store = get_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Remove all queued messages
    session.messages = [m for m in session.messages if m.role != "queued"]
    store.upsert_session(session)

    return RedirectResponse(url=f"/session/{session_id}", status_code=303)


@app.post("/session/{session_id}/delete")
async def delete_session(session_id: str):
    """Delete a session from the dashboard."""
    store = get_store()

    if not store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return RedirectResponse(url="/", status_code=303)


@app.post("/api/remote/session/{session_id}/message")
async def send_message_to_remote(
    session_id: str,
    message: Annotated[str, Form()],
):
    """Send a message to a remote session by proxying to its origin dashboard."""
    from .federation.client import (
        RemoteDashboardClient,
        find_remote_by_hash,
        is_remote_session_id,
        parse_remote_session_id,
    )

    if not is_remote_session_id(session_id):
        raise HTTPException(status_code=400, detail="Not a remote session ID")

    parsed = parse_remote_session_id(session_id)
    if not parsed:
        raise HTTPException(status_code=400, detail="Invalid remote session ID format")

    url_hash, remote_session_id = parsed
    fed_config = _get_federation_config()

    # Find the remote dashboard
    remote = find_remote_by_hash(fed_config.remote_dashboards, url_hash)
    if not remote:
        raise HTTPException(
            status_code=404,
            detail="Remote dashboard not found - it may have been removed from config"
        )

    # Send the message via the remote dashboard's API
    client = RemoteDashboardClient(remote)
    success = await client.send_message(remote_session_id, message.strip())

    if not success:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to send message to remote dashboard '{remote.name}'"
        )

    return RedirectResponse(url=f"/session/{session_id}", status_code=303)


@app.post("/api/remote/session/{session_id}/delete")
async def delete_remote_session(session_id: str):
    """Delete a remote session by proxying to its origin dashboard."""
    import logging
    logger = logging.getLogger(__name__)

    from .federation.client import (
        RemoteDashboardClient,
        find_remote_by_hash,
        is_remote_session_id,
        parse_remote_session_id,
    )

    if not is_remote_session_id(session_id):
        raise HTTPException(status_code=400, detail="Not a remote session ID")

    parsed = parse_remote_session_id(session_id)
    if not parsed:
        raise HTTPException(status_code=400, detail="Invalid remote session ID format")

    url_hash, remote_session_id = parsed
    logger.info(f"Deleting remote session: url_hash={url_hash}, remote_session_id={remote_session_id}")

    fed_config = _get_federation_config()

    # Find the remote dashboard
    remote = find_remote_by_hash(fed_config.remote_dashboards, url_hash)
    if not remote:
        logger.warning(f"Remote dashboard not found for hash {url_hash}")
        raise HTTPException(
            status_code=404,
            detail="Remote dashboard not found - it may have been removed from config"
        )

    logger.info(f"Found remote dashboard: {remote.name} at {remote.url}")

    # Delete the session via the remote dashboard's API
    client = RemoteDashboardClient(remote)
    success = await client.delete_session(remote_session_id)

    if not success:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to delete session on remote dashboard '{remote.name}'"
        )

    return RedirectResponse(url="/", status_code=303)


@app.get("/api/sessions")
async def api_list_sessions(
    status: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
):
    """API endpoint to list sessions."""
    store = get_store()
    sessions = store.get_all_sessions()

    if status:
        try:
            status_filter = SessionStatus(status)
            sessions = [s for s in sessions if s.status == status_filter]
        except ValueError:
            pass

    sessions = sessions[:limit]
    return {"sessions": [s.to_dict() for s in sessions]}


@app.get("/api/sessions/{session_id}")
async def api_get_session(session_id: str):
    """API endpoint to get session details."""
    store = get_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session.to_dict()


@app.get("/api/sessions-html")
async def api_sessions_html(
    sort: Annotated[str, Query()] = "recent",
):
    """API endpoint returning session cards HTML for AJAX updates."""
    store = get_store()
    sessions = store.get_all_sessions()

    # Sort sessions
    if sort == "name":
        sessions = sorted(sessions, key=lambda s: s.workspace_name.lower())
    # Default is "recent" which is already sorted by last_activity in get_all_sessions

    html = _render_session_cards(sessions)
    return HTMLResponse(content=html)


@app.get("/api/swimlanes-html")
async def api_swimlanes_html(
    sort: Annotated[str, Query()] = "recent",
):
    """API endpoint returning swim lanes HTML for AJAX updates."""
    from .federation.client import RemoteDashboardClient

    store = get_store()
    local_sessions = store.get_all_sessions()

    if sort == "name":
        local_sessions = sorted(local_sessions, key=lambda s: s.workspace_name.lower())

    fed_config = _get_federation_config()

    # Build lanes HTML
    lanes_html = ""

    # Local lane
    lanes_html += _render_swim_lane(
        lane_id="local",
        name=fed_config.this_machine_name,
        sessions=local_sessions,
        is_online=True,
        is_local=True,
    )

    # Remote lanes
    if fed_config.enabled and fed_config.remote_dashboards:
        import asyncio

        async def fetch_remote(remote: RemoteDashboard):
            client = RemoteDashboardClient(remote)
            sessions = await client.fetch_sessions()
            return (remote, sessions)

        tasks = [fetch_remote(r) for r in fed_config.remote_dashboards]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        lane_index = 1
        for i, remote in enumerate(fed_config.remote_dashboards):
            result = results[i]
            if isinstance(result, Exception):
                sessions = []
                is_online = False
            else:
                _, sessions = result
                is_online = True
                if sort == "name":
                    sessions = sorted(sessions, key=lambda s: s.workspace_name.lower())

            lanes_html += _render_swim_lane(
                lane_id=f"remote-{lane_index}",
                name=remote.name,
                sessions=sessions,
                is_online=is_online,
                is_local=False,
                origin_url=remote.url,
            )
            lane_index += 1

    return HTMLResponse(content=lanes_html)


@app.post("/api/federation/proxy/session/new")
async def proxy_create_session(
    origin: Annotated[str, Query()],
    working_directory: Annotated[str, Form()],
    prompt: Annotated[str, Form()],
):
    """Proxy session creation to a remote dashboard."""
    from .federation.client import RemoteDashboardClient

    fed_config = _get_federation_config()

    # Find the remote dashboard
    remote = None
    for r in fed_config.remote_dashboards:
        if r.url == origin:
            remote = r
            break

    if not remote:
        raise HTTPException(status_code=404, detail="Remote dashboard not found")

    client = RemoteDashboardClient(remote)
    result = await client.create_session(working_directory, prompt)

    if not result:
        raise HTTPException(status_code=502, detail="Failed to create session on remote")

    return RedirectResponse(url="/", status_code=303)


@app.get("/api/sessions/{session_id}/messages-html")
async def api_session_messages_html(session_id: str):
    """API endpoint returning session messages HTML for AJAX updates."""
    store = get_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    loop_prompts = _get_loop_prompts()
    messages_html, queued_count = _render_messages_html(session)
    status_html = _render_session_status_html(session)
    message_form_html = _render_message_form(session)
    loop_controls_html = _render_loop_controls(session, loop_prompts)

    return {
        "messages_html": messages_html,
        "queued_count": queued_count,
        "status_html": status_html,
        "message_form_html": message_form_html,
        "loop_controls_html": loop_controls_html,
        "status": session.status.value,
        "message_count": session.message_count,
        "last_activity": format_time_ago(session.last_activity),
    }


@app.post("/session/{session_id}/loop/enable")
async def enable_loop(
    session_id: str,
    prompt_name: Annotated[str, Form()],
    background_tasks: BackgroundTasks,
):
    """Enable the quality loop for a session with a specific prompt.

    If the session is idle, sends the initial loop prompt immediately.
    """
    from .models import SessionMessage
    from .state_machine import SessionState

    store = get_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.loop_enabled = True
    session.loop_count = 0
    session.loop_prompt_name = prompt_name
    session.loop_started_at = datetime.now(timezone.utc)

    # Get the loop prompt configuration
    loop_prompts = _get_loop_prompts()
    prompt_config = loop_prompts.get(prompt_name, {})
    if isinstance(prompt_config, str):
        loop_prompt = prompt_config
    else:
        loop_prompt = prompt_config.get("prompt", "Continue working on the task.")

    # If session is idle, send the initial prompt immediately
    is_idle = session.state in (SessionState.IDLE, SessionState.READY_FOR_LOOP, SessionState.TURN_COMPLETE)
    if is_idle and session.workspace_root and session.conversation_id:
        session.loop_count = 1

        # Add the loop prompt as a user message so it shows in the conversation
        session.messages.append(
            SessionMessage(role="user", content=f"🔄 **Loop Started ({prompt_name}):**\n{loop_prompt}")
        )
        store.upsert_session(session)

        # Spawn auggie with the loop prompt in background
        background_tasks.add_task(
            spawn_auggie_message,
            session.conversation_id,
            session.workspace_root,
            loop_prompt,
        )
    else:
        store.upsert_session(session)

    return RedirectResponse(url=f"/session/{session_id}", status_code=303)


@app.post("/session/{session_id}/loop/pause")
async def pause_loop(session_id: str):
    """Pause the quality loop for a session."""
    store = get_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.loop_enabled = False
    store.upsert_session(session)

    return RedirectResponse(url=f"/session/{session_id}", status_code=303)


@app.post("/session/{session_id}/loop/reset")
async def reset_loop(session_id: str):
    """Reset the loop counter and session state.

    This is a manual recovery mechanism for stuck sessions. It:
    - Resets loop_count to 0
    - Disables the loop
    - Sets session state to IDLE
    - Adds a system message noting the manual reset
    """
    from .models import SessionMessage
    from .state_machine import SessionState

    store = get_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    old_state = session.state
    session.loop_count = 0
    session.loop_enabled = False
    session.state = SessionState.IDLE

    # Add a system message about the reset
    session.messages.append(
        SessionMessage(
            role="system",
            content=f"⚠️ **Session manually reset** (was in state: {old_state.value})"
        )
    )
    session.last_activity = datetime.now(timezone.utc)
    store.upsert_session(session)

    # Process any queued messages now that session is idle
    await process_queued_messages(session_id)

    return RedirectResponse(url=f"/session/{session_id}", status_code=303)


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    """Configuration page for loop prompts."""
    dark_mode = request.query_params.get("dark", None)
    loop_prompts = _get_loop_prompts()
    config = _get_full_config()
    html = render_config_page(dark_mode, loop_prompts, config)
    return HTMLResponse(content=html)


@app.post("/config/prompts/add")
async def add_prompt(
    name: Annotated[str, Form()],
    prompt: Annotated[str, Form()],
    end_condition: Annotated[str, Form()] = "",
):
    """Add a new loop prompt with optional end condition."""
    config = _get_full_config()
    loop_prompts = config.get("loop_prompts", DEFAULT_LOOP_PROMPTS.copy())
    loop_prompts[name] = {"prompt": prompt, "end_condition": end_condition}
    config["loop_prompts"] = loop_prompts
    _save_full_config(config)
    return RedirectResponse(url="/config", status_code=303)


@app.post("/config/prompts/delete")
async def delete_prompt(name: Annotated[str, Form()]):
    """Delete a loop prompt."""
    config = _get_full_config()
    loop_prompts = config.get("loop_prompts", DEFAULT_LOOP_PROMPTS.copy())
    if name in loop_prompts:
        del loop_prompts[name]
    config["loop_prompts"] = loop_prompts
    _save_full_config(config)
    return RedirectResponse(url="/config", status_code=303)


@app.post("/config/prompts/edit")
async def edit_prompt(
    name: Annotated[str, Form()],
    prompt: Annotated[str, Form()],
    end_condition: Annotated[str, Form()] = "",
):
    """Edit an existing loop prompt and end condition."""
    config = _get_full_config()
    loop_prompts = config.get("loop_prompts", DEFAULT_LOOP_PROMPTS.copy())
    loop_prompts[name] = {"prompt": prompt, "end_condition": end_condition}
    config["loop_prompts"] = loop_prompts
    _save_full_config(config)
    return RedirectResponse(url="/config", status_code=303)


@app.post("/config/quick-replies/add")
async def add_quick_reply(
    name: Annotated[str, Form()],
    message: Annotated[str, Form()],
):
    """Add a new quick reply."""
    config = _get_full_config()
    quick_replies = config.get("quick_replies", {})
    quick_replies[name] = message
    config["quick_replies"] = quick_replies
    _save_full_config(config)
    return RedirectResponse(url="/config", status_code=303)


@app.post("/config/quick-replies/delete")
async def delete_quick_reply(name: Annotated[str, Form()]):
    """Delete a quick reply."""
    config = _get_full_config()
    quick_replies = config.get("quick_replies", {})
    if name in quick_replies:
        del quick_replies[name]
    config["quick_replies"] = quick_replies
    _save_full_config(config)
    return RedirectResponse(url="/config", status_code=303)


@app.post("/config/quick-replies/edit")
async def edit_quick_reply(
    name: Annotated[str, Form()],
    message: Annotated[str, Form()],
):
    """Edit an existing quick reply."""
    config = _get_full_config()
    quick_replies = config.get("quick_replies", {})
    if name in quick_replies:
        quick_replies[name] = message
    config["quick_replies"] = quick_replies
    _save_full_config(config)
    return RedirectResponse(url="/config", status_code=303)


@app.post("/config/agent-settings")
async def save_agent_settings(
    agent_timeout_minutes: Annotated[int, Form()],
    max_loop_iterations: Annotated[int, Form()],
):
    """Save agent settings (timeout, loop limits)."""
    config = _get_full_config()
    config["agent_timeout_minutes"] = max(1, min(120, agent_timeout_minutes))
    config["max_loop_iterations"] = max(1, min(500, max_loop_iterations))
    _save_full_config(config)
    return RedirectResponse(url="/config", status_code=303)


@app.post("/config/memory")
async def save_memory_config(
    server_url: Annotated[str, Form()] = "",
    namespace: Annotated[str, Form()] = "augment",
    user_id: Annotated[str, Form()] = "",
    api_key: Annotated[str, Form()] = "",
    auto_capture: Annotated[str | None, Form()] = None,
    auto_recall: Annotated[str | None, Form()] = None,
    use_workspace_namespace: Annotated[str | None, Form()] = None,
    use_persistent_session: Annotated[str | None, Form()] = None,
    track_tool_usage: Annotated[str | None, Form()] = None,
):
    """Save memory server configuration."""
    config = _get_full_config()

    # Build memory config - checkboxes send "true" if checked, None if not
    memory_config = {
        "server_url": server_url.strip(),
        "namespace": namespace.strip() or "augment",
        "user_id": user_id.strip(),
        "api_key": api_key.strip(),
        "auto_capture": auto_capture == "true",
        "auto_recall": auto_recall == "true",
        "use_workspace_namespace": use_workspace_namespace == "true",
        "use_persistent_session": use_persistent_session == "true",
        "track_tool_usage": track_tool_usage == "true",
    }

    config["memory"] = memory_config
    _save_full_config(config)

    return RedirectResponse(url="/config", status_code=303)


@app.post("/config/federation")
async def save_federation_config(
    enabled: Annotated[str | None, Form()] = None,
    share_locally: Annotated[str | None, Form()] = None,
    this_machine_name: Annotated[str, Form()] = "This Machine",
    api_key: Annotated[str, Form()] = "",
):
    """Save federation configuration."""
    config = _get_full_config()

    # Preserve existing remote dashboards
    fed_data = config.get("federation", {})
    existing_remotes = fed_data.get("remote_dashboards", [])

    fed_config = {
        "enabled": enabled == "true",
        "share_locally": share_locally == "true",
        "this_machine_name": this_machine_name.strip() or "This Machine",
        "api_key": api_key.strip() or None,
        "remote_dashboards": existing_remotes,
    }

    config["federation"] = fed_config
    _save_full_config(config)

    return RedirectResponse(url="/config", status_code=303)


@app.post("/config/federation/remotes/add")
async def add_remote_dashboard(
    url: Annotated[str, Form()],
    name: Annotated[str, Form()],
    remote_api_key: Annotated[str, Form()] = "",
):
    """Add a remote dashboard."""
    config = _get_full_config()

    fed_data = config.get("federation", {})
    remotes = fed_data.get("remote_dashboards", [])

    # Add new remote
    new_remote = {
        "url": url.strip().rstrip("/"),
        "name": name.strip(),
        "api_key": remote_api_key.strip() or None,
        "is_healthy": True,
    }
    remotes.append(new_remote)

    fed_data["remote_dashboards"] = remotes
    config["federation"] = fed_data
    _save_full_config(config)

    return RedirectResponse(url="/config", status_code=303)


@app.post("/config/federation/remotes/delete")
async def delete_remote_dashboard(
    index: Annotated[int, Form()],
):
    """Delete a remote dashboard by index."""
    config = _get_full_config()

    fed_data = config.get("federation", {})
    remotes = fed_data.get("remote_dashboards", [])

    if 0 <= index < len(remotes):
        remotes.pop(index)
        fed_data["remote_dashboards"] = remotes
        config["federation"] = fed_data
        _save_full_config(config)

    return RedirectResponse(url="/config", status_code=303)


def _get_full_config() -> dict:
    """Get full config from file."""
    import json
    config_path = Path.home() / ".augment" / "dashboard" / "config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text())
        except Exception:
            pass
    return {}


def _get_federation_config() -> FederationConfig:
    """Get federation config from the main config file."""
    config = _get_full_config()
    fed_data = config.get("federation", {})
    return FederationConfig.from_dict(fed_data)


def _save_full_config(config: dict) -> None:
    """Save full config to file."""
    import json
    config_dir = Path.home() / ".augment" / "dashboard"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))


def _get_pending_prompts_path() -> Path:
    """Get path to pending prompts file."""
    return Path.home() / ".augment" / "dashboard" / "pending_prompts.json"


def _save_pending_prompt(workspace_root: str, prompt: str) -> None:
    """Save a pending initial prompt for a workspace.

    When we spawn a new session, we don't have a session_id yet.
    We save the prompt keyed by workspace_root so the SessionStart hook
    can pick it up and add it as the initial user message.
    """
    import json
    from datetime import datetime, timezone

    path = _get_pending_prompts_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    pending = {}
    if path.exists():
        try:
            pending = json.loads(path.read_text())
        except Exception:
            pass

    # Store with timestamp so we can clean up old entries
    pending[workspace_root] = {
        "prompt": prompt,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(pending, indent=2))


def get_and_clear_pending_prompt(workspace_root: str) -> str | None:
    """Get and clear a pending initial prompt for a workspace.

    Called by SessionStart hook to retrieve the initial user message.
    Returns None if no pending prompt exists.
    """
    import json
    from datetime import datetime, timedelta, timezone

    path = _get_pending_prompts_path()
    if not path.exists():
        return None

    try:
        pending = json.loads(path.read_text())
    except Exception:
        return None

    if workspace_root not in pending:
        return None

    entry = pending[workspace_root]
    prompt = entry.get("prompt")
    timestamp_str = entry.get("timestamp")

    # Check if the prompt is recent (within last 5 minutes)
    if timestamp_str:
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - timestamp
            if age > timedelta(minutes=5):
                # Too old, discard it
                del pending[workspace_root]
                path.write_text(json.dumps(pending, indent=2))
                return None
        except Exception:
            pass

    # Clear the pending prompt
    del pending[workspace_root]
    path.write_text(json.dumps(pending, indent=2))

    return prompt


def _get_recent_working_directories(limit: int = 5) -> list[str]:
    """Get the most recent working directories from sessions and config.

    Returns a list of unique working directories, most recent first.
    """
    store = get_store()
    sessions = store.get_all_sessions()

    # Get directories from sessions (already sorted by last_activity)
    seen = set()
    directories = []
    for session in sessions:
        workspace = session.workspace_root
        if workspace and workspace not in seen:
            seen.add(workspace)
            directories.append(workspace)
            if len(directories) >= limit:
                break

    return directories


def _add_recent_working_directory(directory: str) -> None:
    """Add a directory to recent working directories in config.

    Keeps only the last 10 unique directories.
    """
    config = _get_full_config()
    recent = config.get("recent_directories", [])

    # Remove if already exists (will add to front)
    if directory in recent:
        recent.remove(directory)

    # Add to front
    recent.insert(0, directory)

    # Keep only last 10
    config["recent_directories"] = recent[:10]
    _save_full_config(config)


# Browser notification support - stores pending notifications for polling
_pending_notifications: list[dict] = []


@app.post("/api/notifications/send")
async def send_browser_notification(
    title: Annotated[str, Form()],
    body: Annotated[str, Form()],
    url: Annotated[str, Form()] = "",
):
    """Queue a browser notification for connected clients."""
    notification = {
        "id": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "body": body,
        "url": url,
    }
    _pending_notifications.append(notification)
    # Keep only last 50 notifications
    while len(_pending_notifications) > 50:
        _pending_notifications.pop(0)
    return {"status": "queued"}


@app.get("/api/notifications/poll")
async def poll_notifications(since: str = ""):
    """Poll for new notifications since a given timestamp."""
    if not since:
        return {"notifications": []}
    notifications = [n for n in _pending_notifications if n["id"] > since]
    return {"notifications": notifications}


@app.get("/manifest.json")
async def get_manifest():
    """Serve the PWA manifest."""
    return {
        "name": "Augment Agent Dashboard",
        "short_name": "Augment",
        "description": "Monitor and control Augment agent sessions",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#1a1a2e",
        "theme_color": "#6366f1",
        "icons": [
            {
                "src": "/icon-192.png",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "/icon-512.png",
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    }


@app.get("/sw.js")
async def get_service_worker():
    """Serve the service worker JavaScript."""
    from fastapi.responses import Response
    sw_code = """
// Service Worker for Augment Agent Dashboard PWA
const CACHE_NAME = 'augment-dashboard-v1';

self.addEventListener('install', (event) => {
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(clients.claim());
});

// Handle push notifications
self.addEventListener('push', (event) => {
    let data = { title: 'Augment Agent', body: 'Agent turn complete' };

    if (event.data) {
        try {
            data = event.data.json();
        } catch (e) {
            data.body = event.data.text();
        }
    }

    const options = {
        body: data.body,
        icon: '/icon-192.png',
        badge: '/icon-192.png',
        tag: data.tag || 'augment-notification',
        requireInteraction: true,
        data: { url: data.url || '/' }
    };

    event.waitUntil(
        self.registration.showNotification(data.title, options)
    );
});

// Handle notification click
self.addEventListener('notificationclick', (event) => {
    event.notification.close();

    const url = event.notification.data?.url || '/';

    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true })
            .then((clientList) => {
                // Try to focus an existing window
                for (const client of clientList) {
                    if (client.url.includes(self.location.origin) && 'focus' in client) {
                        client.navigate(url);
                        return client.focus();
                    }
                }
                // Open a new window if none exists
                if (clients.openWindow) {
                    return clients.openWindow(url);
                }
            })
    );
});
"""
    return Response(content=sw_code, media_type="application/javascript")


@app.get("/icon-192.png")
async def get_icon_192():
    """Serve a simple SVG icon as PNG placeholder."""
    from fastapi.responses import Response
    # Simple robot emoji as SVG converted to a basic icon
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 192 192">
        <rect width="192" height="192" fill="#6366f1" rx="32"/>
        <text x="96" y="130" font-size="100" text-anchor="middle">🤖</text>
    </svg>'''
    return Response(content=svg.encode(), media_type="image/svg+xml")


@app.get("/icon-512.png")
async def get_icon_512():
    """Serve a simple SVG icon as PNG placeholder."""
    from fastapi.responses import Response
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
        <rect width="512" height="512" fill="#6366f1" rx="64"/>
        <text x="256" y="340" font-size="280" text-anchor="middle">🤖</text>
    </svg>'''
    return Response(content=svg.encode(), media_type="image/svg+xml")


# HTML rendering functions (inline for simplicity)
def get_base_styles(dark_mode: str | None) -> str:
    """Get CSS styles with dark/light mode support."""
    # If dark_mode is explicitly set, use that; otherwise use system preference
    if dark_mode == "true":
        color_scheme = "dark"
    elif dark_mode == "false":
        color_scheme = "light"
    else:
        color_scheme = "auto"

    # Determine color scheme value for CSS
    if color_scheme == "dark":
        scheme_val = "dark"
    elif color_scheme == "light":
        scheme_val = "light"
    else:
        scheme_val = "light dark"

    # Set colors based on mode
    bg_primary = "#1a1a2e" if color_scheme == "dark" else "#ffffff"
    bg_secondary = "#16213e" if color_scheme == "dark" else "#f5f5f5"
    text_primary = "#eee" if color_scheme == "dark" else "#333"
    text_secondary = "#aaa" if color_scheme == "dark" else "#666"
    border_color = "#333" if color_scheme == "dark" else "#ddd"

    # Build dark mode media query for auto mode
    dark_media = ""
    if color_scheme == "auto":
        dark_media = """
        @media (prefers-color-scheme: dark) {
            :root {
                --bg-primary: #1a1a2e;
                --bg-secondary: #16213e;
                --text-primary: #eee;
                --text-secondary: #aaa;
                --border-color: #333;
                --bg-hover: #1f2937;
            }
        }
        """

    return f"""
    <style>
        :root {{
            color-scheme: {scheme_val};
            --bg-primary: {bg_primary};
            --bg-secondary: {bg_secondary};
            --text-primary: {text_primary};
            --text-secondary: {text_secondary};
            --border-color: {border_color};
            --bg-hover: {"#1f2937" if color_scheme == "dark" else "#e5e7eb"};
            --accent: #4a9eff;
            --accent-color: #4a9eff;
            /* Legacy status colors */
            --status-active: #4ade80;
            --status-idle: #fbbf24;
            --status-stopped: #94a3b8;
            /* State machine colors */
            --state-idle: #94a3b8;
            --state-active: #4ade80;
            --state-turn_complete: #60a5fa;
            --state-review_pending: #f472b6;
            --state-under_review: #c084fc;
            --state-ready_for_loop: #22d3ee;
            --state-loop_prompting: #a78bfa;
            --state-error: #f87171;
        }}
        {dark_media}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 12px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{ font-size: 1.4em; }}
        h2 {{ font-size: 1.2em; }}
        h1, h2 {{ margin-bottom: 15px; }}
        a {{ color: var(--accent); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .header {{
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .header > div {{ font-size: 0.9em; }}
        .session-list {{ display: grid; gap: 12px; }}
        .session-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 12px;
            align-items: start;
        }}
        .session-card:hover {{ border-color: var(--accent); }}
        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-top: 5px;
        }}
        .status-active {{ background: var(--status-active); }}
        .status-idle {{ background: var(--status-idle); }}
        .status-stopped {{ background: var(--status-stopped); }}
        /* State machine dot colors */
        .state-idle {{ background: var(--state-idle); }}
        .state-active {{ background: var(--state-active); }}
        .state-turn_complete {{ background: var(--state-turn_complete); }}
        .state-review_pending {{ background: var(--state-review_pending); }}
        .state-under_review {{ background: var(--state-under_review); }}
        .state-ready_for_loop {{ background: var(--state-ready_for_loop); }}
        .state-loop_prompting {{ background: var(--state-loop_prompting); }}
        .state-error {{ background: var(--state-error); }}
        /* State badge styles */
        .state-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
        }}
        .state-badge .state-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-top: 0;
        }}
        .state-badge.badge-idle {{
            background: rgba(148, 163, 184, 0.2); color: var(--state-idle);
        }}
        .state-badge.badge-active {{
            background: rgba(74, 222, 128, 0.2); color: var(--state-active);
        }}
        .state-badge.badge-turn_complete {{
            background: rgba(96, 165, 250, 0.2); color: var(--state-turn_complete);
        }}
        .state-badge.badge-review_pending {{
            background: rgba(244, 114, 182, 0.2); color: var(--state-review_pending);
        }}
        .state-badge.badge-under_review {{
            background: rgba(192, 132, 252, 0.2); color: var(--state-under_review);
        }}
        .state-badge.badge-ready_for_loop {{
            background: rgba(34, 211, 238, 0.2); color: var(--state-ready_for_loop);
        }}
        .state-badge.badge-loop_prompting {{
            background: rgba(167, 139, 250, 0.2); color: var(--state-loop_prompting);
        }}
        .state-badge.badge-error {{
            background: rgba(248, 113, 113, 0.2); color: var(--state-error);
        }}
        .notification-banner {{
            display: none;
            background: var(--accent);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            cursor: pointer;
        }}
        .new-session-form-container {{
            display: none;
            margin-top: 15px;
            background: var(--card-bg);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid var(--border);
        }}
        .btn-new-session {{
            background: var(--accent);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
        }}
        .form-input {{
            width: 100%;
            padding: 10px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--bg);
            color: var(--text);
            box-sizing: border-box;
        }}
        .form-textarea {{
            width: 100%;
            padding: 10px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--bg);
            color: var(--text);
            resize: vertical;
            box-sizing: border-box;
        }}
        .btn-submit {{
            background: var(--accent);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
        }}
        .btn-disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            background: var(--border-color);
            color: var(--text-secondary);
            border: none;
            border-radius: 8px;
        }}
        .no-sessions {{
            color: var(--text-secondary);
            text-align: center;
            padding: 20px;
        }}
        .modal-input {{
            width: 100%;
            padding: 10px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background: var(--bg-primary);
            color: var(--text-primary);
            box-sizing: border-box;
        }}
        .modal-textarea {{
            width: 100%;
            padding: 10px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background: var(--bg-primary);
            color: var(--text-primary);
            resize: vertical;
            box-sizing: border-box;
        }}
        .btn-cancel {{
            flex: 1;
            padding: 10px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background: transparent;
            color: var(--text-primary);
            cursor: pointer;
        }}
        .btn-start {{
            flex: 1;
            background: var(--accent);
            color: white;
            border: none;
            padding: 10px;
            border-radius: 6px;
            cursor: pointer;
        }}
        .btn-small {{
            padding: 4px 8px;
            font-size: 0.8em;
        }}
        .inline-form {{
            display: inline;
        }}
        .queue-count {{
            color: var(--text-secondary);
            font-size: 0.9em;
        }}
        .status-banner {{
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 10px;
        }}
        .status-banner.status-active {{
            background: var(--status-active);
            color: #000;
        }}
        .message-form {{
            margin-bottom: 10px;
        }}
        .machine-badge-inline {{
            display: inline-block;
            background: var(--accent);
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            margin-bottom: 8px;
        }}
        .remote-session-badge {{
            background: var(--border-color);
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 10px;
        }}
        .session-info {{ overflow: hidden; min-width: 0; }}
        .session-info h3 {{
            font-size: 1em;
            margin-bottom: 4px;
            word-break: break-word;
        }}
        .session-info .workspace {{
            color: var(--text-secondary);
            font-size: 0.8em;
            word-break: break-all;
        }}
        .session-info .preview {{
            color: var(--text-secondary);
            font-size: 0.8em;
            margin-top: 6px;
        }}
        .session-meta {{
            font-size: 0.8em;
            color: var(--text-secondary);
            margin-top: 8px;
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }}
        .message-list {{
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin: 15px 0;
        }}
        .message {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
        }}
        .message.user {{ border-left: 3px solid var(--accent); }}
        .message.assistant {{ border-left: 3px solid var(--status-active); }}
        .message.dashboard {{
            border-left: 3px solid var(--status-idle);
            background: #2a2a1e;
        }}
        .message-header {{
            font-size: 0.8em;
            color: var(--text-secondary);
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .message-header-info {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        .copy-btn {{
            background: transparent;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 0.85em;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        .copy-btn:hover {{
            background: var(--bg-secondary);
            color: var(--text-primary);
            border-color: var(--accent);
        }}
        .copy-btn.copied {{
            background: var(--status-active);
            color: white;
            border-color: var(--status-active);
        }}
        .message-content {{
            word-break: break-word;
            font-size: 0.9em;
            overflow-x: auto;
        }}
        .message-content p {{
            margin: 0.5em 0;
        }}
        .message-content table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
            font-size: 0.9em;
        }}
        .message-content th, .message-content td {{
            border: 1px solid var(--border-color);
            padding: 8px 12px;
            text-align: left;
        }}
        .message-content th {{
            background: var(--bg-primary);
            font-weight: 600;
        }}
        .message-content tr:nth-child(even) {{
            background: rgba(255,255,255,0.03);
        }}
        .message-content code {{
            background: var(--bg-primary);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.9em;
        }}
        .message-content pre {{
            background: var(--bg-primary);
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 1em 0;
        }}
        .message-content pre code {{
            padding: 0;
            background: none;
        }}
        .message-content h1, .message-content h2, .message-content h3 {{
            margin: 1em 0 0.5em 0;
        }}
        .message-content h2 {{
            font-size: 1.2em;
        }}
        .message-content h3 {{
            font-size: 1.1em;
        }}
        .message-content ul, .message-content ol {{
            margin: 0.5em 0;
            padding-left: 1.5em;
        }}
        .message-content a {{
            color: var(--accent);
        }}
        .message-form {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }}
        .message-form h3 {{ font-size: 1em; margin-bottom: 8px; }}
        .message-form textarea {{
            width: 100%;
            min-height: 80px;
            padding: 10px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: inherit;
            font-size: 16px;
            resize: vertical;
        }}
        .message-form button {{
            margin-top: 10px;
            padding: 12px 20px;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1em;
            width: 100%;
        }}
        .message-form button:hover {{ opacity: 0.9; }}
        .back-link {{
            margin-bottom: 15px;
            display: inline-block;
            padding: 8px 0;
        }}
        .pending-messages {{
            background: #2a2a1e;
            border: 1px solid var(--status-idle);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 15px;
        }}
        .pending-messages h4 {{
            color: var(--status-idle);
            margin-bottom: 8px;
            font-size: 0.9em;
        }}
        .pending-messages ul {{ padding-left: 20px; font-size: 0.9em; }}
        .empty-state {{
            text-align: center;
            padding: 30px 15px;
            color: var(--text-secondary);
        }}
        .session-detail-meta {{
            color: var(--text-secondary);
            margin-bottom: 15px;
            font-size: 0.85em;
            word-break: break-all;
        }}
        .loop-controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
            margin-top: 8px;
        }}
        .loop-controls form {{
            display: inline-flex;
            gap: 4px;
            align-items: center;
        }}
        .loop-controls select {{
            padding: 8px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: 14px;
            max-width: 150px;
        }}
        .loop-controls button {{
            padding: 8px 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            white-space: nowrap;
        }}
        .btn-enable {{ background: var(--status-active); color: #000; }}
        .btn-pause {{ background: #fbbf24; color: #000; }}
        .btn-reset {{ background: var(--text-secondary); color: #fff; }}
        .btn-delete {{ background: #dc2626; color: white; }}
        .btn-queue {{ background: #8b5cf6; color: white; }}
        .loop-controls-container {{
            margin-top: 8px;
        }}
        .loop-end-condition {{
            margin-top: 8px;
            padding: 8px 12px;
            background: rgba(99, 102, 241, 0.1);
            border-radius: 6px;
            border-left: 3px solid var(--accent);
        }}
        .end-condition-label {{
            color: var(--text-secondary);
            font-size: 0.85em;
            display: block;
            margin-bottom: 4px;
        }}
        .end-condition-text {{
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
            font-size: 0.9em;
            color: var(--accent);
            background: var(--bg-primary);
            padding: 4px 8px;
            border-radius: 4px;
            display: inline-block;
            word-break: break-word;
        }}
        .loop-prompt-details {{
            margin-top: 8px;
        }}
        .loop-prompt-details summary {{
            cursor: pointer;
            color: var(--text-secondary);
            font-size: 0.85em;
            padding: 4px 0;
        }}
        .loop-prompt-details summary:hover {{
            color: var(--accent);
        }}
        .loop-prompt-text {{
            margin-top: 8px;
            padding: 10px;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 0.85em;
            color: var(--text-secondary);
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 200px;
            overflow-y: auto;
        }}
        .loop-prompt-preview {{
            margin-top: 8px;
            padding: 10px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 0.85em;
        }}
        .message.queued {{
            border-left: 3px solid #8b5cf6;
            background: linear-gradient(90deg, rgba(139, 92, 246, 0.1) 0%, transparent 100%);
            opacity: 0.85;
        }}
        .message.queued .message-header {{
            color: #8b5cf6;
        }}
        .queue-actions {{
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }}
        .nav-links {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .nav-links a {{
            padding: 6px 10px;
            background: var(--bg-secondary);
            border-radius: 4px;
            font-size: 0.85em;
        }}

        /* Mobile-first: stack everything vertically */
        @media (max-width: 599px) {{
            .header h1 {{ font-size: 1.2em; }}
            .session-card {{
                grid-template-columns: 1fr;
                gap: 8px;
            }}
            .status-dot {{
                position: absolute;
                top: 12px;
                right: 12px;
            }}
            .session-card {{
                position: relative;
            }}
            .message-content table {{
                display: block;
                overflow-x: auto;
                white-space: nowrap;
                font-size: 0.8em;
            }}
            .message-content th, .message-content td {{
                padding: 6px 8px;
            }}
            .loop-controls {{
                flex-direction: column;
                align-items: stretch;
            }}
            .loop-controls form {{
                width: 100%;
            }}
            .loop-controls select {{
                flex: 1;
                max-width: none;
            }}
            .loop-controls button {{
                flex-shrink: 0;
            }}
        }}

        /* Tablet and up */
        @media (min-width: 600px) {{
            body {{ padding: 20px; }}
            h1 {{ font-size: 1.6em; }}
            .header {{
                flex-direction: row;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 30px;
            }}
            .session-card {{
                grid-template-columns: auto 1fr auto;
                padding: 15px 20px;
                gap: 15px;
                align-items: center;
            }}
            .status-dot {{ margin-top: 0; }}
            .session-meta {{
                text-align: right;
                margin-top: 0;
                flex-direction: column;
                gap: 4px;
            }}
            .message-form button {{ width: auto; }}
        }}

        /* Pull-to-refresh for mobile */
        .pull-to-refresh {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--bg-secondary);
            transform: translateY(-100%);
            transition: transform 0.2s ease-out;
            z-index: 1000;
            pointer-events: none;
        }}
        .pull-to-refresh.pulling {{
            transform: translateY(calc(var(--pull-progress, 0) * 100% - 100%));
        }}
        .pull-to-refresh.refreshing {{
            transform: translateY(0);
        }}
        .pull-to-refresh-spinner {{
            width: 24px;
            height: 24px;
            border: 3px solid var(--border-color);
            border-top-color: var(--accent);
            border-radius: 50%;
            opacity: var(--pull-progress, 0);
        }}
        .pull-to-refresh.refreshing .pull-to-refresh-spinner {{
            animation: spin 0.8s linear infinite;
            opacity: 1;
        }}
        .pull-to-refresh-text {{
            margin-left: 10px;
            font-size: 0.85em;
            color: var(--text-secondary);
            opacity: var(--pull-progress, 0);
        }}
        .pull-to-refresh.refreshing .pull-to-refresh-text {{
            opacity: 1;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        body.ptr-pulling {{
            transform: translateY(calc(var(--pull-distance, 0px)));
            transition: none;
        }}
        body.ptr-refreshing {{
            transform: translateY(60px);
            transition: transform 0.2s ease-out;
        }}
    </style>
    """


def _get_notification_script() -> str:
    """Get JavaScript for browser notifications with PWA support for iOS."""
    return """
        // Browser notification support with PWA for iOS
        const banner = document.getElementById('notification-banner');
        const bannerText = document.getElementById('notification-text');
        let lastNotificationId = new Date().toISOString();
        let swRegistration = null;

        // Detect iOS
        function isIOS() {
            return /iPad|iPhone|iPod/.test(navigator.userAgent) ||
                   (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
        }

        // Detect if running as installed PWA
        function isPWA() {
            return window.matchMedia('(display-mode: standalone)').matches ||
                   window.navigator.standalone === true;
        }

        function checkNotificationSupport() {
            return 'Notification' in window;
        }

        // Register service worker
        async function registerServiceWorker() {
            if ('serviceWorker' in navigator) {
                try {
                    swRegistration = await navigator.serviceWorker.register('/sw.js');
                    console.log('Service worker registered');
                    return true;
                } catch (e) {
                    console.error('Service worker registration failed:', e);
                    return false;
                }
            }
            return false;
        }

        function updateBanner() {
            // iOS but not installed as PWA
            if (isIOS() && !isPWA()) {
                banner.style.display = 'block';
                banner.style.background = 'var(--accent)';
                bannerText.innerHTML = '📱 <strong>Add to Home Screen</strong>: ' +
                    'tap ⎙ then "Add to Home Screen"';
                banner.onclick = null;
                banner.style.cursor = 'default';
                return;
            }

            // iOS PWA - can use notifications
            if (isIOS() && isPWA()) {
                if (!checkNotificationSupport()) {
                    banner.style.display = 'block';
                    banner.style.background = 'var(--text-secondary)';
                    bannerText.textContent = 'Notifications not available - try updating iOS';
                    return;
                }
            }

            // Standard notification flow
            if (!checkNotificationSupport()) {
                banner.style.display = 'none';
                return;
            }

            if (Notification.permission === 'default') {
                banner.style.display = 'block';
                bannerText.textContent = '🔔 Click to enable notifications for agent alerts';
                banner.onclick = requestPermission;
                banner.style.cursor = 'pointer';
            } else if (Notification.permission === 'granted') {
                banner.style.display = 'block';
                banner.style.background = 'var(--status-active)';
                banner.style.color = '#000';
                bannerText.textContent = '✓ Browser notifications enabled';
                banner.onclick = null;
                banner.style.cursor = 'default';
                // Start polling for notifications
                pollNotifications();
            } else {
                banner.style.display = 'block';
                banner.style.background = 'var(--text-secondary)';
                bannerText.textContent = 'Notifications blocked - enable in browser settings';
                banner.onclick = null;
            }
        }

        async function requestPermission() {
            // Register service worker first for PWA support
            await registerServiceWorker();

            const permission = await Notification.requestPermission();
            updateBanner();
        }

        async function pollNotifications() {
            try {
                const url = '/api/notifications/poll?since=' +
                    encodeURIComponent(lastNotificationId);
                const response = await fetch(url);
                const data = await response.json();
                for (const n of data.notifications) {
                    showNotification(n);
                    lastNotificationId = n.id;
                }
            } catch (e) {
                console.error('Notification poll error:', e);
            }
            // Poll every 3 seconds
            setTimeout(pollNotifications, 3000);
        }

        function showNotification(n) {
            if (Notification.permission !== 'granted') return;

            // Use service worker notification if available (better for PWA)
            if (swRegistration && swRegistration.showNotification) {
                swRegistration.showNotification(n.title, {
                    body: n.body,
                    icon: '/icon-192.png',
                    tag: n.id,
                    requireInteraction: true,
                    data: { url: n.url }
                });
            } else {
                // Fallback to regular notification
                const notification = new Notification(n.title, {
                    body: n.body,
                    icon: '/icon-192.png',
                    tag: n.id,
                    requireInteraction: true
                });
                if (n.url) {
                    notification.onclick = () => {
                        window.focus();
                        window.location.href = n.url;
                    };
                }
            }
        }

        // Initialize
        registerServiceWorker().then(() => updateBanner());
    """


def _get_pull_to_refresh_script() -> str:
    """Get JavaScript for pull-to-refresh on mobile."""
    return """
        // Pull-to-refresh for mobile
        (function() {
            const ptrEl = document.getElementById('pull-to-refresh');
            if (!ptrEl) return;

            const ptrSpinner = ptrEl.querySelector('.pull-to-refresh-spinner');
            const ptrText = ptrEl.querySelector('.pull-to-refresh-text');

            let startY = 0;
            let currentY = 0;
            let isPulling = false;
            let isRefreshing = false;
            const threshold = 80;  // Pull distance to trigger refresh
            const maxPull = 120;   // Max pull distance

            function isTouchDevice() {
                return 'ontouchstart' in window || navigator.maxTouchPoints > 0;
            }

            // Only enable on touch devices
            if (!isTouchDevice()) return;

            document.addEventListener('touchstart', function(e) {
                if (isRefreshing) return;
                // Only trigger at top of page
                if (window.scrollY > 5) return;

                startY = e.touches[0].pageY;
                isPulling = true;
            }, { passive: true });

            document.addEventListener('touchmove', function(e) {
                if (!isPulling || isRefreshing) return;
                if (window.scrollY > 5) {
                    isPulling = false;
                    return;
                }

                currentY = e.touches[0].pageY;
                const pullDistance = Math.min(currentY - startY, maxPull);

                if (pullDistance > 0) {
                    const progress = Math.min(pullDistance / threshold, 1);
                    ptrEl.style.setProperty('--pull-progress', progress);
                    ptrEl.classList.add('pulling');
                    document.body.style.setProperty('--pull-distance', pullDistance * 0.4 + 'px');
                    document.body.classList.add('ptr-pulling');

                    if (pullDistance >= threshold) {
                        ptrText.textContent = 'Release to refresh';
                    } else {
                        ptrText.textContent = 'Pull to refresh';
                    }
                }
            }, { passive: true });

            document.addEventListener('touchend', function(e) {
                if (!isPulling) return;
                isPulling = false;

                const pullDistance = currentY - startY;

                if (pullDistance >= threshold && !isRefreshing) {
                    // Trigger refresh
                    isRefreshing = true;
                    ptrEl.classList.remove('pulling');
                    ptrEl.classList.add('refreshing');
                    document.body.classList.remove('ptr-pulling');
                    document.body.classList.add('ptr-refreshing');
                    ptrText.textContent = 'Refreshing...';

                    // Reload the page
                    setTimeout(() => {
                        location.reload();
                    }, 300);
                } else {
                    // Reset
                    ptrEl.classList.remove('pulling');
                    ptrEl.style.setProperty('--pull-progress', 0);
                    document.body.classList.remove('ptr-pulling');
                    document.body.style.setProperty('--pull-distance', '0px');
                }

                startY = 0;
                currentY = 0;
            }, { passive: true });
        })();
    """


def _get_timestamp_script() -> str:
    """Get JavaScript to localize UTC timestamps to local timezone on hover."""
    return """
        function localizeTimestamps() {
            document.querySelectorAll('.timestamp[data-utc]').forEach(el => {
                const utc = el.dataset.utc;
                if (utc && !el.dataset.localized) {
                    const date = new Date(utc);
                    const options = {
                        weekday: 'short',
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                        timeZoneName: 'short'
                    };
                    el.title = date.toLocaleString(undefined, options);
                    el.dataset.localized = 'true';
                }
            });
        }

        // Run on page load
        localizeTimestamps();

        // Re-run after AJAX updates (observe DOM changes)
        const timestampObserver = new MutationObserver(() => localizeTimestamps());
        timestampObserver.observe(document.body, { childList: true, subtree: true });
    """


def format_time_ago(dt: datetime, include_title: bool = False) -> str:
    """Format a datetime as a human-readable relative time string.

    Args:
        dt: The datetime to format
        include_title: If True, wrap in span with full datetime as title for hover

    Returns:
        Relative time string, optionally wrapped in span with hover title
    """
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    diff = now - dt
    seconds = diff.total_seconds()

    if seconds < 60:
        relative = "just now"
    elif seconds < 3600:
        mins = int(seconds / 60)
        relative = f"{mins}m ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        relative = f"{hours}h ago"
    elif seconds < 172800:  # 2 days
        relative = "yesterday"
    elif seconds < 604800:  # 7 days
        days = int(seconds / 86400)
        relative = f"{days} days ago"
    else:
        weeks = int(seconds / 604800)
        if weeks == 1:
            relative = "a week ago"
        else:
            relative = f"{weeks} weeks ago"

    if include_title:
        # Full datetime for hover - will be converted to local time via JS
        iso_str = dt.isoformat()
        return f'<span class="timestamp" data-utc="{iso_str}" title="{iso_str}">{relative}</span>'
    return relative


def _render_session_cards(sessions: list) -> str:
    """Render just the session cards HTML for AJAX updates."""
    if not sessions:
        return (
            '<div class="empty-state">'
            "No active sessions. Start an Augment conversation to see it here."
            "</div>"
        )

    session_cards = ""
    for s in sessions:
        # Get state for styling (fall back to status)
        try:
            state_value = s.state.value
        except (AttributeError, ValueError):
            state_value = s.status.value

        state_class = f"state-{state_value}"
        state_label = _get_state_label(state_value)
        preview = s.last_message_preview or "No messages yet"
        time_ago = format_time_ago(s.last_activity, include_title=True)

        ellipsis = "..." if len(preview) > 80 else ""
        session_cards += f"""
        <a href="/session/{s.session_id}" class="session-card">
            <div class="status-dot {state_class}" title="{state_label}"></div>
            <div class="session-info">
                <h3>{s.workspace_name}</h3>
                <div class="workspace">{s.workspace_root}</div>
                <div class="preview">{preview[:80]}{ellipsis}</div>
            </div>
            <div class="session-meta">
                <div>{s.message_count} messages</div>
                <div>{time_ago}</div>
            </div>
        </a>
        """
    return session_cards


def _render_recent_directories_html() -> str:
    """Render the recent directories picker HTML."""
    recent_dirs = _get_recent_working_directories(limit=5)
    if not recent_dirs:
        return ""

    dirs_html = ""
    for directory in recent_dirs:
        escaped_dir = html.escape(directory)
        # Show shortened path for display
        display_dir = directory
        if len(display_dir) > 40:
            display_dir = "..." + display_dir[-37:]
        escaped_display = html.escape(display_dir)
        dirs_html += f'''
            <button type="button" class="recent-dir-btn"
                onclick="selectRecentDir('{escaped_dir}')"
                title="{escaped_dir}">📁 {escaped_display}</button>
        '''

    return f'''
        <div class="recent-dirs-section">
            <label class="field-label">Recent Directories:</label>
            <div class="recent-dirs-list">{dirs_html}</div>
        </div>
    '''


def _get_recent_dirs_styles() -> str:
    """Get CSS styles for recent directories picker."""
    return """
        .recent-dirs-section {
            margin-bottom: 12px;
        }
        .recent-dirs-list {
            display: flex;
            flex-direction: column;
            gap: 4px;
            margin-top: 4px;
        }
        .recent-dir-btn {
            text-align: left;
            padding: 6px 10px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            color: var(--text-primary);
            font-size: 13px;
            cursor: pointer;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .recent-dir-btn:hover {
            background: var(--bg-hover);
            border-color: var(--accent-color);
        }
    """


def render_dashboard(sessions: list, dark_mode: str | None, sort_by: str = "recent") -> str:
    """Render the main dashboard HTML."""
    styles = get_base_styles(dark_mode)
    recent_dirs_styles = _get_recent_dirs_styles()
    recent_dirs_html = _render_recent_directories_html()

    session_cards = _render_session_cards(sessions)

    # Build sort links preserving dark mode
    dark_param = f"&dark={dark_mode}" if dark_mode else ""
    recent_active = "font-weight:bold;" if sort_by == "recent" else ""
    name_active = "font-weight:bold;" if sort_by == "name" else ""

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Augment Agent Dashboard</title>
        <link rel="manifest" href="/manifest.json">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <meta name="apple-mobile-web-app-title" content="Augment">
        <link rel="apple-touch-icon" href="/icon-192.png">
        <meta name="theme-color" content="#6366f1">
        {styles}
        <style>{recent_dirs_styles}</style>
    </head>
    <body>
        <div id="pull-to-refresh" class="pull-to-refresh">
            <div class="pull-to-refresh-spinner"></div>
            <span class="pull-to-refresh-text">Pull to refresh</span>
        </div>
        <div class="header">
            <h1>🤖 Augment Agent Dashboard</h1>
            <div class="nav-links">
                <a href="?sort=recent{dark_param}" style="{recent_active}">Recent</a>
                <a href="?sort=name{dark_param}" style="{name_active}">Name</a>
                <a href="?dark=true&sort={sort_by}">🌙</a>
                <a href="?dark=false&sort={sort_by}">☀️</a>
                <a href="/config">⚙️ Config</a>
            </div>
        </div>
        <div id="notification-banner" class="notification-banner">
            🔔 <span id="notification-text">Enable notifications for alerts</span>
        </div>
        <div class="new-session-section" style="margin-bottom:20px;">
            <button onclick="toggleNewSession()" class="btn-new-session">
                ➕ New Session
            </button>
            <div id="new-session-form" class="new-session-form-container">
                <form method="POST" action="/session/new">
                    {recent_dirs_html}
                    <div style="margin-bottom:15px;">
                        <label for="working_directory" class="field-label">
                            Working Directory
                        </label>
                        <input type="text" id="working_directory" name="working_directory"
                            placeholder="/path/to/project" class="form-input">
                    </div>
                    <div style="margin-bottom:15px;">
                        <label for="prompt" class="field-label">Initial Prompt</label>
                        <textarea id="prompt" name="prompt" rows="4"
                            placeholder="What would you like the agent to do?"
                            class="form-textarea"></textarea>
                    </div>
                    <button type="submit" class="btn-submit">🚀 Start Session</button>
                </form>
            </div>
        </div>
        <script>
            function toggleNewSession() {{
                const form = document.getElementById('new-session-form');
                form.style.display = form.style.display === 'none' ? 'block' : 'none';
            }}
            function selectRecentDir(dir) {{
                document.getElementById('working_directory').value = dir;
            }}
        </script>
        <div class="session-list" id="session-list">
            {session_cards}
        </div>
        <script>
            {_get_notification_script()}
            {_get_timestamp_script()}
            {_get_pull_to_refresh_script()}

            // AJAX-based session list updates
            const REFRESH_INTERVAL = 5000;
            const sortBy = '{sort_by}';

            // Track scrolling state - pause refresh while scrolling
            let isScrolling = false;
            let scrollTimeout = null;
            const SCROLL_DEBOUNCE = 1500; // Wait 1.5s after scrolling stops before refresh

            // Track scroll on session list and window
            function handleScroll() {{
                isScrolling = true;
                if (scrollTimeout) clearTimeout(scrollTimeout);
                scrollTimeout = setTimeout(() => {{
                    isScrolling = false;
                }}, SCROLL_DEBOUNCE);
            }}

            // Attach scroll listeners
            const sessionList = document.getElementById('session-list');
            if (sessionList) sessionList.addEventListener('scroll', handleScroll);
            window.addEventListener('scroll', handleScroll);

            function isUserInteracting() {{
                // Check if user is scrolling
                if (isScrolling) {{
                    return true;
                }}
                // Check if new session form is visible
                const newSessionForm = document.getElementById('new-session-form');
                if (newSessionForm && newSessionForm.style.display !== 'none') {{
                    return true;
                }}
                // Check if any input/textarea has focus
                const activeEl = document.activeElement;
                const isInput = activeEl && activeEl.tagName === 'INPUT';
                const isTextarea = activeEl && activeEl.tagName === 'TEXTAREA';
                if (isInput || isTextarea) {{
                    return true;
                }}
                return false;
            }}

            async function refreshSessionList() {{
                if (isUserInteracting()) {{
                    // User is interacting, skip this refresh
                    scheduleRefresh();
                    return;
                }}

                // Save scroll position before refresh
                const scrollTop = sessionList ? sessionList.scrollTop : 0;
                const windowScrollY = window.scrollY;

                try {{
                    const url = '/api/sessions-html?sort=' + encodeURIComponent(sortBy);
                    const response = await fetch(url);
                    if (response.ok) {{
                        const html = await response.text();
                        document.getElementById('session-list').innerHTML = html;
                        // Restore scroll position after refresh
                        if (sessionList) sessionList.scrollTop = scrollTop;
                        window.scrollTo(0, windowScrollY);
                    }}
                }} catch (e) {{
                    console.error('Failed to refresh session list:', e);
                }}
                scheduleRefresh();
            }}

            function scheduleRefresh() {{
                setTimeout(refreshSessionList, REFRESH_INTERVAL);
            }}

            // Start the refresh cycle
            scheduleRefresh();
        </script>
    </body>
    </html>
    """


def _get_swimlane_styles() -> str:
    """Get additional CSS for swim lane layout."""
    return """
    <style>
        .swim-lanes-container {
            display: flex;
            gap: 1rem;
            overflow-x: auto;
            scroll-snap-type: x mandatory;
            -webkit-overflow-scrolling: touch;
            padding-bottom: 1rem;
            min-height: calc(100vh - 200px);
        }
        .swim-lane {
            flex: 0 0 340px;
            max-width: 90vw;
            scroll-snap-align: start;
            background: var(--bg-secondary);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            border: 1px solid var(--border-color);
        }
        .swim-lane.offline {
            opacity: 0.7;
        }
        .swim-lane.offline .swim-lane-sessions {
            filter: grayscale(30%);
        }
        .swim-lane-header {
            position: sticky;
            top: 0;
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-secondary);
            border-radius: 12px 12px 0 0;
            z-index: 1;
        }
        .swim-lane-title {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .swim-lane-status {
            font-size: 0.85em;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .swim-lane-status .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        .swim-lane-status .status-indicator.online { background: var(--status-active); }
        .swim-lane-status .status-indicator.offline { background: var(--status-stopped); }
        .swim-lane-sessions {
            flex: 1;
            overflow-y: auto;
            padding: 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            max-height: calc(100vh - 280px);
        }
        .swim-lane .session-card {
            margin: 0;
        }
        .swim-lane .btn-new-session {
            width: 100%;
            margin-top: 8px;
            padding: 8px 12px;
            font-size: 0.9em;
        }
        .swim-lane-indicators {
            display: none;
            justify-content: center;
            gap: 8px;
            padding: 12px 0;
        }
        .swim-lane-indicators .indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--border-color);
            border: none;
            cursor: pointer;
            padding: 0;
        }
        .swim-lane-indicators .indicator.active {
            background: var(--accent);
        }
        @media (min-width: 768px) {
            .swim-lane {
                flex: 1 1 340px;
                max-width: 450px;
            }
        }
        @media (max-width: 767px) {
            .swim-lanes-container {
                padding-left: 0.5rem;
                margin-right: -12px;
                padding-right: 15%;
            }
            .swim-lane {
                flex: 0 0 85vw;
            }
            .swim-lane-indicators {
                display: flex;
            }
        }
        .new-session-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 100;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .new-session-overlay.active {
            display: flex;
        }
        .new-session-modal {
            background: var(--bg-secondary);
            padding: 24px;
            border-radius: 12px;
            max-width: 500px;
            width: 100%;
            border: 1px solid var(--border-color);
        }
        .new-session-modal h3 {
            margin-bottom: 16px;
        }
        .new-session-modal .machine-label {
            color: var(--accent);
            font-size: 0.9em;
            margin-bottom: 16px;
        }
    </style>
    """


def _render_swim_lane(
    lane_id: str,
    name: str,
    sessions: list,
    is_online: bool,
    is_local: bool,
    origin_url: str | None = None,
) -> str:
    """Render a single swim lane with its sessions."""
    status_class = "online" if is_online else "offline"
    status_text = "Online" if is_online else "Offline"
    lane_class = "swim-lane" + (" offline" if not is_online else "")
    session_count = len(sessions)

    # Build session cards for this lane
    session_cards = ""
    for s in sessions:
        # Handle both AgentSession objects and RemoteSession objects
        if hasattr(s, 'status') and hasattr(s.status, 'value'):
            status_val = s.status.value
        elif hasattr(s, 'status'):
            status_val = s.status
        else:
            status_val = "stopped"

        session_id = s.session_id
        workspace_name = html.escape(s.workspace_name)
        preview = html.escape(s.last_message_preview or "No messages yet")[:80]
        msg_count = getattr(s, 'message_count', 0)

        session_cards += f'''
        <a href="/session/{session_id}" class="session-card">
            <div class="status-dot status-{status_val}"></div>
            <div class="session-info">
                <h3>{workspace_name}</h3>
                <div class="preview">{preview}</div>
                <div class="session-meta">
                    <span>{msg_count} messages</span>
                </div>
            </div>
        </a>
        '''

    # New session button - different action for local vs remote
    escaped_name = html.escape(name)
    if is_local:
        new_session_btn = f'''
        <button onclick="openNewSession('local', '{escaped_name}')" class="btn-new-session">
            ➕ New Session
        </button>
        '''
    else:
        escaped_origin = html.escape(origin_url or "")
        if not is_online:
            disabled = ' disabled class="btn-disabled"'
        else:
            onclick = f"openNewSession('{escaped_origin}', '{escaped_name}')"
            disabled = f' onclick="{onclick}" class="btn-new-session"'
        new_session_btn = f'''
        <button{disabled}>
            ➕ New Session
        </button>
        '''

    no_sessions_msg = '<div class="no-sessions">No sessions</div>'
    sessions_html = session_cards if session_cards else no_sessions_msg
    return f'''
    <div class="{lane_class}" data-lane-id="{lane_id}" data-origin="{origin_url or 'local'}">
        <div class="swim-lane-header">
            <div class="swim-lane-title">
                💻 {escaped_name}
            </div>
            <div class="swim-lane-status">
                <span class="status-indicator {status_class}"></span>
                {status_text} · {session_count} session{"s" if session_count != 1 else ""}
            </div>
            {new_session_btn}
        </div>
        <div class="swim-lane-sessions" id="lane-sessions-{lane_id}">
            {sessions_html}
        </div>
    </div>
    '''


def render_dashboard_swimlanes(
    local_sessions: list,
    remote_sessions_by_origin: dict,
    fed_config: FederationConfig,
    dark_mode: str | None,
    sort_by: str = "recent",
) -> str:
    """Render the dashboard with swim lanes for multiple machines."""
    styles = get_base_styles(dark_mode)
    swimlane_styles = _get_swimlane_styles()
    recent_dirs_styles = _get_recent_dirs_styles()
    recent_dirs_html = _render_recent_directories_html()

    dark_param = f"&dark={dark_mode}" if dark_mode else ""
    recent_active = "font-weight:bold;" if sort_by == "recent" else ""
    name_active = "font-weight:bold;" if sort_by == "name" else ""

    # Build swim lanes HTML
    lanes_html = ""
    lane_indicators = ""
    lane_index = 0

    # Local machine lane
    lanes_html += _render_swim_lane(
        lane_id="local",
        name=fed_config.this_machine_name,
        sessions=local_sessions,
        is_online=True,
        is_local=True,
    )
    lane_indicators += f'<button class="indicator active" data-lane="{lane_index}"></button>'
    lane_index += 1

    # Remote machine lanes
    for remote in fed_config.remote_dashboards:
        remote_data = remote_sessions_by_origin.get(remote.url, {})
        sessions = remote_data.get("sessions", []) if remote_data else []
        is_online = remote.is_healthy

        lanes_html += _render_swim_lane(
            lane_id=f"remote-{lane_index}",
            name=remote.name,
            sessions=sessions,
            is_online=is_online,
            is_local=False,
            origin_url=remote.url,
        )
        lane_indicators += f'<button class="indicator" data-lane="{lane_index}"></button>'
        lane_index += 1

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Augment Agent Dashboard</title>
        <link rel="manifest" href="/manifest.json">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <meta name="apple-mobile-web-app-title" content="Augment">
        <link rel="apple-touch-icon" href="/icon-192.png">
        <meta name="theme-color" content="#6366f1">
        {styles}
        {swimlane_styles}
        <style>{recent_dirs_styles}</style>
    </head>
    <body>
        <div id="pull-to-refresh" class="pull-to-refresh">
            <div class="pull-to-refresh-spinner"></div>
            <span class="pull-to-refresh-text">Pull to refresh</span>
        </div>
        <div class="header">
            <h1>🤖 Augment Agent Dashboard</h1>
            <div class="nav-links">
                <a href="?sort=recent{dark_param}" style="{recent_active}">Recent</a>
                <a href="?sort=name{dark_param}" style="{name_active}">Name</a>
                <a href="?dark=true&sort={sort_by}">🌙</a>
                <a href="?dark=false&sort={sort_by}">☀️</a>
                <a href="/config">⚙️ Config</a>
            </div>
        </div>

        <div id="notification-banner" class="notification-banner">
            🔔 <span id="notification-text">Enable browser notifications</span>
        </div>

        <div class="swim-lanes-container" id="swim-lanes">
            {lanes_html}
        </div>

        <div class="swim-lane-indicators">
            {lane_indicators}
        </div>

        <!-- New Session Modal -->
        <div id="new-session-overlay" class="new-session-overlay"
            onclick="if(event.target===this)closeNewSession()">
            <div class="new-session-modal">
                <h3>➕ New Session</h3>
                <div class="machine-label" id="new-session-machine">on: This Machine</div>
                <form id="new-session-form" method="POST" action="/session/new">
                    <input type="hidden" id="new-session-origin" name="origin" value="local">
                    {recent_dirs_html}
                    <div style="margin-bottom:15px;">
                        <label class="field-label">Working Directory</label>
                        <input type="text" id="working_directory" name="working_directory"
                            placeholder="/path/to/project" class="modal-input">
                    </div>
                    <div style="margin-bottom:15px;">
                        <label class="field-label">Initial Prompt</label>
                        <textarea id="prompt" name="prompt" rows="4"
                            placeholder="What would you like the agent to do?"
                            class="modal-textarea"></textarea>
                    </div>
                    <div style="display:flex;gap:10px;">
                        <button type="button" onclick="closeNewSession()"
                            class="btn-cancel">Cancel</button>
                        <button type="submit" class="btn-start">🚀 Start</button>
                    </div>
                </form>
            </div>
        </div>

        <script>
            {_get_notification_script()}
            {_get_timestamp_script()}
            {_get_pull_to_refresh_script()}

            // Swim lane scroll indicator updates
            const swimLanes = document.getElementById('swim-lanes');
            const indicators = document.querySelectorAll('.swim-lane-indicators .indicator');

            if (swimLanes && indicators.length > 0) {{
                swimLanes.addEventListener('scroll', () => {{
                    const scrollLeft = swimLanes.scrollLeft;
                    const laneWidth = swimLanes.querySelector('.swim-lane').offsetWidth + 16;
                    const activeIndex = Math.round(scrollLeft / laneWidth);

                    indicators.forEach((ind, i) => {{
                        ind.classList.toggle('active', i === activeIndex);
                    }});
                }});

                indicators.forEach((ind, i) => {{
                    ind.addEventListener('click', () => {{
                        const laneWidth = swimLanes.querySelector('.swim-lane').offsetWidth + 16;
                        swimLanes.scrollTo({{ left: i * laneWidth, behavior: 'smooth' }});
                    }});
                }});
            }}

            // New session modal
            let currentOrigin = 'local';

            function openNewSession(origin, machineName) {{
                currentOrigin = origin;
                document.getElementById('new-session-machine').textContent = 'on: ' + machineName;
                document.getElementById('new-session-origin').value = origin;

                // Update form action based on origin
                const form = document.getElementById('new-session-form');
                if (origin === 'local') {{
                    form.action = '/session/new';
                }} else {{
                    const baseUrl = '/api/federation/proxy/session/new?origin=';
                    form.action = baseUrl + encodeURIComponent(origin);
                }}

                document.getElementById('new-session-overlay').classList.add('active');
                document.getElementById('working_directory').focus();
            }}

            function closeNewSession() {{
                document.getElementById('new-session-overlay').classList.remove('active');
            }}

            function selectRecentDir(dir) {{
                document.getElementById('working_directory').value = dir;
            }}

            // Close on Escape
            document.addEventListener('keydown', (e) => {{
                if (e.key === 'Escape') closeNewSession();
            }});

            // AJAX refresh for swim lanes
            const REFRESH_INTERVAL = 5000;
            const sortBy = '{sort_by}';

            // Track scrolling state - pause refresh while scrolling
            let isScrolling = false;
            let scrollTimeout = null;
            const SCROLL_DEBOUNCE = 1500; // Wait 1.5s after scrolling stops

            function handleScroll() {{
                isScrolling = true;
                if (scrollTimeout) clearTimeout(scrollTimeout);
                scrollTimeout = setTimeout(() => {{
                    isScrolling = false;
                }}, SCROLL_DEBOUNCE);
            }}

            // Attach scroll listeners to swim lanes container and individual lanes
            const swimLanesContainer = document.querySelector('.swim-lanes-container');
            if (swimLanesContainer) swimLanesContainer.addEventListener('scroll', handleScroll);
            window.addEventListener('scroll', handleScroll);
            // Also track scroll on individual session lists within lanes
            document.querySelectorAll('.session-list').forEach(el => {{
                el.addEventListener('scroll', handleScroll);
            }});

            function isUserInteracting() {{
                // Check if user is scrolling
                if (isScrolling) return true;
                // Check if modal is open
                const overlay = document.getElementById('new-session-overlay');
                if (overlay && overlay.classList.contains('active')) return true;
                // Check if any input/textarea has focus
                const activeEl = document.activeElement;
                if (activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA')) return true;
                return false;
            }}

            async function refreshSwimLanes() {{
                if (isUserInteracting()) {{
                    scheduleRefresh();
                    return;
                }}

                // Save scroll positions before refresh
                const containerScrollLeft = swimLanesContainer ? swimLanesContainer.scrollLeft : 0;
                const windowScrollY = window.scrollY;
                // Save individual lane scroll positions
                const laneScrolls = {{}};
                document.querySelectorAll('.swim-lane').forEach(lane => {{
                    const laneId = lane.dataset.machine || lane.querySelector('h3')?.textContent;
                    const sessionList = lane.querySelector('.session-list');
                    if (laneId && sessionList) {{
                        laneScrolls[laneId] = sessionList.scrollTop;
                    }}
                }});

                try {{
                    const url = '/api/swimlanes-html?sort=' + encodeURIComponent(sortBy);
                    const response = await fetch(url);
                    if (response.ok) {{
                        const html = await response.text();
                        document.getElementById('swim-lanes').innerHTML = html;

                        // Restore scroll positions
                        const newContainer = document.querySelector('.swim-lanes-container');
                        if (newContainer) {{
                            newContainer.scrollLeft = containerScrollLeft;
                            // Re-attach scroll listener to new container
                            newContainer.addEventListener('scroll', handleScroll);
                        }}
                        window.scrollTo(0, windowScrollY);

                        // Restore lane scroll positions
                        document.querySelectorAll('.swim-lane').forEach(lane => {{
                            const laneId = lane.dataset.machine || lane.querySelector('h3')?.textContent;
                            const sessionList = lane.querySelector('.session-list');
                            if (laneId && sessionList && laneScrolls[laneId] !== undefined) {{
                                sessionList.scrollTop = laneScrolls[laneId];
                            }}
                            // Re-attach scroll listener
                            if (sessionList) sessionList.addEventListener('scroll', handleScroll);
                        }});
                    }}
                }} catch (e) {{
                    console.error('Failed to refresh swim lanes:', e);
                }}
                scheduleRefresh();
            }}

            function scheduleRefresh() {{
                setTimeout(refreshSwimLanes, REFRESH_INTERVAL);
            }}

            scheduleRefresh();
        </script>
    </body>
    </html>
    """


def _render_memory_config_section(config: dict) -> str:
    """Render the memory configuration section HTML."""
    memory_config = config.get("memory", {})

    server_url = html.escape(memory_config.get("server_url", ""))
    namespace = html.escape(memory_config.get("namespace", "augment"))
    user_id = html.escape(memory_config.get("user_id", ""))
    api_key = html.escape(memory_config.get("api_key", ""))

    # Boolean options
    auto_capture = memory_config.get("auto_capture", True)
    auto_recall = memory_config.get("auto_recall", True)
    use_workspace_namespace = memory_config.get("use_workspace_namespace", True)
    use_persistent_session = memory_config.get("use_persistent_session", True)
    track_tool_usage = memory_config.get("track_tool_usage", False)

    # Status indicator
    enabled = bool(server_url)
    status_color = "var(--status-idle)" if enabled else "var(--text-secondary)"
    status_text = "Configured" if enabled else "Not configured"

    return f'''
        <p class="section-description">
            Configure the Agent Memory Server to persist context across sessions.
            Settings are applied when hooks run (no restart needed).
        </p>

        <div class="config-card">
            <div class="memory-status">
                <span class="status-dot" style="background:{status_color};"></span>
                <strong>Status: {status_text}</strong>
            </div>

            <form method="POST" action="/config/memory">
                <label class="field-label">Memory Server URL:</label>
                <input type="text" name="server_url" value="{server_url}"
                       placeholder="http://localhost:8000" style="width:100%;padding:8px;
                       border:1px solid var(--border-color);border-radius:4px;
                       background:var(--bg-secondary);color:var(--text-primary);
                       font-size:13px;margin-bottom:8px;">

                <label class="field-label">Namespace:</label>
                <input type="text" name="namespace" value="{namespace}"
                       placeholder="augment" style="width:100%;padding:8px;
                       border:1px solid var(--border-color);border-radius:4px;
                       background:var(--bg-secondary);color:var(--text-primary);
                       font-size:13px;margin-bottom:8px;">

                <label class="field-label">User ID:</label>
                <input type="text" name="user_id" value="{user_id}"
                       placeholder="your-user-id" style="width:100%;padding:8px;
                       border:1px solid var(--border-color);border-radius:4px;
                       background:var(--bg-secondary);color:var(--text-primary);
                       font-size:13px;margin-bottom:8px;">

                <label class="field-label">API Key (optional):</label>
                <input type="password" name="api_key" value="{api_key}"
                       placeholder="Leave empty if not required" style="width:100%;
                       padding:8px;border:1px solid var(--border-color);border-radius:4px;
                       background:var(--bg-secondary);color:var(--text-primary);
                       font-size:13px;margin-bottom:8px;">

                <div class="memory-options">
                    <strong style="display:block;margin-bottom:10px;">Options</strong>

                    <label class="memory-option">
                        <input type="checkbox" name="auto_capture" value="true"
                               {"checked" if auto_capture else ""}>
                        <span>Auto-capture conversations</span>
                    </label>

                    <label class="memory-option">
                        <input type="checkbox" name="auto_recall" value="true"
                               {"checked" if auto_recall else ""}>
                        <span>Auto-recall relevant memories at session start</span>
                    </label>

                    <label class="memory-option">
                        <input type="checkbox" name="use_workspace_namespace" value="true"
                               {"checked" if use_workspace_namespace else ""}>
                        <span>Scope memories by workspace</span>
                    </label>

                    <label class="memory-option">
                        <input type="checkbox" name="use_persistent_session" value="true"
                               {"checked" if use_persistent_session else ""}>
                        <span>Use persistent session IDs</span>
                    </label>

                    <label class="memory-option" style="margin-bottom:0;">
                        <input type="checkbox" name="track_tool_usage" value="true"
                               {"checked" if track_tool_usage else ""}>
                        <span>Track tool usage as memories</span>
                    </label>
                </div>

                <button type="submit" class="btn-primary" style="margin-top:12px;">
                    Save Settings
                </button>
            </form>
        </div>
    '''


def _render_federation_config_section(config: dict) -> str:
    """Render the federation configuration section HTML."""
    fed_config = FederationConfig.from_dict(config.get("federation", {}))

    enabled_checked = "checked" if fed_config.enabled else ""
    share_locally_checked = "checked" if fed_config.share_locally else ""
    machine_name = html.escape(fed_config.this_machine_name)
    api_key = html.escape(fed_config.api_key or "")

    # Status indicator
    num_remotes = len(fed_config.remote_dashboards)
    if fed_config.enabled:
        status_color = "var(--status-idle)"
        plural = "s" if num_remotes != 1 else ""
        status_text = f"Enabled ({num_remotes} remote{plural})"
    else:
        status_color = "var(--text-secondary)"
        status_text = "Disabled"

    # Build remotes list HTML
    remotes_html = ""
    for i, remote in enumerate(fed_config.remote_dashboards):
        health_color = "var(--status-idle)" if remote.is_healthy else "var(--status-active)"
        health_icon = "✓" if remote.is_healthy else "✗"
        escaped_name = html.escape(remote.name)
        escaped_url = html.escape(remote.url)
        remotes_html += f'''
            <div class="remote-item">
                <div class="remote-info">
                    <span class="remote-health" style="color:{health_color};">
                        {health_icon}
                    </span>
                    <strong>{escaped_name}</strong>
                    <span class="remote-url">{escaped_url}</span>
                </div>
                <form method="POST" action="/config/federation/remotes/delete"
                      style="margin:0;">
                    <input type="hidden" name="index" value="{i}">
                    <button type="submit" class="btn-delete-remote">Remove</button>
                </form>
            </div>
        '''

    if not remotes_html:
        remotes_html = (
            '<p style="color:var(--text-secondary);margin:10px 0;">'
            "No remote dashboards configured.</p>"
        )

    return f'''
        <p class="section-description">
            Configure federation to connect multiple dashboards across machines.
        </p>

        <div class="config-card">
            <div class="memory-status">
                <span class="status-dot" style="background:{status_color};"></span>
                <strong>Status: {status_text}</strong>
            </div>

            <form method="POST" action="/config/federation">
                <div class="memory-options" style="margin-bottom:15px;">
                    <label class="memory-option">
                        <input type="checkbox" name="enabled" value="true" {enabled_checked}>
                        <span>Enable federation</span>
                    </label>

                    <label class="memory-option">
                        <input type="checkbox" name="share_locally" value="true"
                               {share_locally_checked}>
                        <span>Share sessions with other dashboards</span>
                    </label>
                </div>

                <label class="field-label">This Machine's Name:</label>
                <input type="text" name="this_machine_name" value="{machine_name}"
                       placeholder="e.g., Work Laptop" style="width:100%;padding:8px;
                       border:1px solid var(--border-color);border-radius:4px;
                       background:var(--bg-secondary);color:var(--text-primary);
                       font-size:13px;margin-bottom:8px;">

                <label class="field-label">API Key (for incoming connections):</label>
                <input type="password" name="api_key" value="{api_key}"
                       placeholder="Leave empty for no authentication" style="width:100%;
                       padding:8px;border:1px solid var(--border-color);border-radius:4px;
                       background:var(--bg-secondary);color:var(--text-primary);
                       font-size:13px;margin-bottom:8px;">

                <button type="submit" class="btn-primary" style="margin-top:8px;">
                    Save Settings
                </button>
            </form>
        </div>

        <div class="config-card" style="margin-top:12px;">
            <strong style="display:block;margin-bottom:8px;">Remote Dashboards</strong>
            <p style="color:var(--text-secondary);font-size:0.85em;margin-bottom:12px;">
                Add other machines' dashboards to see their sessions.
            </p>

            <div class="remotes-list">
                {remotes_html}
            </div>

            <div class="add-form" style="margin-top:12px;">
                <form method="POST" action="/config/federation/remotes/add">
                    <label class="field-label">Dashboard URL:</label>
                    <input type="text" name="url" placeholder="http://other-machine:8080" required>

                    <label class="field-label">Name:</label>
                    <input type="text" name="name" placeholder="e.g., Home Desktop" required>

                    <label class="field-label">API Key (if required):</label>
                    <input type="password" name="remote_api_key"
                           placeholder="Leave empty if not required" style="width:100%;
                           padding:8px;border:1px solid var(--border-color);border-radius:4px;
                           background:var(--bg-secondary);color:var(--text-primary);
                           font-size:13px;margin-bottom:8px;">

                    <button type="submit" class="btn-primary">Add Remote</button>
                </form>
            </div>
        </div>
    '''


def _render_quick_replies_config_section(config: dict) -> str:
    """Render the quick replies configuration section."""
    quick_replies = config.get("quick_replies", {})
    num_replies = len(quick_replies)

    # Build quick reply cards
    replies_html = ""
    if num_replies == 0:
        replies_html = '''
        <p style="color: var(--text-secondary); font-style: italic; margin: 10px 0;">
            No quick replies configured. Add one below to get started.
        </p>
        '''
    else:
        for name, message in quick_replies.items():
            escaped_name = html.escape(name)
            escaped_message = html.escape(message)
            replies_html += f'''
            <div class="config-card">
                <div class="config-card-header">
                    <strong>{escaped_name}</strong>
                    <form method="POST" action="/config/quick-replies/delete" class="inline-form">
                        <input type="hidden" name="name" value="{escaped_name}">
                        <button type="submit" onclick="return confirm('Delete this quick reply?')"
                            class="btn-icon btn-danger" title="Delete">🗑</button>
                    </form>
                </div>
                <form method="POST" action="/config/quick-replies/edit" class="config-edit-form">
                    <input type="hidden" name="name" value="{escaped_name}">
                    <label class="field-label">Message:</label>
                    <textarea name="message" rows="2">{escaped_message}</textarea>
                    <button type="submit" class="btn-primary btn-sm">Save</button>
                </form>
            </div>
            '''

    # Status indicator
    if num_replies > 0:
        status_color = "var(--status-idle)"
        plural = "ies" if num_replies != 1 else "y"
        status_text = f"{num_replies} quick repl{plural}"
    else:
        status_color = "var(--text-secondary)"
        status_text = "None configured"

    return f'''
    <div class="config-section">
        <div class="section-header" onclick="toggleSection('quick-replies-content')">
            <h2>⚡ Quick Replies
                <span style="font-size:12px;color:{status_color};margin-left:8px;">
                    ({status_text})
                </span>
            </h2>
            <span class="section-toggle" id="quick-replies-toggle">▼</span>
        </div>
        <div class="section-content" id="quick-replies-content">
            <p class="section-description">
                Pre-configured messages for quick agent communication. Click a quick reply
                button in a session to populate the message field.
            </p>
            {replies_html}
            <div class="add-form">
                <h4>Add New Quick Reply</h4>
                <form method="POST" action="/config/quick-replies/add">
                    <label class="field-label">Name (button label):</label>
                    <input type="text" name="name"
                        placeholder="e.g., 'Review Code'" required>
                    <label class="field-label">Message:</label>
                    <textarea name="message"
                        placeholder="Enter the message to send" required></textarea>
                    <button type="submit" class="btn-primary">Add Quick Reply</button>
                </form>
            </div>
        </div>
    </div>
    '''


def _render_agent_settings_section(config: dict) -> str:
    """Render the agent settings configuration section."""
    agent_timeout_minutes = config.get("agent_timeout_minutes", 15)
    max_loop_iterations = config.get("max_loop_iterations", 50)

    return f'''
    <div class="config-section">
        <div class="section-header" onclick="toggleSection('agent-settings-content')">
            <h2>🤖 Agent Settings</h2>
            <span class="section-toggle" id="agent-settings-toggle">▼</span>
        </div>
        <div class="section-content" id="agent-settings-content">
            <p class="section-description">
                Configure agent behavior, timeouts, and loop settings.
            </p>

            <div class="config-card">
                <form method="POST" action="/config/agent-settings">
                    <label class="field-label">Agent Timeout (minutes):</label>
                    <input type="number" name="agent_timeout_minutes" value="{agent_timeout_minutes}"
                           min="1" max="120" style="width:100%;padding:8px;
                           border:1px solid var(--border-color);border-radius:4px;
                           background:var(--bg-secondary);color:var(--text-primary);
                           font-size:13px;margin-bottom:8px;">
                    <p style="color:var(--text-secondary);font-size:0.85em;margin-bottom:12px;">
                        If an agent hasn't responded for this long, the session is reset (default: 15).
                    </p>

                    <label class="field-label">Max Loop Iterations:</label>
                    <input type="number" name="max_loop_iterations" value="{max_loop_iterations}"
                           min="1" max="500" style="width:100%;padding:8px;
                           border:1px solid var(--border-color);border-radius:4px;
                           background:var(--bg-secondary);color:var(--text-primary);
                           font-size:13px;margin-bottom:8px;">
                    <p style="color:var(--text-secondary);font-size:0.85em;margin-bottom:12px;">
                        Maximum number of loop iterations before automatically stopping (default: 50).
                    </p>

                    <button type="submit" class="btn-primary" style="margin-top:8px;">
                        Save Settings
                    </button>
                </form>
            </div>
        </div>
    </div>
    '''


def render_config_page(
    dark_mode: str | None,
    loop_prompts: dict[str, dict[str, str]],
    config: dict,
) -> str:
    """Render the configuration page HTML."""
    styles = get_base_styles(dark_mode)
    prompt_count = len(loop_prompts)

    # Build prompt list
    prompts_html = ""
    for name, prompt_config in loop_prompts.items():
        escaped_name = html.escape(name)
        # Handle both new format (dict) and legacy format (string)
        if isinstance(prompt_config, str):
            escaped_prompt = html.escape(prompt_config)
            escaped_condition = ""
        else:
            escaped_prompt = html.escape(prompt_config.get("prompt", ""))
            escaped_condition = html.escape(prompt_config.get("end_condition", ""))
        prompts_html += f'''
        <div class="config-card">
            <div class="config-card-header">
                <strong>{escaped_name}</strong>
                <form method="POST" action="/config/prompts/delete" class="inline-form">
                    <input type="hidden" name="name" value="{escaped_name}">
                    <button type="submit" onclick="return confirm('Delete this prompt?')"
                        class="btn-icon btn-danger" title="Delete">🗑</button>
                </form>
            </div>
            <form method="POST" action="/config/prompts/edit" class="config-edit-form">
                <input type="hidden" name="name" value="{escaped_name}">
                <label class="field-label">Prompt (instructions for the LLM):</label>
                <textarea name="prompt" rows="3">{escaped_prompt}</textarea>
                <label class="field-label">End Condition (stops loop when found):</label>
                <input type="text" name="end_condition" value="{escaped_condition}"
                    placeholder="e.g., LOOP_COMPLETE: Task finished.">
                <button type="submit" class="btn-primary btn-sm">Save</button>
            </form>
        </div>
        '''

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Configuration - Augment Dashboard</title>
        <link rel="manifest" href="/manifest.json">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <meta name="apple-mobile-web-app-title" content="Augment">
        <link rel="apple-touch-icon" href="/icon-192.png">
        <meta name="theme-color" content="#6366f1">
        {styles}
        <style>
            /* Collapsible sections */
            .config-section {{
                background: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                margin-bottom: 16px;
                overflow: hidden;
            }}
            .section-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 14px 16px;
                cursor: pointer;
                user-select: none;
                background: var(--bg-secondary);
            }}
            .section-header:hover {{
                background: var(--bg-hover);
            }}
            .section-header h2 {{
                margin: 0;
                font-size: 1.1em;
            }}
            .section-toggle {{
                font-size: 0.9em;
                color: var(--text-secondary);
                transition: transform 0.2s;
            }}
            .section-content {{
                padding: 0 16px 16px;
                display: block;
            }}
            .section-content.collapsed {{
                display: none;
            }}
            .section-description {{
                color: var(--text-secondary);
                margin-bottom: 15px;
                font-size: 0.9em;
            }}
            .section-badge {{
                background: var(--accent-color);
                color: white;
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 0.75em;
                margin-left: 8px;
            }}
            /* Config cards */
            .config-card {{
                background: var(--bg-primary);
                border: 1px solid var(--border-color);
                border-radius: 6px;
                padding: 12px;
                margin-bottom: 10px;
            }}
            .config-card-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }}
            .config-edit-form textarea,
            .config-edit-form input[type="text"] {{
                width: 100%;
                padding: 8px;
                border: 1px solid var(--border-color);
                border-radius: 4px;
                background: var(--bg-secondary);
                color: var(--text-primary);
                font-family: inherit;
                font-size: 13px;
                resize: vertical;
                margin-bottom: 8px;
            }}
            .field-label {{
                display: block;
                font-size: 0.8em;
                color: var(--text-secondary);
                margin-bottom: 4px;
                margin-top: 8px;
            }}
            .field-label:first-of-type {{
                margin-top: 0;
            }}
            /* Add forms */
            .add-form {{
                background: var(--bg-primary);
                border: 1px dashed var(--border-color);
                border-radius: 6px;
                padding: 14px;
                margin-top: 12px;
            }}
            .add-form h4 {{
                margin: 0 0 12px 0;
                font-size: 0.95em;
                color: var(--text-secondary);
            }}
            .add-form input[type="text"],
            .add-form textarea {{
                width: 100%;
                padding: 8px;
                border: 1px solid var(--border-color);
                border-radius: 4px;
                background: var(--bg-secondary);
                color: var(--text-primary);
                font-size: 13px;
                margin-bottom: 8px;
            }}
            .add-form textarea {{
                min-height: 60px;
                resize: vertical;
                font-family: inherit;
            }}
            /* Button styles */
            .btn-primary {{
                padding: 8px 16px;
                background: var(--accent-color);
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                font-weight: 500;
            }}
            .btn-primary:hover {{
                opacity: 0.9;
            }}
            .btn-sm {{
                padding: 5px 10px;
                font-size: 12px;
            }}
            .btn-icon {{
                padding: 4px 8px;
                background: transparent;
                border: 1px solid var(--border-color);
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            }}
            .btn-danger {{
                color: var(--status-active);
                border-color: var(--status-active);
            }}
            .btn-danger:hover {{
                background: var(--status-active);
                color: white;
            }}
            /* Legacy styles for federation/memory sections */
            .memory-status {{
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 12px;
            }}
            .status-dot {{
                width: 10px;
                height: 10px;
                border-radius: 50%;
            }}
            .memory-options {{
                margin-top: 15px;
                padding: 12px;
                background: var(--bg-primary);
                border-radius: 8px;
            }}
            .memory-option {{
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 8px;
                cursor: pointer;
            }}
            .remotes-list {{
                margin: 10px 0;
            }}
            .remote-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 12px;
                background: var(--bg-primary);
                border-radius: 6px;
                margin-bottom: 8px;
            }}
            .remote-info {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .remote-health {{
                font-size: 1.1em;
            }}
            .remote-url {{
                color: var(--text-secondary);
                font-size: 0.9em;
            }}
            .btn-delete-remote {{
                padding: 4px 10px;
                background: transparent;
                color: var(--status-active);
                border: 1px solid var(--status-active);
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.85em;
            }}
            .btn-delete-remote:hover {{
                background: var(--status-active);
                color: white;
            }}
        </style>
    </head>
    <body>
        <a href="/" class="back-link">← Back to Dashboard</a>
        <h1>⚙️ Configuration</h1>

        <!-- Quick Replies Section (expanded by default - fewer items) -->
        {_render_quick_replies_config_section(config)}

        <!-- Agent Settings Section -->
        {_render_agent_settings_section(config)}

        <!-- Loop Prompts Section (collapsed by default - many items) -->
        <div class="config-section">
            <div class="section-header" onclick="toggleSection('loop-prompts-content')">
                <h2>🔄 Loop Prompts <span class="section-badge">{prompt_count}</span></h2>
                <span class="section-toggle" id="loop-prompts-toggle">▶</span>
            </div>
            <div class="section-content collapsed" id="loop-prompts-content">
                <p class="section-description">
                    Configure loop prompts with end conditions. The prompt tells the LLM what to do
                    and should explain the end condition. When the LLM includes the end condition
                    text in its response, the loop stops.
                </p>
                {prompts_html}
                <div class="add-form">
                    <h4>Add New Prompt</h4>
                    <form method="POST" action="/config/prompts/add">
                        <label class="field-label">Name:</label>
                        <input type="text" name="name"
                            placeholder="e.g., 'Security Review'" required>
                        <label class="field-label">Prompt (instructions):</label>
                        <textarea name="prompt"
                            placeholder="Enter instructions for the agent."
                            required></textarea>
                        <label class="field-label">End Condition:</label>
                        <input type="text" name="end_condition"
                            placeholder="e.g., LOOP_COMPLETE: Done.">
                        <button type="submit" class="btn-primary">Add Prompt</button>
                    </form>
                </div>
            </div>
        </div>

        <!-- Federation Section -->
        <div class="config-section">
            <div class="section-header" onclick="toggleSection('federation-content')">
                <h2>🌐 Federation</h2>
                <span class="section-toggle" id="federation-toggle">▼</span>
            </div>
            <div class="section-content" id="federation-content">
                {_render_federation_config_section(config)}
            </div>
        </div>

        <!-- Memory Section -->
        <div class="config-section">
            <div class="section-header" onclick="toggleSection('memory-content')">
                <h2>🧠 Memory</h2>
                <span class="section-toggle" id="memory-toggle">▼</span>
            </div>
            <div class="section-content" id="memory-content">
                {_render_memory_config_section(config)}
            </div>
        </div>

        <script>
            function toggleSection(sectionId) {{
                const content = document.getElementById(sectionId);
                const toggleId = sectionId.replace('-content', '-toggle');
                const toggle = document.getElementById(toggleId);

                if (content.classList.contains('collapsed')) {{
                    content.classList.remove('collapsed');
                    toggle.textContent = '▼';
                }} else {{
                    content.classList.add('collapsed');
                    toggle.textContent = '▶';
                }}
            }}
        </script>
    </body>
    </html>
    """


def _format_elapsed_time(started_at: datetime | None) -> str:
    """Format elapsed time since loop started."""
    if not started_at:
        return ""
    now = datetime.now(timezone.utc)
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    elapsed = now - started_at
    total_seconds = int(elapsed.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def _render_quick_replies_html(session_id: str) -> str:
    """Render quick reply buttons for the message form."""
    quick_replies = _get_quick_replies()
    if not quick_replies:
        return ""

    buttons_html = ""
    for name, message in quick_replies.items():
        escaped_name = html.escape(name)
        escaped_message = html.escape(message).replace("'", "\\'")
        buttons_html += f'''
            <button type="button" class="quick-reply-btn"
                onclick="insertQuickReply('{escaped_message}')"
                title="{escaped_message}">⚡ {escaped_name}</button>
        '''

    return f'''
        <div class="quick-replies-section">
            <label class="field-label" style="margin-bottom:6px;">Quick Replies:</label>
            <div class="quick-replies-list">{buttons_html}</div>
        </div>
    '''


def _get_quick_replies_styles() -> str:
    """Get CSS styles for quick replies section."""
    return """
        .quick-replies-section {
            margin-bottom: 12px;
        }
        .quick-replies-list {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        .quick-reply-btn {
            padding: 5px 10px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            color: var(--text-primary);
            font-size: 12px;
            cursor: pointer;
            white-space: nowrap;
        }
        .quick-reply-btn:hover {
            background: var(--bg-hover);
            border-color: var(--accent-color);
        }
    """


def _render_message_form(session) -> str:
    """Render the message form - send when idle, enqueue when busy."""
    from augment_agent_dashboard.models import SessionStatus

    # Count queued messages
    queued_count = sum(1 for m in session.messages if m.role == "queued")
    queue_info = ""
    if queued_count > 0:
        queue_info = f' <span class="queue-count">({queued_count} queued)</span>'

    sid = session.session_id
    quick_replies_html = _render_quick_replies_html(sid)

    if session.status == SessionStatus.ACTIVE:
        # Agent is working - can only enqueue
        return f'''
            <div class="status-banner status-active">
                ⏳ Agent working{queue_info}
            </div>
            {quick_replies_html}
            <form method="POST" action="/session/{sid}/queue">
                <textarea id="message-input" name="message"
                    placeholder="Type a message..."></textarea>
                <button type="submit" class="btn-queue">🕐 Enqueue</button>
            </form>
        '''
    else:
        # Agent is idle - can send directly
        return f'''
            {quick_replies_html}
            <form method="POST" action="/session/{sid}/message" class="message-form">
                <textarea id="message-input" name="message"
                    placeholder="Type a message..."></textarea>
                <button type="submit">▶ Send</button>
            </form>
        '''


def _render_loop_controls(session, loop_prompts: dict[str, dict[str, str]]) -> str:
    """Render the loop control UI section."""
    if session.loop_enabled:
        elapsed = _format_elapsed_time(session.loop_started_at)
        prompt_name = session.loop_prompt_name or "Unknown"

        # Get the end condition for the active loop
        loop_config = loop_prompts.get(prompt_name, {})
        if isinstance(loop_config, str):
            end_condition = ""
            prompt_text = loop_config
        else:
            end_condition = loop_config.get("end_condition", "")
            prompt_text = loop_config.get("prompt", "")

        # Build end condition display
        end_condition_html = ""
        if end_condition:
            escaped_condition = html.escape(end_condition)
            end_condition_html = f'''
                <div class="loop-end-condition">
                    <span class="end-condition-label">🎯 Stops when response contains:</span>
                    <code class="end-condition-text">{escaped_condition}</code>
                </div>
            '''

        # Build prompt preview (collapsed by default)
        prompt_preview_html = ""
        if prompt_text:
            escaped_prompt = html.escape(prompt_text)
            prompt_preview_html = f'''
                <details class="loop-prompt-details">
                    <summary>📝 View prompt</summary>
                    <div class="loop-prompt-text">{escaped_prompt}</div>
                </details>
            '''

        return f'''
            <div class="loop-controls-container">
                <div class="loop-controls">
                    <span style="color:var(--status-active);font-weight:bold;">
                        🔄 {html.escape(prompt_name)}
                    </span>
                    <span style="color:var(--text-secondary);">
                        {session.loop_count} iterations, {elapsed}
                    </span>
                    <form method="POST" action="/session/{session.session_id}/loop/pause">
                        <button type="submit" class="btn-pause">⏸ Pause</button>
                    </form>
                    <form method="POST" action="/session/{session.session_id}/loop/reset">
                        <button type="submit" class="btn-reset">↺ Reset</button>
                    </form>
                </div>
                {end_condition_html}
                {prompt_preview_html}
            </div>
        '''
    else:
        # Build dropdown options with title tooltips showing prompt preview
        options_html = ""
        for name, config in loop_prompts.items():
            escaped_name = html.escape(name)
            if isinstance(config, str):
                tooltip = config[:100] + "..." if len(config) > 100 else config
            else:
                prompt = config.get("prompt", "")
                end_cond = config.get("end_condition", "")
                prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
                tooltip = f"Prompt: {prompt_preview}"
                if end_cond:
                    tooltip += f"\n\nStops when: {end_cond}"
            escaped_tooltip = html.escape(tooltip)
            opt = f'<option value="{escaped_name}" title="{escaped_tooltip}">'
            options_html += f'{opt}{escaped_name}</option>'

        return f'''
            <div class="loop-controls">
                <span style="color:var(--text-secondary);">Loop Paused</span>
                <form method="POST" action="/session/{session.session_id}/loop/enable">
                    <select name="prompt_name" id="loop-prompt-select">{options_html}</select>
                    <button type="submit" class="btn-enable">▶ Enable</button>
                </form>
                <form method="POST" action="/session/{session.session_id}/loop/reset">
                    <button type="submit" class="btn-reset">↺ Reset</button>
                </form>
            </div>
            <div id="loop-prompt-preview" class="loop-prompt-preview" style="display:none;"></div>
        '''


def _render_messages_html(session) -> tuple[str, int]:
    """Render just the messages HTML for a session.

    Returns a tuple of (messages_html, queued_count).
    """
    import base64

    messages_html = ""
    queued_count = 0
    if not session.messages:
        messages_html = '<div class="empty-state">No messages in this session yet.</div>'
    else:
        for idx, msg in enumerate(session.messages):
            role_class = msg.role
            time_str = (
                format_time_ago(msg.timestamp, include_title=True)
                if msg.timestamp
                else ""
            )

            if msg.role == "queued":
                queued_count += 1
                role_label = f"🕐 Queued #{queued_count}"
                content_html = f"<p>{html.escape(msg.content)}</p>"
            elif msg.role == "assistant":
                role_label = "Assistant"
                content_html = render_markdown(msg.content)
            else:
                role_label = msg.role.capitalize()
                content_html = f"<p>{html.escape(msg.content)}</p>"

            # Encode raw content as base64 for the copy button
            raw_content_b64 = base64.b64encode(msg.content.encode("utf-8")).decode("ascii")
            msg_id = f"msg-{idx}"

            copy_onclick = f"copyMessage(this, '{raw_content_b64}')"
            messages_html += f"""
            <div class="message {role_class}" id="{msg_id}">
                <div class="message-header">
                    <span class="message-header-info">{role_label} • {time_str}</span>
                    <button class="copy-btn" onclick="{copy_onclick}" title="Copy">
                        📋 Copy
                    </button>
                </div>
                <div class="message-content">{content_html}</div>
            </div>
            """

    # Add clear queue button if there are queued messages
    if queued_count > 0:
        confirm_msg = f"Clear all {queued_count} queued messages?"
        messages_html += f'''
        <div class="queue-actions">
            <form method="POST" action="/session/{session.session_id}/queue/clear">
                <button type="submit" class="btn-delete btn-small"
                    onclick="return confirm('{confirm_msg}')">
                    🗑 Clear Queue ({queued_count})
                </button>
            </form>
        </div>
        '''

    return messages_html, queued_count


def _get_state_label(state_value: str) -> str:
    """Get a human-readable label for a session state."""
    labels = {
        "idle": "Idle",
        "active": "Working",
        "turn_complete": "Turn Complete",
        "review_pending": "Review Pending",
        "under_review": "Under Review",
        "ready_for_loop": "Ready",
        "loop_prompting": "Looping",
        "error": "Error",
    }
    return labels.get(state_value, state_value.replace("_", " ").title())


def _render_state_badge(session) -> str:
    """Render a state badge showing the detailed session state."""
    try:
        state_value = session.state.value
    except (AttributeError, ValueError):
        # Fall back to status if state not available
        state_value = session.status.value

    label = _get_state_label(state_value)
    return f'''<span class="state-badge badge-{state_value}">
        <span class="state-dot state-{state_value}"></span>
        {label}
    </span>'''


def _render_session_status_html(session) -> str:
    """Render the session status indicator HTML."""
    time_ago = format_time_ago(session.last_activity, include_title=True)
    state_badge = _render_state_badge(session)
    return f"""
        <div>{state_badge} • {time_ago}</div>
        <div>{session.message_count} messages</div>
    """


def render_session_detail(
    session,
    dark_mode: str | None,
    loop_prompts: dict[str, dict[str, str]],
    machine_name: str = "This Machine",
) -> str:
    """Render the session detail HTML."""
    styles = get_base_styles(dark_mode)
    quick_replies_styles = _get_quick_replies_styles()

    # Render message history
    messages_html, queued_count = _render_messages_html(session)

    # Get state for styling
    try:
        state_value = session.state.value
    except (AttributeError, ValueError):
        state_value = session.status.value

    state_class = f"state-{state_value}"
    time_ago = format_time_ago(session.last_activity, include_title=True)
    escaped_machine = html.escape(machine_name)
    state_badge = _render_state_badge(session)

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{session.workspace_name} - Augment Dashboard</title>
        <link rel="manifest" href="/manifest.json">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <meta name="apple-mobile-web-app-title" content="Augment">
        <link rel="apple-touch-icon" href="/icon-192.png">
        <meta name="theme-color" content="#6366f1">
        <style>{quick_replies_styles}</style>
        {styles}
    </head>
    <body>
        <div id="pull-to-refresh" class="pull-to-refresh">
            <div class="pull-to-refresh-spinner"></div>
            <span class="pull-to-refresh-text">Pull to refresh</span>
        </div>
        <a href="/" class="back-link">← Back to Dashboard</a>

        <div class="header">
            <h1>
                <span class="status-dot {state_class}"
                    style="display:inline-block;vertical-align:middle;margin-right:10px;">
                </span>
                {session.workspace_name}
            </h1>
            <div class="session-meta" id="session-status">
                <div>{state_badge} • {time_ago}</div>
                <div>{session.message_count} messages</div>
            </div>
        </div>

        <div class="session-detail-meta">
            <div class="machine-badge-inline">
                💻 {escaped_machine}
            </div>
            <br>
            <strong>Workspace:</strong> {session.workspace_root}<br>
            <strong>Session ID:</strong> {session.session_id}
            <div id="loop-controls-container">
                {_render_loop_controls(session, loop_prompts)}
            </div>
            <div class="loop-controls" style="margin-top:8px;">
                <form method="POST" action="/session/{session.session_id}/delete">
                    <button type="submit" class="btn-delete"
                        onclick="return confirm('Delete this session?')">
                        🗑 Delete Session
                    </button>
                </form>
            </div>
        </div>

        <h2>Conversation</h2>
        <div class="message-list" id="message-list">
            {messages_html}
        </div>

        <div class="message-form" id="message-form-container">
            <h3>Send Message to Agent</h3>
            <div id="message-form-content">
                {_render_message_form(session)}
            </div>
        </div>

        <script>
            {_get_timestamp_script()}
            {_get_pull_to_refresh_script()}

            // Insert quick reply into message input
            function insertQuickReply(message) {{
                const textarea = document.getElementById('message-input');
                if (textarea) {{
                    textarea.value = message;
                    textarea.focus();
                    // Also save to cache
                    if (typeof saveMessageToCache === 'function') {{
                        saveMessageToCache();
                    }}
                }}
            }}

            // Copy message to clipboard
            async function copyMessage(btn, base64Content) {{
                try {{
                    // Decode base64 content
                    const text = atob(base64Content);
                    await navigator.clipboard.writeText(text);

                    // Visual feedback
                    const originalText = btn.innerHTML;
                    btn.innerHTML = '✓ Copied';
                    btn.classList.add('copied');

                    setTimeout(() => {{
                        btn.innerHTML = originalText;
                        btn.classList.remove('copied');
                    }}, 2000);
                }} catch (err) {{
                    console.error('Failed to copy:', err);
                    // Fallback for older browsers
                    const text = atob(base64Content);
                    const textarea = document.createElement('textarea');
                    textarea.value = text;
                    textarea.style.position = 'fixed';
                    textarea.style.opacity = '0';
                    document.body.appendChild(textarea);
                    textarea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textarea);

                    btn.innerHTML = '✓ Copied';
                    btn.classList.add('copied');
                    setTimeout(() => {{
                        btn.innerHTML = '📋 Copy';
                        btn.classList.remove('copied');
                    }}, 2000);
                }}
            }}

            // AJAX-based session updates
            const REFRESH_INTERVAL = 3000;
            const sessionId = '{session.session_id}';
            let lastMessageCount = {session.message_count};

            function isUserInteracting() {{
                const textarea = document.getElementById('message-input');
                if (!textarea) return false;

                // Check if textarea has focus
                if (document.activeElement === textarea) return true;

                // Check if textarea has content
                if (textarea.value.trim()) return true;

                return false;
            }}

            async function refreshSession() {{
                try {{
                    const url = '/api/sessions/' + encodeURIComponent(sessionId);
                    const response = await fetch(url + '/messages-html');
                    if (!response.ok) return;

                    const data = await response.json();

                    // Update status indicator in header
                    const statusMeta = document.querySelector('.session-meta');
                    if (statusMeta) {{
                        statusMeta.innerHTML = data.status_html;
                    }}

                    // Update status dot class
                    const statusDot = document.querySelector('.status-dot');
                    if (statusDot) {{
                        statusDot.className = 'status-dot status-' + data.status;
                    }}

                    // Update messages - preserve scroll position
                    const messageList = document.getElementById('message-list');
                    if (messageList) {{
                        const scrollDiff = messageList.scrollHeight - messageList.scrollTop;
                        const wasAtBottom = scrollDiff <= messageList.clientHeight + 100;
                        const oldScrollTop = messageList.scrollTop;

                        messageList.innerHTML = data.messages_html;

                        // If user was at bottom or there are new messages, scroll to bottom
                        if (wasAtBottom || data.message_count > lastMessageCount) {{
                            messageList.scrollTop = messageList.scrollHeight;
                        }} else {{
                            messageList.scrollTop = oldScrollTop;
                        }}
                        lastMessageCount = data.message_count;
                    }}

                    // Update loop controls
                    const loopControls = document.getElementById('loop-controls-container');
                    if (loopControls) {{
                        loopControls.innerHTML = data.loop_controls_html;
                    }}

                    // Update message form only if user is not interacting
                    if (!isUserInteracting()) {{
                        const formContent = document.getElementById('message-form-content');
                        if (formContent) {{
                            formContent.innerHTML = data.message_form_html;
                            // Re-setup textarea caching after form replacement
                            if (typeof setupTextareaCache === 'function') {{
                                setupTextareaCache();
                            }}
                        }}
                    }}
                }} catch (e) {{
                    console.error('Failed to refresh session:', e);
                }}
                scheduleRefresh();
            }}

            function scheduleRefresh() {{
                setTimeout(refreshSession, REFRESH_INTERVAL);
            }}

            // Start the refresh cycle
            scheduleRefresh();

            // Scroll to bottom on initial load
            const messageList = document.getElementById('message-list');
            if (messageList) {{
                messageList.scrollTop = messageList.scrollHeight;
            }}

            // Cmd+Enter (Mac) or Ctrl+Enter (Windows/Linux) to send/queue message
            document.addEventListener('keydown', function(e) {{
                if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {{
                    const textarea = document.getElementById('message-input');
                    if (!textarea || !textarea.value.trim()) return;

                    // Find the form containing the textarea
                    const form = textarea.closest('form');
                    if (!form) return;

                    e.preventDefault();

                    // Find the first submit button in the form (whatever it is)
                    const firstBtn = form.querySelector('button[type="submit"]');
                    if (firstBtn) {{
                        // Clear the cache since we're submitting
                        clearMessageCache();
                        // Click the button to ensure formaction is used if present
                        firstBtn.click();
                    }}
                }}
            }});

            // localStorage caching for message input
            const MESSAGE_CACHE_KEY = 'augment_dashboard_message_' + sessionId;
            let cacheSaveTimeout;

            function saveMessageToCache() {{
                const textarea = document.getElementById('message-input');
                if (textarea) {{
                    const value = textarea.value;
                    if (value.trim()) {{
                        localStorage.setItem(MESSAGE_CACHE_KEY, value);
                    }} else {{
                        localStorage.removeItem(MESSAGE_CACHE_KEY);
                    }}
                }}
            }}

            function loadMessageFromCache() {{
                const textarea = document.getElementById('message-input');
                if (textarea) {{
                    const cached = localStorage.getItem(MESSAGE_CACHE_KEY);
                    if (cached && !textarea.value.trim()) {{
                        textarea.value = cached;
                    }}
                }}
            }}

            function clearMessageCache() {{
                localStorage.removeItem(MESSAGE_CACHE_KEY);
            }}

            // Set up caching on textarea - called on load and after AJAX form updates
            function setupTextareaCache() {{
                const textarea = document.getElementById('message-input');
                if (textarea && !textarea.dataset.cacheSetup) {{
                    // Mark as set up to avoid duplicate listeners
                    textarea.dataset.cacheSetup = 'true';

                    // Load cached message
                    loadMessageFromCache();

                    // Save on input (debounced)
                    textarea.addEventListener('input', function() {{
                        clearTimeout(cacheSaveTimeout);
                        cacheSaveTimeout = setTimeout(saveMessageToCache, 300);
                    }});

                    // Clear cache when form is submitted
                    const form = textarea.closest('form');
                    if (form && !form.dataset.cacheSetup) {{
                        form.dataset.cacheSetup = 'true';
                        form.addEventListener('submit', clearMessageCache);
                    }}
                }}
            }}

            // Initial setup
            setupTextareaCache();
        </script>
    </body>
    </html>
    """


def render_remote_session_detail(
    session_data: dict,
    remote,
    federated_session_id: str,
    dark_mode: str | None,
) -> str:
    """Render the session detail HTML for a remote session.

    Args:
        session_data: Session data dict from the remote dashboard.
        remote: RemoteDashboard instance.
        federated_session_id: The federated session ID for URL links.
        dark_mode: Dark mode setting.
    """
    styles = get_base_styles(dark_mode)

    workspace_name = html.escape(session_data.get("workspace_name", "Unknown"))
    workspace_root = html.escape(session_data.get("workspace_root", ""))
    status = session_data.get("status", "stopped")
    message_count = session_data.get("message_count", 0)
    remote_session_id = session_data.get("session_id", "")

    # Parse last_activity for time ago
    last_activity_str = session_data.get("last_activity", "")
    if last_activity_str:
        try:
            from datetime import datetime
            last_activity = datetime.fromisoformat(last_activity_str.replace("Z", "+00:00"))
            time_ago = format_time_ago(last_activity, include_title=True)
        except Exception:
            time_ago = "Unknown"
    else:
        time_ago = "Unknown"

    # Render messages
    messages = session_data.get("messages", [])
    messages_html = ""
    for msg in messages:
        role = msg.get("role", "system")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")

        role_class = f"message-{role}"
        role_label = role.upper()

        if role == "assistant":
            content_html = render_markdown(content)
        else:
            content_html = f"<pre>{html.escape(content)}</pre>"

        # Base64 encode for copy button
        import base64
        base64_content = base64.b64encode(content.encode()).decode()

        copy_fn = f"copyMessage(this, '{base64_content}')"
        messages_html += f'''
        <div class="message {role_class}">
            <div class="message-header">
                <span class="role-badge">{role_label}</span>
                <span class="timestamp" data-timestamp="{timestamp}">{timestamp}</span>
                <button class="copy-btn" onclick="{copy_fn}">📋 Copy</button>
            </div>
            <div class="message-content">{content_html}</div>
        </div>
        '''

    if not messages_html:
        messages_html = '<div class="no-sessions">No messages yet</div>'

    # Message form for remote sessions - proxied through our server
    form_action = f"/api/remote/session/{federated_session_id}/message"
    message_form = f'''
        <p style="color: var(--text-secondary); margin-bottom: 10px;">
            Send a message to <strong>{html.escape(remote.name)}</strong>
        </p>
        <form method="POST" action="{form_action}">
            <textarea id="message-input" name="message"
                placeholder="Type a message for the agent..."></textarea>
            <button type="submit">▶ Send to Remote</button>
        </form>
    '''

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{workspace_name} - {html.escape(remote.name)} - Augment Dashboard</title>
        <link rel="manifest" href="/manifest.json">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <meta name="apple-mobile-web-app-title" content="Augment">
        <link rel="apple-touch-icon" href="/icon-192.png">
        <meta name="theme-color" content="#6366f1">
        {styles}
    </head>
    <body>
        <a href="/" class="back-link">← Back to Dashboard</a>

        <div class="header">
            <h1>
                <span class="status-dot status-{status}"
                    style="display:inline-block;vertical-align:middle;margin-right:10px;">
                </span>
                {workspace_name}
            </h1>
            <div class="session-meta">
                <div>{status} • {time_ago}</div>
                <div>{message_count} messages</div>
            </div>
        </div>

        <div class="session-detail-meta">
            <div class="remote-session-badge">
                🌐 <strong>Remote Session</strong> from
                <strong>{html.escape(remote.name)}</strong>
                <span class="queue-count">({html.escape(remote.url)})</span>
            </div>
            <strong>Workspace:</strong> {workspace_root}<br>
            <strong>Remote Session ID:</strong> {remote_session_id}
            <div class="loop-controls" style="margin-top:8px;">
                <form method="POST" action="/api/remote/session/{federated_session_id}/delete">
                    <button type="submit" class="btn-delete"
                        onclick="return confirm('Delete this remote session?')">
                        🗑 Delete Session
                    </button>
                </form>
            </div>
        </div>

        <h2>Conversation</h2>
        <div class="message-list" id="message-list">
            {messages_html}
        </div>

        <div class="message-form" id="message-form-container">
            <h3>Send Message to Agent</h3>
            <div id="message-form-content">
                {message_form}
            </div>
        </div>

        <script>
            {_get_timestamp_script()}

            // Copy message to clipboard
            async function copyMessage(btn, base64Content) {{
                try {{
                    const text = atob(base64Content);
                    await navigator.clipboard.writeText(text);
                    btn.innerHTML = '✓ Copied';
                    btn.classList.add('copied');
                    setTimeout(() => {{
                        btn.innerHTML = '📋 Copy';
                        btn.classList.remove('copied');
                    }}, 2000);
                }} catch (err) {{
                    console.error('Failed to copy:', err);
                }}
            }}

            // Scroll to bottom on load
            const messageList = document.getElementById('message-list');
            if (messageList) {{
                messageList.scrollTop = messageList.scrollHeight;
            }}

            // localStorage caching for message input (remote session)
            const remoteSessionId = '{federated_session_id}';
            const MESSAGE_CACHE_KEY = 'augment_dashboard_message_' + remoteSessionId;

            function saveMessageToCache() {{
                const textarea = document.getElementById('message-input');
                if (textarea) {{
                    const value = textarea.value;
                    if (value.trim()) {{
                        localStorage.setItem(MESSAGE_CACHE_KEY, value);
                    }} else {{
                        localStorage.removeItem(MESSAGE_CACHE_KEY);
                    }}
                }}
            }}

            function loadMessageFromCache() {{
                const textarea = document.getElementById('message-input');
                if (textarea) {{
                    const cached = localStorage.getItem(MESSAGE_CACHE_KEY);
                    if (cached && !textarea.value.trim()) {{
                        textarea.value = cached;
                    }}
                }}
            }}

            function clearMessageCache() {{
                localStorage.removeItem(MESSAGE_CACHE_KEY);
            }}

            // Set up caching on textarea
            (function() {{
                const textarea = document.getElementById('message-input');
                if (textarea) {{
                    // Load cached message on page load
                    loadMessageFromCache();

                    // Save on input (debounced)
                    let saveTimeout;
                    textarea.addEventListener('input', function() {{
                        clearTimeout(saveTimeout);
                        saveTimeout = setTimeout(saveMessageToCache, 300);
                    }});

                    // Clear cache when form is submitted
                    const form = textarea.closest('form');
                    if (form) {{
                        form.addEventListener('submit', clearMessageCache);
                    }}
                }}
            }})();

            // Cmd+Enter (Mac) or Ctrl+Enter (Windows/Linux) to send message
            document.addEventListener('keydown', function(e) {{
                if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {{
                    const textarea = document.getElementById('message-input');
                    if (!textarea || !textarea.value.trim()) return;

                    const form = textarea.closest('form');
                    if (!form) return;

                    e.preventDefault();

                    // Find the first submit button
                    const firstBtn = form.querySelector('button[type="submit"]');
                    if (firstBtn) {{
                        clearMessageCache();
                        firstBtn.click();
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """


# Default loop prompts - each line must be <= 100 chars to pass linting
DEFAULT_LOOP_PROMPTS: dict[str, dict[str, str]] = {
    "TDD Quality": {
        "prompt": (
            "Continue working on this task using TDD. Write tests first, then "
            "implement code to pass them. Verify code quality with mfcqi "
            "(target score >= 0.8). When you have absolutely completed every "
            "requirement—including ones you think don't matter—respond with "
            "exactly: 'LOOP_COMPLETE: TDD quality goals achieved.'"
        ),
        "end_condition": "LOOP_COMPLETE: TDD quality goals achieved.",
    },
    "Code Review": {
        "prompt": (
            "Review the code you just wrote. Look for bugs, security issues, "
            "performance problems, and style violations. Fix any issues you "
            "find. When you have absolutely completed every requirement—"
            "including ones you think don't matter—respond with exactly: "
            "'LOOP_COMPLETE: Code review finished.'"
        ),
        "end_condition": "LOOP_COMPLETE: Code review finished.",
    },
    "Refactor": {
        "prompt": (
            "Analyze the code you just wrote for opportunities to improve. "
            "Look for: duplicated code, overly complex logic, poor naming, "
            "violation of SOLID principles. Refactor where beneficial. When "
            "you have absolutely completed every requirement—including ones "
            "you think don't matter—respond with exactly: 'LOOP_COMPLETE: "
            "Refactoring finished.'"
        ),
        "end_condition": "LOOP_COMPLETE: Refactoring finished.",
    },
    "Documentation": {
        "prompt": (
            "Review the code you just wrote for documentation quality. Add or "
            "improve: docstrings for all public functions/classes, inline "
            "comments for complex logic, type hints for all parameters and "
            "return values. When you have absolutely completed every "
            "requirement—including ones you think don't matter—respond with "
            "exactly: 'LOOP_COMPLETE: Documentation complete.'"
        ),
        "end_condition": "LOOP_COMPLETE: Documentation complete.",
    },
    "Test Coverage": {
        "prompt": (
            "Analyze test coverage for the code you just wrote. Write "
            "additional tests for: edge cases, error conditions, boundary "
            "values, integration scenarios. Aim for 100% coverage. When you "
            "have absolutely completed every requirement—including ones you "
            "think don't matter—respond with exactly: 'LOOP_COMPLETE: Test "
            "coverage achieved.'"
        ),
        "end_condition": "LOOP_COMPLETE: Test coverage achieved.",
    },
}


def load_loop_prompts(prompts_file: str | None) -> dict[str, dict[str, str]]:
    """Load loop prompts from file or return defaults.

    Handles backward compatibility with old configs that stored prompts as strings.
    """
    import json
    if prompts_file:
        try:
            with open(prompts_file) as f:
                raw = json.load(f)
                # Normalize to new format
                normalized: dict[str, dict[str, str]] = {}
                for name, value in raw.items():
                    if isinstance(value, str):
                        normalized[name] = {"prompt": value, "end_condition": ""}
                    else:
                        normalized[name] = value
                return normalized
        except Exception:
            pass
    return DEFAULT_LOOP_PROMPTS.copy()


def save_config(
    port: int,
    notification_sound: bool,
    loop_prompts: dict[str, dict[str, str]],
    max_loop_iterations: int,
) -> None:
    """Save dashboard config for hooks to read.

    Merges with existing config to preserve other settings like federation,
    memory, quick_replies, and recent_directories.
    """
    # Load existing config to preserve other settings
    config = _get_full_config()

    # Update only the startup-related fields
    config["port"] = port
    config["notification_sound"] = notification_sound
    config["loop_prompts"] = loop_prompts
    config["max_loop_iterations"] = max_loop_iterations

    _save_full_config(config)


def main():
    """Run the dashboard server."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Augment Agent Dashboard")
    parser.add_argument(
        "-p", "--port", type=int, default=8080, help="Port to run the server on (default: 8080)"
    )
    parser.add_argument(
        "--sound", action="store_true", default=True,
        help="Enable notification sound (default: enabled)"
    )
    parser.add_argument(
        "--no-sound", action="store_true", help="Disable notification sound"
    )
    parser.add_argument(
        "--loop-prompts-file", type=str, default=None,
        help="JSON file with loop prompts dict {name: prompt}"
    )
    parser.add_argument(
        "--max-loop-iterations", type=int, default=50,
        help="Maximum number of loop iterations (default: 50)"
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload on file changes (for development)"
    )
    args = parser.parse_args()

    # Determine sound setting
    notification_sound = not args.no_sound

    # Load loop prompts
    loop_prompts = load_loop_prompts(args.loop_prompts_file)

    # Save config for hooks
    save_config(args.port, notification_sound, loop_prompts, args.max_loop_iterations)

    if args.reload:
        # For reload mode, uvicorn needs the app as a string import path
        uvicorn.run(
            "augment_agent_dashboard.server:app",
            host="0.0.0.0",
            port=args.port,
            reload=True,
            reload_dirs=["src/augment_agent_dashboard"],
        )
    else:
        uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

