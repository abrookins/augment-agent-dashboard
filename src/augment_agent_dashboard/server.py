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
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {working_directory}")

    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Spawn auggie in background
    background_tasks.add_task(spawn_new_session, working_directory, prompt.strip())

    logger.info(f"Starting new session in {working_directory}")

    # Redirect back to dashboard
    return RedirectResponse(url="/", status_code=303)


@app.get("/session/{session_id}", response_class=HTMLResponse)
async def session_detail(session_id: str, request: Request):
    """Session detail view showing conversation history."""
    store = get_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    dark_mode = request.query_params.get("dark", None)
    loop_prompts = _get_loop_prompts()
    html = render_session_detail(session, dark_mode, loop_prompts)
    return HTMLResponse(content=html)


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
async def enable_loop(session_id: str, prompt_name: Annotated[str, Form()]):
    """Enable the quality loop for a session with a specific prompt."""
    store = get_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.loop_enabled = True
    session.loop_count = 0
    session.loop_prompt_name = prompt_name
    session.loop_started_at = datetime.now(timezone.utc)
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
    """Reset the loop counter for a session."""
    store = get_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.loop_count = 0
    store.upsert_session(session)

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
            --accent: #4a9eff;
            --status-active: #4ade80;
            --status-idle: #fbbf24;
            --status-stopped: #94a3b8;
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
                bannerText.innerHTML = '📱 <strong>Add to Home Screen</strong> for notifications: tap <span style="font-size:1.2em">⎙</span> then "Add to Home Screen"';
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
                bannerText.textContent = '🔔 Click to enable browser notifications for agent alerts';
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
                const response = await fetch('/api/notifications/poll?since=' + encodeURIComponent(lastNotificationId));
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
        status_class = f"status-{s.status.value}"
        preview = s.last_message_preview or "No messages yet"
        time_ago = format_time_ago(s.last_activity, include_title=True)

        ellipsis = "..." if len(preview) > 80 else ""
        session_cards += f"""
        <a href="/session/{s.session_id}" class="session-card">
            <div class="status-dot {status_class}" title="{s.status.value}"></div>
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


def render_dashboard(sessions: list, dark_mode: str | None, sort_by: str = "recent") -> str:
    """Render the main dashboard HTML."""
    styles = get_base_styles(dark_mode)

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
    </head>
    <body>
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
        <div id="notification-banner" style="display:none;background:var(--accent);color:white;padding:10px 15px;border-radius:8px;margin-bottom:15px;cursor:pointer;">
            🔔 <span id="notification-text">Enable browser notifications to get alerts on your phone</span>
        </div>
        <div class="new-session-section" style="margin-bottom:20px;">
            <button onclick="toggleNewSession()" class="btn-new-session" style="background:var(--accent);color:white;border:none;padding:10px 20px;border-radius:8px;cursor:pointer;font-size:1em;">
                ➕ New Session
            </button>
            <div id="new-session-form" style="display:none;margin-top:15px;background:var(--card-bg);padding:20px;border-radius:12px;border:1px solid var(--border);">
                <form method="POST" action="/session/new">
                    <div style="margin-bottom:15px;">
                        <label for="working_directory" style="display:block;margin-bottom:5px;font-weight:500;">Working Directory</label>
                        <input type="text" id="working_directory" name="working_directory" placeholder="/path/to/project" style="width:100%;padding:10px;border:1px solid var(--border);border-radius:6px;background:var(--bg);color:var(--text);box-sizing:border-box;">
                    </div>
                    <div style="margin-bottom:15px;">
                        <label for="prompt" style="display:block;margin-bottom:5px;font-weight:500;">Initial Prompt</label>
                        <textarea id="prompt" name="prompt" rows="4" placeholder="What would you like the agent to do?" style="width:100%;padding:10px;border:1px solid var(--border);border-radius:6px;background:var(--bg);color:var(--text);resize:vertical;box-sizing:border-box;"></textarea>
                    </div>
                    <button type="submit" style="background:var(--accent);color:white;border:none;padding:10px 20px;border-radius:6px;cursor:pointer;font-size:1em;">🚀 Start Session</button>
                </form>
            </div>
        </div>
        <script>
            function toggleNewSession() {{
                const form = document.getElementById('new-session-form');
                form.style.display = form.style.display === 'none' ? 'block' : 'none';
            }}
        </script>
        <div class="session-list" id="session-list">
            {session_cards}
        </div>
        <script>
            {_get_notification_script()}
            {_get_timestamp_script()}

            // AJAX-based session list updates
            const REFRESH_INTERVAL = 5000;
            const sortBy = '{sort_by}';

            function isUserInteracting() {{
                // Check if new session form is visible
                const newSessionForm = document.getElementById('new-session-form');
                if (newSessionForm && newSessionForm.style.display !== 'none') {{
                    return true;
                }}
                // Check if any input/textarea has focus
                const activeEl = document.activeElement;
                if (activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA')) {{
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

                try {{
                    const response = await fetch('/api/sessions-html?sort=' + encodeURIComponent(sortBy));
                    if (response.ok) {{
                        const html = await response.text();
                        document.getElementById('session-list').innerHTML = html;
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
    if is_local:
        new_session_btn = f'''
        <button onclick="openNewSession('local', '{html.escape(name)}')" class="btn-new-session" style="background:var(--accent);color:white;border:none;border-radius:8px;cursor:pointer;">
            ➕ New Session
        </button>
        '''
    else:
        disabled = ' disabled style="opacity:0.5;cursor:not-allowed;background:var(--border-color);color:var(--text-secondary);border:none;border-radius:8px;"' if not is_online else f' onclick="openNewSession(\'{html.escape(origin_url or "")}\', \'{html.escape(name)}\')" style="background:var(--accent);color:white;border:none;border-radius:8px;cursor:pointer;"'
        new_session_btn = f'''
        <button class="btn-new-session"{disabled}>
            ➕ New Session
        </button>
        '''

    return f'''
    <div class="{lane_class}" data-lane-id="{lane_id}" data-origin="{origin_url or 'local'}">
        <div class="swim-lane-header">
            <div class="swim-lane-title">
                💻 {html.escape(name)}
            </div>
            <div class="swim-lane-status">
                <span class="status-indicator {status_class}"></span>
                {status_text} · {session_count} session{"s" if session_count != 1 else ""}
            </div>
            {new_session_btn}
        </div>
        <div class="swim-lane-sessions" id="lane-sessions-{lane_id}">
            {session_cards if session_cards else '<div style="color:var(--text-secondary);text-align:center;padding:20px;">No sessions</div>'}
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
    </head>
    <body>
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

        <div id="notification-banner" style="display:none;background:var(--accent);color:white;padding:10px 15px;border-radius:8px;margin-bottom:15px;cursor:pointer;">
            🔔 <span id="notification-text">Enable browser notifications</span>
        </div>

        <div class="swim-lanes-container" id="swim-lanes">
            {lanes_html}
        </div>

        <div class="swim-lane-indicators">
            {lane_indicators}
        </div>

        <!-- New Session Modal -->
        <div id="new-session-overlay" class="new-session-overlay" onclick="if(event.target===this)closeNewSession()">
            <div class="new-session-modal">
                <h3>➕ New Session</h3>
                <div class="machine-label" id="new-session-machine">on: This Machine</div>
                <form id="new-session-form" method="POST" action="/session/new">
                    <input type="hidden" id="new-session-origin" name="origin" value="local">
                    <div style="margin-bottom:15px;">
                        <label style="display:block;margin-bottom:5px;font-weight:500;">Working Directory</label>
                        <input type="text" id="working_directory" name="working_directory" placeholder="/path/to/project" style="width:100%;padding:10px;border:1px solid var(--border-color);border-radius:6px;background:var(--bg-primary);color:var(--text-primary);box-sizing:border-box;">
                    </div>
                    <div style="margin-bottom:15px;">
                        <label style="display:block;margin-bottom:5px;font-weight:500;">Initial Prompt</label>
                        <textarea id="prompt" name="prompt" rows="4" placeholder="What would you like the agent to do?" style="width:100%;padding:10px;border:1px solid var(--border-color);border-radius:6px;background:var(--bg-primary);color:var(--text-primary);resize:vertical;box-sizing:border-box;"></textarea>
                    </div>
                    <div style="display:flex;gap:10px;">
                        <button type="button" onclick="closeNewSession()" style="flex:1;padding:10px;border:1px solid var(--border-color);border-radius:6px;background:transparent;color:var(--text-primary);cursor:pointer;">Cancel</button>
                        <button type="submit" style="flex:1;background:var(--accent);color:white;border:none;padding:10px;border-radius:6px;cursor:pointer;">🚀 Start</button>
                    </div>
                </form>
            </div>
        </div>

        <script>
            {_get_notification_script()}
            {_get_timestamp_script()}

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
                    form.action = '/api/federation/proxy/session/new?origin=' + encodeURIComponent(origin);
                }}

                document.getElementById('new-session-overlay').classList.add('active');
                document.getElementById('working_directory').focus();
            }}

            function closeNewSession() {{
                document.getElementById('new-session-overlay').classList.remove('active');
            }}

            // Close on Escape
            document.addEventListener('keydown', (e) => {{
                if (e.key === 'Escape') closeNewSession();
            }});

            // AJAX refresh for swim lanes
            const REFRESH_INTERVAL = 5000;
            const sortBy = '{sort_by}';

            async function refreshSwimLanes() {{
                // Skip if modal is open
                if (document.getElementById('new-session-overlay').classList.contains('active')) {{
                    scheduleRefresh();
                    return;
                }}

                try {{
                    const response = await fetch('/api/swimlanes-html?sort=' + encodeURIComponent(sortBy));
                    if (response.ok) {{
                        const html = await response.text();
                        document.getElementById('swim-lanes').innerHTML = html;
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
        <h2>🧠 Agent Memory</h2>
        <p style="color:var(--text-secondary);margin-bottom:15px;">
            Configure the Agent Memory Server to persist context across sessions.
            Settings are applied when hooks run (no restart needed).
        </p>

        <div class="prompt-card">
            <div class="memory-status">
                <span class="status-dot" style="background:{status_color};"></span>
                <strong>Status: {status_text}</strong>
            </div>

            <form method="POST" action="/config/memory">
                <label class="field-label">Memory Server URL:</label>
                <input type="text" name="server_url" value="{server_url}"
                       placeholder="http://localhost:8000">

                <label class="field-label">Namespace:</label>
                <input type="text" name="namespace" value="{namespace}"
                       placeholder="augment">

                <label class="field-label">User ID:</label>
                <input type="text" name="user_id" value="{user_id}"
                       placeholder="your-user-id">

                <label class="field-label">API Key (optional):</label>
                <input type="password" name="api_key" value="{api_key}"
                       placeholder="Leave empty if not required">

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

                <button type="submit" class="btn-enable" style="margin-top:15px;width:100%;">
                    Save Memory Settings
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
        <h2>🌐 Federation Settings</h2>
        <p style="color:var(--text-secondary);margin-bottom:15px;">
            Configure federation to connect multiple dashboards across machines.
        </p>

        <div class="prompt-card">
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
                       placeholder="e.g., Work Laptop">

                <label class="field-label">API Key (for incoming connections):</label>
                <input type="password" name="api_key" value="{api_key}"
                       placeholder="Leave empty for no authentication">

                <button type="submit" class="btn-enable" style="margin-top:15px;width:100%;">
                    Save Federation Settings
                </button>
            </form>
        </div>

        <div class="prompt-card" style="margin-top:20px;">
            <h3 style="margin-top:0;">Remote Dashboards</h3>
            <p style="color:var(--text-secondary);font-size:0.9em;margin-bottom:15px;">
                Add other machines' dashboards to see their sessions.
            </p>

            <div class="remotes-list">
                {remotes_html}
            </div>

            <hr style="margin:15px 0;border:none;border-top:1px solid var(--border-color);">

            <form method="POST" action="/config/federation/remotes/add">
                <label class="field-label">Dashboard URL:</label>
                <input type="text" name="url" placeholder="http://other-machine:8080" required>

                <label class="field-label">Name:</label>
                <input type="text" name="name" placeholder="e.g., Home Desktop" required>

                <label class="field-label">API Key (if required):</label>
                <input type="password" name="remote_api_key"
                       placeholder="Leave empty if not required">

                <button type="submit" class="btn-enable" style="margin-top:10px;width:100%;">
                    Add Remote Dashboard
                </button>
            </form>
        </div>
    '''


def render_config_page(
    dark_mode: str | None,
    loop_prompts: dict[str, dict[str, str]],
    config: dict,
) -> str:
    """Render the configuration page HTML."""
    styles = get_base_styles(dark_mode)

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
        <div class="prompt-card">
            <div class="prompt-header">
                <strong>{escaped_name}</strong>
                <form method="POST" action="/config/prompts/delete" style="display:inline;">
                    <input type="hidden" name="name" value="{escaped_name}">
                    <button type="submit" onclick="return confirm('Delete this prompt?')" class="btn-delete" style="padding:4px 8px;font-size:0.8em;">🗑</button>
                </form>
            </div>
            <form method="POST" action="/config/prompts/edit" class="prompt-edit-form">
                <input type="hidden" name="name" value="{escaped_name}">
                <label class="field-label">Prompt (instructions for the LLM):</label>
                <textarea name="prompt" rows="4">{escaped_prompt}</textarea>
                <label class="field-label">End Condition (text that stops the loop when found in response):</label>
                <input type="text" name="end_condition" value="{escaped_condition}" placeholder="e.g., LOOP_COMPLETE: Task finished.">
                <button type="submit" class="btn-enable" style="margin-top:8px;">Save</button>
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
            .prompt-card {{
                background: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 12px;
            }}
            .prompt-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }}
            .prompt-edit-form textarea,
            .prompt-edit-form input[type="text"] {{
                width: 100%;
                padding: 8px;
                border: 1px solid var(--border-color);
                border-radius: 4px;
                background: var(--bg-primary);
                color: var(--text-primary);
                font-family: inherit;
                font-size: 14px;
                resize: vertical;
                margin-bottom: 8px;
            }}
            .prompt-edit-form input[type="text"] {{
                resize: none;
            }}
            .field-label {{
                display: block;
                font-size: 0.85em;
                color: var(--text-secondary);
                margin-bottom: 4px;
                margin-top: 8px;
            }}
            .field-label:first-of-type {{
                margin-top: 0;
            }}
            .add-prompt-form {{
                background: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 15px;
                margin-top: 20px;
            }}
            .add-prompt-form input[type="text"] {{
                width: 100%;
                padding: 10px;
                border: 1px solid var(--border-color);
                border-radius: 4px;
                background: var(--bg-primary);
                color: var(--text-primary);
                font-size: 14px;
                margin-bottom: 10px;
            }}
            .add-prompt-form textarea {{
                width: 100%;
                padding: 10px;
                border: 1px solid var(--border-color);
                border-radius: 4px;
                background: var(--bg-primary);
                color: var(--text-primary);
                font-family: inherit;
                font-size: 14px;
                resize: vertical;
                min-height: 80px;
                margin-bottom: 10px;
            }}
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

        <h2>Loop Prompts</h2>
        <p style="color:var(--text-secondary);margin-bottom:15px;">
            Configure loop prompts with end conditions. The prompt tells the LLM what to do and should explain
            the end condition. When the LLM includes the end condition text in its response, the loop stops.
        </p>

        {prompts_html}

        <div class="add-prompt-form">
            <h3>Add New Prompt</h3>
            <form method="POST" action="/config/prompts/add">
                <label class="field-label">Name:</label>
                <input type="text" name="name" placeholder="Prompt name (e.g., 'Security Review')" required>
                <label class="field-label">Prompt (instructions for the LLM):</label>
                <textarea name="prompt" placeholder="Enter instructions. Include what the LLM should say when done, e.g., 'When finished, say LOOP_COMPLETE: Security review done.'" required></textarea>
                <label class="field-label">End Condition (text that stops the loop):</label>
                <input type="text" name="end_condition" placeholder="e.g., LOOP_COMPLETE: Security review done.">
                <button type="submit" class="btn-enable" style="width:100%;">Add Prompt</button>
            </form>
        </div>

        <hr style="margin: 30px 0; border: none; border-top: 1px solid var(--border-color);">

        {_render_federation_config_section(config)}

        <hr style="margin: 30px 0; border: none; border-top: 1px solid var(--border-color);">

        {_render_memory_config_section(config)}
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


def _render_message_form(session) -> str:
    """Render the message form with queue support."""
    from augment_agent_dashboard.models import SessionStatus

    # Count queued messages
    queued_count = sum(1 for m in session.messages if m.role == "queued")
    queue_info = f'<span style="color:var(--text-secondary);font-size:0.9em;">({queued_count} queued)</span>' if queued_count > 0 else ""

    if session.status == SessionStatus.ACTIVE:
        return f'''
            <div style="background:var(--status-active);color:#000;padding:12px;border-radius:8px;margin-bottom:10px;">
                ⏳ Agent is currently working. Messages will be queued and sent when ready. {queue_info}
            </div>
            <form method="POST" action="/session/{session.session_id}/queue">
                <textarea id="message-input" name="message" placeholder="Type a message to queue..."></textarea>
                <button type="submit" class="btn-queue">🕐 Enqueue Message</button>
            </form>
        '''
    else:
        return f'''
            <p style="color: var(--text-secondary); margin-bottom: 10px;">
                Send a message directly, or queue it for later. {queue_info}
            </p>
            <form method="POST" action="/session/{session.session_id}/message" style="margin-bottom:10px;">
                <textarea id="message-input" name="message" placeholder="Type a message for the agent..."></textarea>
                <div style="display:flex;gap:8px;flex-wrap:wrap;">
                    <button type="submit">▶ Send Now</button>
                    <button type="submit" formaction="/session/{session.session_id}/queue" class="btn-queue">🕐 Enqueue</button>
                </div>
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
                tooltip = f"Prompt: {prompt[:80]}..." if len(prompt) > 80 else f"Prompt: {prompt}"
                if end_cond:
                    tooltip += f"\n\nStops when: {end_cond}"
            escaped_tooltip = html.escape(tooltip)
            options_html += f'<option value="{escaped_name}" title="{escaped_tooltip}">{escaped_name}</option>'

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
        messages_html += f'''
        <div class="queue-actions">
            <form method="POST" action="/session/{session.session_id}/queue/clear">
                <button type="submit" class="btn-delete" style="font-size:0.85em;" onclick="return confirm('Clear all {queued_count} queued messages?')">
                    🗑 Clear Queue ({queued_count})
                </button>
            </form>
        </div>
        '''

    return messages_html, queued_count


def _render_session_status_html(session) -> str:
    """Render the session status indicator HTML."""
    time_ago = format_time_ago(session.last_activity, include_title=True)
    return f"""
        <div>{session.status.value} • {time_ago}</div>
        <div>{session.message_count} messages</div>
    """


def render_session_detail(session, dark_mode: str | None, loop_prompts: dict[str, dict[str, str]]) -> str:
    """Render the session detail HTML."""
    styles = get_base_styles(dark_mode)

    # Render message history
    messages_html, queued_count = _render_messages_html(session)

    status_class = f"status-{session.status.value}"
    time_ago = format_time_ago(session.last_activity, include_title=True)

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
        {styles}
    </head>
    <body>
        <a href="/" class="back-link">← Back to Dashboard</a>

        <div class="header">
            <h1>
                <span class="status-dot {status_class}"
                    style="display:inline-block;vertical-align:middle;margin-right:10px;">
                </span>
                {session.workspace_name}
            </h1>
            <div class="session-meta">
                <div>{session.status.value} • {time_ago}</div>
                <div>{session.message_count} messages</div>
            </div>
        </div>

        <div class="session-detail-meta">
            <strong>Workspace:</strong> {session.workspace_root}<br>
            <strong>Session ID:</strong> {session.session_id}
            <div id="loop-controls-container">
                {_render_loop_controls(session, loop_prompts)}
            </div>
            <div class="loop-controls" style="margin-top:8px;">
                <form method="POST" action="/session/{session.session_id}/delete">
                    <button type="submit" onclick="return confirm('Delete this session?')" class="btn-delete">
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
                    const response = await fetch('/api/sessions/' + encodeURIComponent(sessionId) + '/messages-html');
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
                        const wasAtBottom = messageList.scrollHeight - messageList.scrollTop <= messageList.clientHeight + 100;
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

                    // Check if there's a "Send Now" button (session is IDLE)
                    const sendBtn = form.querySelector('button[type="submit"]:not(.btn-queue)');
                    if (sendBtn) {{
                        // Session is idle - submit via send button
                        form.submit();
                    }} else {{
                        // Session is active or only queue available - submit to queue
                        const queueBtn = form.querySelector('.btn-queue');
                        if (queueBtn) {{
                            form.submit();
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """


DEFAULT_LOOP_PROMPTS: dict[str, dict[str, str]] = {
    "TDD Quality": {
        "prompt": (
            "Continue working on this task using TDD. Write tests first, then implement code to pass them. "
            "Verify code quality with mfcqi (target score >= 0.8). When you have absolutely completed every "
            "requirement—including ones you think don't matter—respond with exactly: 'LOOP_COMPLETE: TDD quality goals achieved.'"
        ),
        "end_condition": "LOOP_COMPLETE: TDD quality goals achieved.",
    },
    "Code Review": {
        "prompt": (
            "Review the code you just wrote. Look for bugs, security issues, performance problems, and style violations. "
            "Fix any issues you find. When you have absolutely completed every requirement—including ones you think "
            "don't matter—respond with exactly: 'LOOP_COMPLETE: Code review finished.'"
        ),
        "end_condition": "LOOP_COMPLETE: Code review finished.",
    },
    "Refactor": {
        "prompt": (
            "Analyze the code you just wrote for opportunities to improve. Look for: duplicated code, overly complex logic, "
            "poor naming, violation of SOLID principles. Refactor where beneficial. When you have absolutely completed every "
            "requirement—including ones you think don't matter—respond with exactly: 'LOOP_COMPLETE: Refactoring finished.'"
        ),
        "end_condition": "LOOP_COMPLETE: Refactoring finished.",
    },
    "Documentation": {
        "prompt": (
            "Review the code you just wrote for documentation quality. Add or improve: docstrings for all public "
            "functions/classes, inline comments for complex logic, type hints for all parameters and return values. "
            "When you have absolutely completed every requirement—including ones you think don't matter—respond with "
            "exactly: 'LOOP_COMPLETE: Documentation complete.'"
        ),
        "end_condition": "LOOP_COMPLETE: Documentation complete.",
    },
    "Test Coverage": {
        "prompt": (
            "Analyze test coverage for the code you just wrote. Write additional tests for: edge cases, error conditions, "
            "boundary values, integration scenarios. Aim for 100% coverage. When you have absolutely completed every "
            "requirement—including ones you think don't matter—respond with exactly: 'LOOP_COMPLETE: Test coverage achieved.'"
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
    """Save dashboard config for hooks to read."""
    import json
    config_dir = Path.home() / ".augment" / "dashboard"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps({
        "port": port,
        "notification_sound": notification_sound,
        "loop_prompts": loop_prompts,
        "max_loop_iterations": max_loop_iterations,
    }))


def main():
    """Run the dashboard server."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Augment Agent Dashboard")
    parser.add_argument(
        "-p", "--port", type=int, default=8080, help="Port to run the server on (default: 8080)"
    )
    parser.add_argument(
        "--sound", action="store_true", default=True, help="Enable notification sound (default: enabled)"
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
    args = parser.parse_args()

    # Determine sound setting
    notification_sound = not args.no_sound

    # Load loop prompts
    loop_prompts = load_loop_prompts(args.loop_prompts_file)

    # Save config for hooks
    save_config(args.port, notification_sound, loop_prompts, args.max_loop_iterations)

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

