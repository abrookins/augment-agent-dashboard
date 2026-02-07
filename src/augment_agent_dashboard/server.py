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

from .models import SessionStatus
from .store import SessionStore


def render_markdown(text: str) -> str:
    """Render markdown text to HTML."""
    return markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "nl2br"],
    )

app = FastAPI(title="Augment Agent Dashboard", version="0.1.0")

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


def get_store() -> SessionStore:
    """Get the session store instance."""
    return SessionStore()


def _get_loop_prompts() -> dict[str, str]:
    """Get loop prompts from config file."""
    import json
    config_path = Path.home() / ".augment" / "dashboard" / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            return config.get("loop_prompts", DEFAULT_LOOP_PROMPTS)
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


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard view showing all sessions."""
    store = get_store()
    sessions = store.get_all_sessions()

    # Determine dark mode from query param or default to system preference
    dark_mode = request.query_params.get("dark", None)
    sort_by = request.query_params.get("sort", "recent")

    # Sort sessions
    if sort_by == "name":
        sessions = sorted(sessions, key=lambda s: s.workspace_name.lower())
    # Default is "recent" which is already sorted by last_activity in get_all_sessions

    html = render_dashboard(sessions, dark_mode, sort_by)
    return HTMLResponse(content=html)


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
    store = get_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.conversation_id or session.conversation_id == "unknown":
        raise HTTPException(status_code=400, detail="Session has no conversation ID for resuming")

    if not session.workspace_root:
        raise HTTPException(status_code=400, detail="Session has no workspace root")

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
async def add_prompt(name: Annotated[str, Form()], prompt: Annotated[str, Form()]):
    """Add a new loop prompt."""
    config = _get_full_config()
    loop_prompts = config.get("loop_prompts", DEFAULT_LOOP_PROMPTS.copy())
    loop_prompts[name] = prompt
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
async def edit_prompt(name: Annotated[str, Form()], prompt: Annotated[str, Form()]):
    """Edit an existing loop prompt."""
    config = _get_full_config()
    loop_prompts = config.get("loop_prompts", DEFAULT_LOOP_PROMPTS.copy())
    loop_prompts[name] = prompt
    config["loop_prompts"] = loop_prompts
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


def format_time_ago(dt: datetime) -> str:
    """Format a datetime as a human-readable time ago string."""
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    diff = now - dt
    seconds = diff.total_seconds()

    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        mins = int(seconds / 60)
        return f"{mins}m ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}h ago"
    else:
        days = int(seconds / 86400)
        return f"{days}d ago"


def render_dashboard(sessions: list, dark_mode: str | None, sort_by: str = "recent") -> str:
    """Render the main dashboard HTML."""
    styles = get_base_styles(dark_mode)

    session_cards = ""
    if not sessions:
        session_cards = (
            '<div class="empty-state">'
            "No active sessions. Start an Augment conversation to see it here."
            "</div>"
        )
    else:
        for s in sessions:
            status_class = f"status-{s.status.value}"
            preview = s.last_message_preview or "No messages yet"
            time_ago = format_time_ago(s.last_activity)

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
        <meta http-equiv="refresh" content="10">
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
        <div class="session-list">
            {session_cards}
        </div>
        <script>
            {_get_notification_script()}
            // Auto-refresh every 10 seconds
            setTimeout(() => location.reload(), 10000);
        </script>
    </body>
    </html>
    """


def render_config_page(dark_mode: str | None, loop_prompts: dict[str, str], config: dict) -> str:
    """Render the configuration page HTML."""
    styles = get_base_styles(dark_mode)

    # Build prompt list
    prompts_html = ""
    for name, prompt in loop_prompts.items():
        escaped_name = html.escape(name)
        escaped_prompt = html.escape(prompt)
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
                <textarea name="prompt" rows="3">{escaped_prompt}</textarea>
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
            .prompt-edit-form textarea {{
                width: 100%;
                padding: 8px;
                border: 1px solid var(--border-color);
                border-radius: 4px;
                background: var(--bg-primary);
                color: var(--text-primary);
                font-family: inherit;
                font-size: 14px;
                resize: vertical;
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
        </style>
    </head>
    <body>
        <a href="/" class="back-link">← Back to Dashboard</a>
        <h1>⚙️ Configuration</h1>

        <h2>Loop Prompts</h2>
        <p style="color:var(--text-secondary);margin-bottom:15px;">
            These prompts are used for the quality loop feature. Select one when enabling a loop on a session.
        </p>

        {prompts_html}

        <div class="add-prompt-form">
            <h3>Add New Prompt</h3>
            <form method="POST" action="/config/prompts/add">
                <input type="text" name="name" placeholder="Prompt name (e.g., 'Security Review')" required>
                <textarea name="prompt" placeholder="Enter the prompt text..." required></textarea>
                <button type="submit" class="btn-enable" style="width:100%;">Add Prompt</button>
            </form>
        </div>
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


def _render_loop_controls(session, loop_prompts: dict[str, str]) -> str:
    """Render the loop control UI section."""
    if session.loop_enabled:
        elapsed = _format_elapsed_time(session.loop_started_at)
        prompt_name = session.loop_prompt_name or "Unknown"
        return f'''
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
        '''
    else:
        # Build dropdown options
        options = "".join(
            f'<option value="{html.escape(name)}">{html.escape(name)}</option>'
            for name in loop_prompts.keys()
        )
        return f'''
            <div class="loop-controls">
                <span style="color:var(--text-secondary);">Loop Paused</span>
                <form method="POST" action="/session/{session.session_id}/loop/enable">
                    <select name="prompt_name">{options}</select>
                    <button type="submit" class="btn-enable">▶ Enable</button>
                </form>
                <form method="POST" action="/session/{session.session_id}/loop/reset">
                    <button type="submit" class="btn-reset">↺ Reset</button>
                </form>
            </div>
        '''


def render_session_detail(session, dark_mode: str | None, loop_prompts: dict[str, str]) -> str:
    """Render the session detail HTML."""
    styles = get_base_styles(dark_mode)

    # Render message history
    messages_html = ""
    queued_count = 0
    if not session.messages:
        messages_html = '<div class="empty-state">No messages in this session yet.</div>'
    else:
        for msg in session.messages:
            role_class = msg.role
            time_str = msg.timestamp.strftime("%H:%M:%S") if msg.timestamp else ""

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

            messages_html += f"""
            <div class="message {role_class}">
                <div class="message-header">{role_label} • {time_str}</div>
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

    status_class = f"status-{session.status.value}"
    time_ago = format_time_ago(session.last_activity)

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
            {_render_loop_controls(session, loop_prompts)}
            <div class="loop-controls" style="margin-top:8px;">
                <form method="POST" action="/session/{session.session_id}/delete">
                    <button type="submit" onclick="return confirm('Delete this session?')" class="btn-delete">
                        🗑 Delete Session
                    </button>
                </form>
            </div>
        </div>

        <h2>Conversation</h2>
        <div class="message-list">
            {messages_html}
        </div>

        <div class="message-form">
            <h3>Send Message to Agent</h3>
            {_render_message_form(session)}
        </div>

        <script>
            // Auto-refresh every 5 seconds, but only if user isn't typing
            let refreshTimer;
            const textarea = document.getElementById('message-input');

            function scheduleRefresh() {{
                refreshTimer = setTimeout(() => location.reload(), 5000);
            }}

            textarea.addEventListener('focus', () => {{
                clearTimeout(refreshTimer);
            }});

            textarea.addEventListener('blur', () => {{
                if (!textarea.value.trim()) {{
                    scheduleRefresh();
                }}
            }});

            // Only start auto-refresh if textarea is empty and not focused
            if (document.activeElement !== textarea && !textarea.value.trim()) {{
                scheduleRefresh();
            }}
        </script>
    </body>
    </html>
    """


DEFAULT_LOOP_PROMPTS = {
    "TDD Quality": "Did you use TDD, reach 100% test coverage, and verify quality at .8 or above with mfcqi? If not, continue working. If choices must be made, choose wisely.",
    "Code Review": "Review the code you just wrote. Look for bugs, security issues, performance problems, and style violations. Fix any issues you find.",
    "Refactor": "Look at the code you just wrote. Can it be simplified, made more readable, or better organized? Refactor if so.",
    "Documentation": "Review the code you just wrote. Is it well-documented? Add or improve docstrings, comments, and type hints as needed.",
    "Test Coverage": "Check test coverage for the code you just wrote. Write additional tests to cover edge cases and error conditions.",
}


def load_loop_prompts(prompts_file: str | None) -> dict[str, str]:
    """Load loop prompts from file or return defaults."""
    import json
    if prompts_file:
        try:
            with open(prompts_file) as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_LOOP_PROMPTS.copy()


def save_config(port: int, notification_sound: bool, loop_prompts: dict[str, str], max_loop_iterations: int) -> None:
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

