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


async def spawn_auggie_message(conversation_id: str, workspace_root: str, message: str) -> bool:
    """Spawn auggie subprocess to inject a message into a session.

    Returns True if successful, False otherwise.
    """
    auggie_path = shutil.which("auggie")
    if not auggie_path:
        return False

    try:
        process = await asyncio.create_subprocess_exec(
            auggie_path,
            "--resume", conversation_id,
            "--print", message,
            cwd=workspace_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.wait()
        return process.returncode == 0
    except Exception:
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
    html = render_session_detail(session, dark_mode)
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
async def enable_loop(session_id: str):
    """Enable the quality loop for a session."""
    store = get_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.loop_enabled = True
    session.loop_count = 0
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
        {styles}
    </head>
    <body>
        <div class="header">
            <h1>🤖 Augment Agent Dashboard</h1>
            <div>
                Sort: <a href="?sort=recent{dark_param}" style="{recent_active}">Recent</a> |
                <a href="?sort=name{dark_param}" style="{name_active}">Name</a>
                &nbsp;|&nbsp;
                <a href="?dark=true&sort={sort_by}">Dark</a> |
                <a href="?dark=false&sort={sort_by}">Light</a> |
                <a href="?sort={sort_by}">Auto</a>
            </div>
        </div>
        <div class="session-list">
            {session_cards}
        </div>
        <script>
            // Auto-refresh every 10 seconds
            setTimeout(() => location.reload(), 10000);
        </script>
    </body>
    </html>
    """


def render_session_detail(session, dark_mode: str | None) -> str:
    """Render the session detail HTML."""
    styles = get_base_styles(dark_mode)

    # Render message history
    messages_html = ""
    if not session.messages:
        messages_html = '<div class="empty-state">No messages in this session yet.</div>'
    else:
        for msg in session.messages:
            role_class = msg.role
            role_label = msg.role.capitalize()
            time_str = msg.timestamp.strftime("%H:%M:%S") if msg.timestamp else ""
            # Render markdown for assistant messages, escape HTML for user messages
            if msg.role == "assistant":
                content_html = render_markdown(msg.content)
            else:
                content_html = f"<p>{html.escape(msg.content)}</p>"

            messages_html += f"""
            <div class="message {role_class}">
                <div class="message-header">{role_label} • {time_str}</div>
                <div class="message-content">{content_html}</div>
            </div>
            """

    status_class = f"status-{session.status.value}"
    time_ago = format_time_ago(session.last_activity)

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{session.workspace_name} - Augment Dashboard</title>
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
            <strong>Session ID:</strong> {session.session_id}<br>
            <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
                {"<span style='color:var(--status-active);font-weight:bold;'>🔄 Loop Active (" + str(session.loop_count) + " iterations)</span>" if session.loop_enabled else "<span style='color:var(--text-secondary);'>Loop Paused</span>"}
                {'''<form method="POST" action="/session/''' + session.session_id + '''/loop/pause" style="display:inline;">
                    <button type="submit" style="background:#fbbf24;color:#000;border:none;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:0.85em;">
                        ⏸ Pause Loop
                    </button>
                </form>''' if session.loop_enabled else '''<form method="POST" action="/session/''' + session.session_id + '''/loop/enable" style="display:inline;">
                    <button type="submit" style="background:var(--status-active);color:#000;border:none;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:0.85em;">
                        ▶ Enable Loop
                    </button>
                </form>'''}
                <form method="POST" action="/session/{session.session_id}/loop/reset" style="display:inline;">
                    <button type="submit" style="background:var(--text-secondary);color:#fff;border:none;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:0.85em;">
                        ↺ Reset Count
                    </button>
                </form>
                <form method="POST" action="/session/{session.session_id}/delete" style="display:inline;">
                    <button type="submit" onclick="return confirm('Delete this session?')"
                        style="background:#dc2626;color:white;border:none;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:0.85em;">
                        Delete Session
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
            <p style="color: var(--text-secondary); margin-bottom: 10px;">
                This will spawn a new auggie process to handle your message in this session.
            </p>
            <form method="POST" action="/session/{session.session_id}/message">
                <textarea id="message-input" name="message" placeholder="Type a message for the agent..."></textarea>
                <button type="submit">Send Message</button>
            </form>
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


DEFAULT_LOOP_PROMPT = """Did you use TDD, reach 100% test coverage, and verify quality at .8 or above with mfcqi? If not, continue working. If choices must be made, choose wisely."""


def save_config(port: int, notification_sound: bool, loop_prompt: str, max_loop_iterations: int) -> None:
    """Save dashboard config for hooks to read."""
    import json
    config_dir = Path.home() / ".augment" / "dashboard"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps({
        "port": port,
        "notification_sound": notification_sound,
        "loop_prompt": loop_prompt,
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
        "--loop-prompt", type=str, default=DEFAULT_LOOP_PROMPT,
        help="Prompt to send after each turn when loop is enabled"
    )
    parser.add_argument(
        "--max-loop-iterations", type=int, default=50,
        help="Maximum number of loop iterations (default: 50)"
    )
    args = parser.parse_args()

    # Determine sound setting
    notification_sound = not args.no_sound

    # Save config for hooks
    save_config(args.port, notification_sound, args.loop_prompt, args.max_loop_iterations)

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

