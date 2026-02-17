"""Federation API routes for cross-dashboard communication."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from ..store import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/federation", tags=["federation"])


def get_store() -> SessionStore:
    """Get the session store instance."""
    return SessionStore()


def _get_federation_config():
    """Get federation config from the main config."""
    import json
    from pathlib import Path

    config_path = Path.home() / ".augment" / "dashboard" / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            return config.get("federation", {})
        except Exception:
            pass
    return {}


def verify_api_key(
    x_dashboard_api_key: Annotated[str | None, Header()] = None,
) -> bool:
    """Verify the API key if one is configured.

    Returns True if access is allowed, raises HTTPException otherwise.
    """
    fed_config = _get_federation_config()

    # Check if federation sharing is enabled
    if not fed_config.get("share_locally", True):
        raise HTTPException(status_code=403, detail="Federation sharing disabled")

    # Check API key if one is required
    required_key = fed_config.get("api_key")
    if required_key:
        if not x_dashboard_api_key or x_dashboard_api_key != required_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return True


@router.get("/health")
async def health_check():
    """Health check endpoint for remote dashboards to verify connectivity."""
    return {"status": "ok", "federation": "enabled"}


@router.get("/sessions")
async def list_sessions(
    _authorized: bool = Depends(verify_api_key),
    store: SessionStore = Depends(get_store),
):
    """List all local sessions for federation.

    This is the endpoint remote dashboards call to fetch our sessions.
    """
    sessions = store.get_all_sessions()

    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "conversation_id": s.conversation_id,
                "workspace_root": s.workspace_root,
                "workspace_name": s.workspace_name,
                "status": s.status.value,
                "started_at": s.started_at.isoformat(),
                "last_activity": s.last_activity.isoformat(),
                "current_task": s.current_task,
                "message_count": s.message_count,
                "last_message_preview": s.last_message_preview,
            }
            for s in sessions
        ]
    }


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    _authorized: bool = Depends(verify_api_key),
    store: SessionStore = Depends(get_store),
):
    """Get details of a specific session."""
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session.to_dict()


class MessageRequest(BaseModel):
    """Request body for sending a message."""
    message: str


class NewSessionRequest(BaseModel):
    """Request body for creating a new session."""
    workspace_root: str
    prompt: str


@router.post("/sessions/{session_id}/message")
async def send_message(
    session_id: str,
    request: MessageRequest,
    _authorized: bool = Depends(verify_api_key),
    store: SessionStore = Depends(get_store),
):
    """Send a message to a local session (from a remote dashboard)."""
    from ..models import SessionMessage, SessionStatus

    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.conversation_id or session.conversation_id == "unknown":
        raise HTTPException(status_code=400, detail="Session has no conversation ID")

    if not session.workspace_root:
        raise HTTPException(status_code=400, detail="Session has no workspace root")

    # Add the user message
    user_msg = SessionMessage(role="user", content=request.message.strip())
    store.add_message(session_id, user_msg)
    store.update_session_status(session_id, SessionStatus.ACTIVE)

    # Spawn auggie in background (import here to avoid circular import)
    import asyncio

    from ..server import spawn_auggie_message

    asyncio.create_task(
        spawn_auggie_message(
            session.conversation_id,
            session.workspace_root,
            request.message,
        )
    )

    return {"status": "ok", "message": "Message sent"}


@router.post("/sessions/new")
async def create_session(
    request: NewSessionRequest,
    _authorized: bool = Depends(verify_api_key),
):
    """Create a new session on this machine (from a remote dashboard)."""
    import os

    # Validate workspace
    workspace = os.path.expanduser(request.workspace_root.strip())
    if not os.path.isdir(workspace):
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {workspace}")

    # Spawn auggie (import here to avoid circular import)
    import asyncio

    from ..server import spawn_new_session

    asyncio.create_task(spawn_new_session(workspace, request.prompt.strip()))

    return {"status": "ok", "message": "Session creation started"}

