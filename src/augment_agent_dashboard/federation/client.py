"""HTTP client for fetching sessions from remote dashboards."""

import hashlib
import logging
from datetime import datetime, timezone

import httpx

from .models import RemoteDashboard, RemoteSession

logger = logging.getLogger(__name__)

# Timeout for remote requests
REQUEST_TIMEOUT = 3.0


def _generate_federated_session_id(origin_url: str, remote_session_id: str) -> str:
    """Generate a unique session ID for a remote session.

    Format: remote-{hash}-{original_id}
    This ensures no collisions between different remotes.
    """
    url_hash = hashlib.md5(origin_url.encode()).hexdigest()[:8]
    return f"remote-{url_hash}-{remote_session_id}"


class RemoteDashboardClient:
    """Client for communicating with remote dashboard servers."""

    def __init__(self, remote: RemoteDashboard):
        """Initialize the client.

        Args:
            remote: The remote dashboard configuration.
        """
        self.remote = remote
        self.base_url = remote.url.rstrip("/")

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers including API key if configured."""
        headers = {
            "Accept": "application/json",
            "User-Agent": "AugmentDashboard/1.0",
        }
        if self.remote.api_key:
            headers["X-Dashboard-Api-Key"] = self.remote.api_key
        return headers

    async def health_check(self) -> bool:
        """Check if the remote dashboard is reachable.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.get(
                    f"{self.base_url}/api/federation/health",
                    headers=self._get_headers(),
                )
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed for {self.remote.name}: {e}")
            return False

    async def fetch_sessions(self) -> list[RemoteSession]:
        """Fetch all sessions from the remote dashboard.

        Returns:
            List of RemoteSession objects, or empty list on failure.
        """
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.get(
                    f"{self.base_url}/api/federation/sessions",
                    headers=self._get_headers(),
                )

                if response.status_code == 401:
                    logger.warning(f"Auth failed for {self.remote.name}: API key rejected")
                    self.remote.is_healthy = False
                    return []

                if response.status_code != 200:
                    logger.warning(
                        f"Failed to fetch from {self.remote.name}: {response.status_code}"
                    )
                    self.remote.is_healthy = False
                    return []

                data = response.json()
                sessions = []

                for s in data.get("sessions", []):
                    remote_session = RemoteSession(
                        session_id=_generate_federated_session_id(self.remote.url, s["session_id"]),
                        conversation_id=s.get("conversation_id", ""),
                        workspace_root=s.get("workspace_root", ""),
                        workspace_name=s.get("workspace_name", "Unknown"),
                        status=s.get("status", "stopped"),
                        started_at=s.get("started_at", ""),
                        last_activity=s.get("last_activity", ""),
                        current_task=s.get("current_task"),
                        message_count=s.get("message_count", 0),
                        last_message_preview=s.get("last_message_preview"),
                        origin_url=self.remote.url,
                        origin_name=self.remote.name,
                        remote_session_id=s["session_id"],
                    )
                    sessions.append(remote_session)

                self.remote.is_healthy = True
                self.remote.last_seen = datetime.now(timezone.utc)
                return sessions

        except httpx.TimeoutException:
            logger.debug(f"Timeout fetching from {self.remote.name}")
            self.remote.is_healthy = False
            return []
        except Exception as e:
            logger.debug(f"Error fetching from {self.remote.name}: {e}")
            self.remote.is_healthy = False
            return []

    async def send_message(self, remote_session_id: str, message: str) -> bool:
        """Send a message to a session on the remote dashboard.

        Args:
            remote_session_id: The original session ID on the remote.
            message: The message to send.

        Returns:
            True if successful, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/federation/sessions/{remote_session_id}/message",
                    headers=self._get_headers(),
                    json={"message": message},
                )
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Failed to send message to {self.remote.name}: {e}")
            return False

    async def create_session(self, workspace_root: str, prompt: str) -> dict | None:
        """Create a new session on the remote dashboard.

        Args:
            workspace_root: Working directory for the session.
            prompt: Initial prompt for the agent.

        Returns:
            Session data dict if successful, None otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/federation/sessions/new",
                    headers=self._get_headers(),
                    json={"workspace_root": workspace_root, "prompt": prompt},
                )
                if response.status_code == 200:
                    return response.json()
                return None
        except Exception as e:
            logger.warning(f"Failed to create session on {self.remote.name}: {e}")
            return None

