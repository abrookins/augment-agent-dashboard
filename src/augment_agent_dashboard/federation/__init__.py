"""Federation module for multi-machine dashboard support."""

from .client import RemoteDashboardClient
from .models import FederationConfig, RemoteDashboard

__all__ = ["FederationConfig", "RemoteDashboard", "RemoteDashboardClient"]

