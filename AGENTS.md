# Augment Agent Dashboard

Web dashboard for monitoring Augment coding agent sessions.

## Architecture

- **FastAPI server** (`server.py`) - Main web app with HTML templates inline
- **Augment hooks** (`hooks/`) - Capture session data via Augment's hook system
- **File-based storage** (`store.py`) - Sessions stored in `~/.augment/dashboard/sessions.json`
- **Models** (`models.py`) - Pydantic models for sessions and messages

## Key Concepts

### Augment Hooks
The dashboard uses Augment's hook system to capture session data:
- `SessionStart` - Fires when a session begins
- `Stop` - Fires after each agent turn (includes conversation data)
- `PostToolUse` - Fires after tool execution

Hooks are configured in `~/.augment/settings.json`. The install script (`install.py`) adds dashboard hooks without overwriting other plugins' hooks.

### Session IDs
Sessions use `conversation_id` from Augment directly. This is required for `auggie --resume <id>` to work correctly.

### Message Injection
Messages are sent to running sessions by spawning `auggie --resume <conversation_id> --print "message"`.

## CLI Commands

```bash
augment-dashboard          # Start the web server
augment-dashboard-install  # Install hooks into Augment
```

## Development

```bash
uv sync --dev              # Install dependencies
uv run augment-dashboard   # Run server
uv run pytest              # Run tests
uv run ruff check .        # Lint
```

## File Locations

- `~/.augment/settings.json` - Augment config with hooks
- `~/.augment/dashboard/sessions.json` - Session data
- `~/.augment/dashboard/config.json` - Dashboard config (loop prompts, etc.)
- `~/.augment/dashboard/hooks/` - Shell wrapper scripts for hooks

## Testing

No tests exist yet. When adding tests:
- Use pytest with pytest-asyncio
- Use testcontainers if Redis is needed
- Place tests in `tests/` directory

