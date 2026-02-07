# Augment Agent Dashboard

A web dashboard for monitoring active Augment coding agents.

## Features

- View all active coding agent sessions
- See what each agent is working on and their workspace
- Monitor session status (active/idle/stopped)
- View conversation history for each session
- Send messages to agents (delivered on next session start)
- Dark/light mode support
- Mobile-friendly responsive design

## Installation

```bash
pip install augment-agent-dashboard
```

Or with uv:

```bash
uv pip install augment-agent-dashboard
```

## Usage

Start the dashboard server:

```bash
augment-dashboard
```

Then open http://localhost:8080 in your browser.

## Integration with Augment

Install the dashboard hooks to connect with your Augment coding agents:

```bash
augment-dashboard-install
```

This registers the dashboard's hooks with Augment. The hooks will:
- Register sessions when agents start
- Track conversation messages after each turn
- Deliver any pending messages you've sent from the dashboard

Session data is stored in `~/.augment/dashboard/sessions.json`.

## Configuration

The dashboard uses file-based storage with proper locking for concurrent access. Data is stored in:

- `~/.augment/dashboard/sessions.json` - Session data
- `~/.augment/dashboard/sessions.lock` - Lock file for concurrent access

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run linting
uv run ruff check src/

# Run tests
uv run pytest
```

