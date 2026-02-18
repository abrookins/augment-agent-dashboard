"""Tests for the session state machine."""

import pytest

from augment_agent_dashboard.models import AgentSession
from augment_agent_dashboard.state_machine import (
    SessionState,
    SessionStateMachine,
    Transition,
    get_state_machine,
)


class TestSessionState:
    """Tests for SessionState enum."""

    def test_state_values(self):
        """Test enum values exist."""
        assert SessionState.IDLE.value == "idle"
        assert SessionState.ACTIVE.value == "active"
        assert SessionState.TURN_COMPLETE.value == "turn_complete"
        assert SessionState.REVIEW_PENDING.value == "review_pending"
        assert SessionState.UNDER_REVIEW.value == "under_review"
        assert SessionState.READY_FOR_LOOP.value == "ready_for_loop"
        assert SessionState.LOOP_PROMPTING.value == "loop_prompting"
        assert SessionState.ERROR.value == "error"

    def test_is_busy(self):
        """Test is_busy() method."""
        assert SessionState.ACTIVE.is_busy() is True
        assert SessionState.UNDER_REVIEW.is_busy() is True
        assert SessionState.LOOP_PROMPTING.is_busy() is True
        assert SessionState.IDLE.is_busy() is False
        assert SessionState.TURN_COMPLETE.is_busy() is False
        assert SessionState.REVIEW_PENDING.is_busy() is False

    def test_to_simple_status(self):
        """Test conversion to simple status."""
        assert SessionState.ACTIVE.to_simple_status() == "active"
        assert SessionState.UNDER_REVIEW.to_simple_status() == "active"
        assert SessionState.IDLE.to_simple_status() == "idle"
        assert SessionState.TURN_COMPLETE.to_simple_status() == "idle"
        assert SessionState.ERROR.to_simple_status() == "stopped"


class TestTransition:
    """Tests for Transition dataclass."""

    def test_create_transition(self):
        """Test creating a transition."""
        t = Transition(
            from_state=SessionState.IDLE,
            event="session_start",
            to_state=SessionState.ACTIVE,
        )
        assert t.from_state == SessionState.IDLE
        assert t.event == "session_start"
        assert t.to_state == SessionState.ACTIVE
        assert t.condition is None
        assert t.action is None

    def test_transition_with_condition(self):
        """Test transition with condition function."""

        def condition(s):
            return s.loop_enabled

        t = Transition(
            from_state=SessionState.READY_FOR_LOOP,
            event="evaluate",
            to_state=SessionState.LOOP_PROMPTING,
            condition=condition,
        )
        assert t.condition is condition


class TestSessionStateMachine:
    """Tests for SessionStateMachine."""

    @pytest.fixture
    def state_machine(self):
        """Create a fresh state machine."""
        return SessionStateMachine()

    @pytest.fixture
    def idle_session(self):
        """Create a session in IDLE state."""
        session = AgentSession(
            session_id="test-1",
            conversation_id="conv-1",
            workspace_root="/test",
            workspace_name="test",
        )
        session._state = "idle"
        return session

    def test_default_transitions_exist(self, state_machine):
        """Test that default transitions are created."""
        assert len(state_machine.transitions) > 0

    def test_session_start_transition(self, state_machine, idle_session):
        """Test IDLE -> ACTIVE on session_start."""
        result = state_machine.process_event(idle_session, "session_start")
        assert result.success is True
        assert result.old_state == SessionState.IDLE
        assert result.new_state == SessionState.ACTIVE

    def test_turn_end_transition(self, state_machine, idle_session):
        """Test ACTIVE -> TURN_COMPLETE on turn_end."""
        idle_session._state = "active"
        result = state_machine.process_event(idle_session, "turn_end")
        assert result.success is True
        assert result.new_state == SessionState.TURN_COMPLETE

    def test_no_matching_transition(self, state_machine, idle_session):
        """Test that invalid event returns failure."""
        result = state_machine.process_event(idle_session, "invalid_event")
        assert result.success is False
        assert result.old_state == SessionState.IDLE
        assert result.new_state == SessionState.IDLE

    def test_evaluate_with_review(self, state_machine, idle_session):
        """Test TURN_COMPLETE -> REVIEW_PENDING when review enabled."""
        idle_session._state = "turn_complete"
        idle_session.files_changed = ["file.py"]
        idle_session.review_enabled = True
        result = state_machine.process_event(idle_session, "evaluate")
        assert result.success is True
        assert result.new_state == SessionState.REVIEW_PENDING

    def test_evaluate_without_review(self, state_machine, idle_session):
        """Test TURN_COMPLETE -> READY_FOR_LOOP when no review needed."""
        idle_session._state = "turn_complete"
        idle_session.files_changed = []
        idle_session.review_enabled = False
        result = state_machine.process_event(idle_session, "evaluate")
        assert result.success is True
        assert result.new_state == SessionState.READY_FOR_LOOP

    def test_evaluate_loop_enabled(self, state_machine, idle_session):
        """Test READY_FOR_LOOP -> LOOP_PROMPTING when loop enabled."""
        idle_session._state = "ready_for_loop"
        idle_session.loop_enabled = True
        result = state_machine.process_event(idle_session, "evaluate")
        assert result.success is True
        assert result.new_state == SessionState.LOOP_PROMPTING

    def test_evaluate_loop_disabled(self, state_machine, idle_session):
        """Test READY_FOR_LOOP -> IDLE when loop disabled."""
        idle_session._state = "ready_for_loop"
        idle_session.loop_enabled = False
        result = state_machine.process_event(idle_session, "evaluate")
        assert result.success is True
        assert result.new_state == SessionState.IDLE

    def test_get_valid_events(self, state_machine):
        """Test getting valid events for a state."""
        events = state_machine.get_valid_events(SessionState.IDLE)
        assert "session_start" in events

    def test_can_transition(self, state_machine, idle_session):
        """Test can_transition check."""
        assert state_machine.can_transition(idle_session, "session_start") is True
        assert state_machine.can_transition(idle_session, "invalid") is False


class TestGetStateMachine:
    """Tests for get_state_machine singleton."""

    def test_returns_instance(self):
        """Test that get_state_machine returns an instance."""
        sm = get_state_machine()
        assert isinstance(sm, SessionStateMachine)

    def test_returns_same_instance(self):
        """Test singleton behavior."""
        sm1 = get_state_machine()
        sm2 = get_state_machine()
        assert sm1 is sm2

