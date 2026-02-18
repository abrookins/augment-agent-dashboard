"""State machine for managing agent session lifecycle.

This module defines the states and transitions for agent sessions,
including review cycles and loop prompts.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .models import AgentSession


class SessionState(str, Enum):
    """Detailed state of an agent session.

    States are organized into groups:
    - Core: IDLE, ACTIVE
    - Post-turn: TURN_COMPLETE
    - Review: REVIEW_PENDING, UNDER_REVIEW
    - Loop: READY_FOR_LOOP, LOOP_PROMPTING
    - Terminal: ERROR
    """

    # Core states
    IDLE = "idle"  # No agent activity
    ACTIVE = "active"  # Agent is working on a turn

    # Post-turn states
    TURN_COMPLETE = "turn_complete"  # Turn just ended, evaluating next action

    # Review states
    REVIEW_PENDING = "review_pending"  # Files changed, review needs to run
    UNDER_REVIEW = "under_review"  # Review agent is currently running

    # Loop states
    READY_FOR_LOOP = "ready_for_loop"  # Review done (or skipped), checking loop
    LOOP_PROMPTING = "loop_prompting"  # Loop prompt being sent

    # Terminal states
    ERROR = "error"  # Something went wrong

    def is_busy(self) -> bool:
        """Check if this state represents the agent being busy."""
        return self in (
            SessionState.ACTIVE,
            SessionState.UNDER_REVIEW,
            SessionState.LOOP_PROMPTING,
        )

    def to_simple_status(self) -> str:
        """Convert to simple status for backwards compatibility.

        Returns 'active', 'idle', or 'stopped'.
        """
        if self.is_busy():
            return "active"
        elif self == SessionState.ERROR:
            return "stopped"
        else:
            return "idle"


# Type alias for condition functions
ConditionFn = Callable[["AgentSession"], bool]
# Type alias for action functions
ActionFn = Callable[["AgentSession"], None]


@dataclass
class Transition:
    """A state machine transition.

    Attributes:
        from_state: The state to transition from.
        event: The event that triggers this transition.
        to_state: The state to transition to.
        condition: Optional function that must return True for transition to occur.
        action: Optional function to execute during the transition.
    """

    from_state: SessionState
    event: str
    to_state: SessionState
    condition: ConditionFn | None = None
    action: ActionFn | None = None


@dataclass
class TransitionResult:
    """Result of attempting a state transition.

    Attributes:
        success: Whether a transition occurred.
        old_state: The state before the transition attempt.
        new_state: The state after the transition (same as old if no transition).
        transition: The transition that was executed, if any.
    """

    success: bool
    old_state: SessionState
    new_state: SessionState
    transition: Transition | None = None


# Condition functions
def files_changed_and_review_enabled(session: "AgentSession") -> bool:
    """Check if files changed and review is enabled."""
    return bool(session.files_changed) and session.review_enabled


def no_review_needed(session: "AgentSession") -> bool:
    """Check if no review is needed (no changes or review disabled)."""
    return not session.files_changed or not session.review_enabled


def review_satisfied(session: "AgentSession") -> bool:
    """Check if review is satisfied or max iterations reached."""
    if session.review_iteration >= session.max_review_iterations:
        return True
    return session.review_satisfied


def review_not_satisfied(session: "AgentSession") -> bool:
    """Check if review is not yet satisfied."""
    return not review_satisfied(session)


def loop_enabled(session: "AgentSession") -> bool:
    """Check if loop is enabled for this session."""
    return session.loop_enabled


def loop_disabled(session: "AgentSession") -> bool:
    """Check if loop is disabled for this session."""
    return not session.loop_enabled


def in_review_cycle(session: "AgentSession") -> bool:
    """Check if we're in a review cycle."""
    return session.in_review_cycle


def not_in_review_cycle(session: "AgentSession") -> bool:
    """Check if we're not in a review cycle."""
    return not session.in_review_cycle


@dataclass
class SessionStateMachine:
    """State machine for managing session transitions.

    The state machine defines all valid state transitions and their conditions.
    It processes events and returns the new state.
    """

    transitions: list[Transition] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default transitions if none provided."""
        if not self.transitions:
            self.transitions = self._default_transitions()

    def _default_transitions(self) -> list[Transition]:
        """Define the default state transitions."""
        return [
            # Session start
            Transition(
                from_state=SessionState.IDLE,
                event="session_start",
                to_state=SessionState.ACTIVE,
            ),
            # Turn completion (not in review cycle)
            Transition(
                from_state=SessionState.ACTIVE,
                event="turn_end",
                to_state=SessionState.TURN_COMPLETE,
                condition=not_in_review_cycle,
            ),
            # Turn completion (in review cycle - agent responded to review)
            Transition(
                from_state=SessionState.ACTIVE,
                event="turn_end",
                to_state=SessionState.TURN_COMPLETE,
                condition=in_review_cycle,
            ),
            # Evaluate after turn - needs review
            Transition(
                from_state=SessionState.TURN_COMPLETE,
                event="evaluate",
                to_state=SessionState.REVIEW_PENDING,
                condition=files_changed_and_review_enabled,
            ),
            # Evaluate after turn - no review needed
            Transition(
                from_state=SessionState.TURN_COMPLETE,
                event="evaluate",
                to_state=SessionState.READY_FOR_LOOP,
                condition=no_review_needed,
            ),
            # Spawn reviewer
            Transition(
                from_state=SessionState.REVIEW_PENDING,
                event="spawn_reviewer",
                to_state=SessionState.UNDER_REVIEW,
            ),
            # Review feedback sent - back to active to let agent respond
            Transition(
                from_state=SessionState.UNDER_REVIEW,
                event="feedback_sent",
                to_state=SessionState.ACTIVE,
            ),
            # After review cycle, check if satisfied
            Transition(
                from_state=SessionState.TURN_COMPLETE,
                event="check_review",
                to_state=SessionState.READY_FOR_LOOP,
                condition=review_satisfied,
            ),
            Transition(
                from_state=SessionState.TURN_COMPLETE,
                event="check_review",
                to_state=SessionState.REVIEW_PENDING,
                condition=review_not_satisfied,
            ),
            # Loop evaluation
            Transition(
                from_state=SessionState.READY_FOR_LOOP,
                event="evaluate",
                to_state=SessionState.LOOP_PROMPTING,
                condition=loop_enabled,
            ),
            Transition(
                from_state=SessionState.READY_FOR_LOOP,
                event="evaluate",
                to_state=SessionState.IDLE,
                condition=loop_disabled,
            ),
            # Loop prompt sent
            Transition(
                from_state=SessionState.LOOP_PROMPTING,
                event="prompt_sent",
                to_state=SessionState.ACTIVE,
            ),
            # Error handling - can transition to error from most states
            Transition(
                from_state=SessionState.ACTIVE,
                event="error",
                to_state=SessionState.ERROR,
            ),
            Transition(
                from_state=SessionState.UNDER_REVIEW,
                event="error",
                to_state=SessionState.ERROR,
            ),
            Transition(
                from_state=SessionState.LOOP_PROMPTING,
                event="error",
                to_state=SessionState.ERROR,
            ),
            # Recovery from error
            Transition(
                from_state=SessionState.ERROR,
                event="reset",
                to_state=SessionState.IDLE,
            ),
            # Manual intervention - force idle
            Transition(
                from_state=SessionState.TURN_COMPLETE,
                event="force_idle",
                to_state=SessionState.IDLE,
            ),
            Transition(
                from_state=SessionState.REVIEW_PENDING,
                event="force_idle",
                to_state=SessionState.IDLE,
            ),
            Transition(
                from_state=SessionState.READY_FOR_LOOP,
                event="force_idle",
                to_state=SessionState.IDLE,
            ),
        ]

    def process_event(
        self, session: "AgentSession", event: str
    ) -> TransitionResult:
        """Process an event and return the transition result.

        Args:
            session: The session to evaluate conditions against.
            event: The event to process.

        Returns:
            TransitionResult with success status and state information.
        """
        old_state = session.state

        for transition in self.transitions:
            if transition.from_state != old_state:
                continue
            if transition.event != event:
                continue

            # Check condition if present
            if transition.condition is not None:
                if not transition.condition(session):
                    continue

            # Execute action if present
            if transition.action is not None:
                transition.action(session)

            # Update the session state
            session.state = transition.to_state

            return TransitionResult(
                success=True,
                old_state=old_state,
                new_state=transition.to_state,
                transition=transition,
            )

        # No matching transition found
        return TransitionResult(
            success=False,
            old_state=old_state,
            new_state=old_state,
            transition=None,
        )

    def get_valid_events(self, state: SessionState) -> list[str]:
        """Get all valid events for a given state.

        Args:
            state: The current state.

        Returns:
            List of event names that have transitions from this state.
        """
        events = set()
        for transition in self.transitions:
            if transition.from_state == state:
                events.add(transition.event)
        return sorted(events)

    def can_transition(
        self, session: "AgentSession", event: str
    ) -> bool:
        """Check if a transition is possible without executing it.

        Args:
            session: The session to check.
            event: The event to check.

        Returns:
            True if a matching transition exists and conditions are met.
        """
        for transition in self.transitions:
            if transition.from_state != session.state:
                continue
            if transition.event != event:
                continue
            if transition.condition is not None:
                if not transition.condition(session):
                    continue
            return True
        return False


# Global state machine instance
_state_machine: SessionStateMachine | None = None


def get_state_machine() -> SessionStateMachine:
    """Get the global state machine instance."""
    global _state_machine
    if _state_machine is None:
        _state_machine = SessionStateMachine()
    return _state_machine

