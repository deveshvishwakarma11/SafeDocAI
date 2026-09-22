"""
SafeDocAI - Conversation State Layer (Step 1)
=============================================

Minimal, in-memory-only conversation state. This step makes PREVIOUS
conversation turns available to the answer engine as READ-ONLY CONTEXT;
it deliberately does NOT interpret that context.

Scope guard (approved roadmap, Step 1 only):

* Stores user queries, assistant responses and the (optional) already-
  detected language per turn. Nothing else.
* NO ambiguity detection, NO clarification generation, NO follow-up or
  pending-request handling, and NO pronoun/context resolution ("it",
  "that", "uska", "wahi", ...). The current query's meaning is never
  changed by this layer. Context-aware interpretation is a LATER step.
* Pure Python. No SQLite, no ChromaDB, no files, no data/ directory, no
  network, no new dependencies. State lives and dies with the process
  (the UI keeps it in ``st.session_state``); nothing is persisted.

Structure is designed for future extension (clarification / pending
requests / context resolution) without schema churn: a turn is an
immutable record, the state is an ordered, bounded ring of turns, and
:func:`ConversationState.to_context` emits the plain-dict shape the
answer engine receives as its optional ``context`` parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


__all__ = [
    "MAX_STATE_TURNS",
    "ConversationTurn",
    "ConversationState",
]


#: Default cap on stored turns. Mirrors the UI's MAX_HISTORY_ENTRIES so
#: the engine-side state never grows beyond what the UI session keeps.
MAX_STATE_TURNS = 10


@dataclass(frozen=True)
class ConversationTurn:
    """One immutable conversation turn.

    Attributes:
        user_query: The user's message for this turn (stripped).
        assistant_response: The assistant's reply ("" when still pending).
        language: Already-detected language of the user query
            (``None`` when unknown). Preserved verbatim when provided;
            detection itself is NOT performed here.
    """

    user_query: str
    assistant_response: str = ""
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict shape of this turn (JSON-friendly, order-stable)."""

        return {
            "user_query": self.user_query,
            "assistant_response": self.assistant_response,
            "language": self.language,
        }


class ConversationState:
    """Ordered, bounded, in-memory conversation turns (oldest first).

    Pure state container: same input sequence -> same state, always.
    Adding a turn beyond ``max_turns`` silently drops the oldest turn
    so the state stays bounded exactly like the UI history.
    """

    __slots__ = ("_turns", "_max_turns")

    def __init__(self, max_turns: int = MAX_STATE_TURNS) -> None:
        if int(max_turns) < 1:
            raise ValueError("max_turns must be at least 1.")
        self._max_turns = int(max_turns)
        self._turns: list[ConversationTurn] = []

    # ------------------------------------------------------------------
    # Mutation (single method; no interpretation of any kind)
    # ------------------------------------------------------------------

    def add_turn(
        self,
        user_query: str,
        assistant_response: str = "",
        language: str | None = None,
    ) -> ConversationTurn:
        """Record one turn; returns the stored (immutable) turn.

        Defensive only: non-string inputs are coerced and the query is
        stripped. No parsing, no language detection, no interpretation.
        """

        query = "" if user_query is None else str(user_query).strip()
        response = "" if assistant_response is None else str(assistant_response)
        lang = None if language is None else str(language)

        turn = ConversationTurn(
            user_query=query,
            assistant_response=response,
            language=lang,
        )

        self._turns.append(turn)
        while len(self._turns) > self._max_turns:
            self._turns.pop(0)

        return turn

    def clear(self) -> None:
        """Drop every stored turn (memory-only reset)."""

        self._turns.clear()

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        """Stored turns, oldest first, as an immutable tuple."""

        return tuple(self._turns)

    @property
    def last(self) -> ConversationTurn | None:
        """Most recent turn, or ``None`` when the state is empty."""

        return self._turns[-1] if self._turns else None

    @property
    def max_turns(self) -> int:
        """Configured bound on stored turns."""

        return self._max_turns

    def is_empty(self) -> bool:
        """True when no turns are stored."""

        return not self._turns

    def as_dicts(self) -> list[dict[str, Any]]:
        """All turns as plain dicts (oldest first)."""

        return [turn.to_dict() for turn in self._turns]

    def to_context(self) -> dict[str, Any]:
        """Emit the read-only context dict handed to ``answer_query``.

        Shape (stable contract for the engine's optional ``context``
        parameter and for future layers):

            {
                "turns": [ {"user_query", "assistant_response",
                            "language"}, ... ],   # oldest first
                "last_query": str | None,
                "last_response": str | None,
                "last_language": str | None,
            }

        An empty state emits the same keys with empty/None values, so a
        consumer can always read the same keys without key checks.
        """

        last = self.last
        return {
            "turns": self.as_dicts(),
            "last_query": last.user_query if last else None,
            "last_response": last.assistant_response if last else None,
            "last_language": last.language if last else None,
        }

    # ------------------------------------------------------------------
    # Conveniences
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._turns)

    def __iter__(self) -> Iterator[ConversationTurn]:
        return iter(self._turns)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ConversationState(turns={len(self._turns)}, "
            f"max_turns={self._max_turns})"
        )
