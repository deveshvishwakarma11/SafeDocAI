"""
Adaptive selection budget for the "selection instead of transcription" redesign.

This module is pure logic; it is wired into the pipeline via
selection_engine.py, which plans every window's calls with these budgets.

Approved budget logic (B=196/219)
---------------------------------
The approved rule fixes two response token budgets:

* **B_FIRST = 196 tokens** -- the budget for a window whose candidates fit in
  a single LLM call (the common, sparse case). num_predict=196.
* **B_EXTENDED = 219 tokens** -- the escalation budget used once a window has
  more candidates than the 196-budget K allows, i.e. continuation calls are
  needed. num_predict=219.

K (max selections per call) is derived from the active budget:

    K = clamp((B - ENVELOPE_TOKEN_RESERVATION) // SELECTION_TOKEN_COST,
              SELECTION_K_MIN, SELECTION_K_HARD_CAP)

With the constants below this yields K_FIRST = 5 (from 196) and
K_EXTENDED = 6 (from 219); both approved numbers are therefore operative,
not decorative.

Provenance note (kept visible on purpose): the approved plan text with the
full derivation was not recoverable from disk after the failed turn, so the
envelope reservation (120) and per-selection cost (15) below are the
best-evidence mapping of the recovered fragment ("envelope allowance
300 - 196 - 40" -> 40-token non-envelope slot -> envelope/data split of
120/40; measured record cost ~14-17 -> 15). If the approved text surfaces,
only these two constants should need adjusting; the two budgets and the
mechanics are implemented exactly as approved.

Guarantees preserved
--------------------
* Greedy first call (offered candidates = min(count, K)).
* Continuation splitting with disjoint candidate ids.
* No silent drops: every candidate is offered exactly once.
* No tiny tail calls: whenever a naive greedy split would end in a tail
  smaller than SELECTION_K_MIN, the slices are rebalanced (within K) so no
  call wastes a round trip on a near-empty batch.
* Truncation retry: a call that hits num_predict (done_reason == "length")
  is retried on the SAME slice with K halved, floored, limited attempts.

Nothing here talks to Ollama; callers pass done_reason in.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Approved budget constants (isolated; tests and callers share this source)
# --------------------------------------------------------------------------

#: Approved response token budget for a single-call window (num_predict).
SELECTION_RESPONSE_TOKEN_BUDGET = 196

#: Approved escalation response token budget once continuations are needed.
SELECTION_RESPONSE_TOKEN_BUDGET_EXTENDED = 219

#: Upper bound on selections a single LLM call may ever return.
SELECTION_K_HARD_CAP = 12

#: Smallest useful batch size. A continuation call below this spends more on
#: prompt tokens than it saves; the splitter also refuses tiny tails.
SELECTION_K_MIN = 3

#: Token reservation for the JSON envelope of one response (braces, keys the
#: model repeats, whitespace, occasional commentary). Derived from the
#: approved envelope allowance; see the provenance note in the docstring.
_ENVELOPE_TOKEN_RESERVATION = 120

#: Worst-case token cost of ONE selected field in the response
#: ({"id": 12, "k": "Roll No", "v": "2407510100067"} shape, measured ~14-17).
_SELECTION_TOKEN_COST = 15

#: When a truncation is detected (done_reason == "length"), retry the same
#: candidate slice with K halved, down to this floor.
_TRUNCATION_RETRY_K_FLOOR = 2

#: Maximum number of truncation retries per slice before giving up.
_TRUNCATION_RETRY_MAX_ATTEMPTS = 2


# --------------------------------------------------------------------------
# Core budget computation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionPlan:
    """Everything the caller needs to make one or more bounded LLM calls."""

    #: Number of candidates the window actually offers (after any caps applied
    #: upstream). The plan never exceeds this.
    candidate_count: int

    #: K for the FIRST call: how many selections it may return.
    first_k: int

    #: Whether candidate_count > first_k, i.e. at least one continuation call
    #: will be needed for the remaining candidates.
    needs_continuation: bool

    #: Total number of LLM calls this window will need (first + continuations).
    #: 1 if everything fits in the first call.
    total_calls: int

    #: Size of each call's offered candidate slice, in order.
    #: Slices are disjoint and cover all candidates (no silent drops).
    slice_sizes: tuple[int, ...] = ()

    #: Response token budget that must be passed as num_predict for each call
    #: (196 for single-call windows, 219 once continuations are involved).
    num_predict: int = SELECTION_RESPONSE_TOKEN_BUDGET


def max_k_for_budget(
    response_token_budget: int = SELECTION_RESPONSE_TOKEN_BUDGET,
) -> int:
    """Largest K that provably fits inside the given response token budget.

    K = clamp((budget - envelope) // per-selection cost, K_MIN, K_HARD_CAP).
    With the approved budgets this gives K=5 (B=196) and K=6 (B=219).
    """
    if response_token_budget <= _ENVELOPE_TOKEN_RESERVATION:
        return SELECTION_K_MIN
    theoretical = (
        (response_token_budget - _ENVELOPE_TOKEN_RESERVATION)
        // _SELECTION_TOKEN_COST
    )
    # Clamp here so this function's answer is always directly usable and
    # consistent with plan_selection_calls (which clamps again defensively).
    return min(max(theoretical, SELECTION_K_MIN), SELECTION_K_HARD_CAP)


def _balanced_slice_sizes(count: int, k: int) -> tuple[int, ...]:
    """Split ``count`` candidates into calls of at most ``k`` each.

    Greedy-first: the first call takes k whenever count > k. The remainder is
    split as evenly as possible, EXCEPT that a naive split ending in a tail
    smaller than SELECTION_K_MIN triggers a full rebalance over ceil(count/k)
    calls, so no call is a wasted near-empty round trip. All slice sizes stay
    within [.., k] and sum exactly to count.
    """
    if count <= 0:
        return ()
    if count <= k:
        return (count,)

    first = k
    remaining = count - first
    calls = -(-remaining // k)  # ceil
    base, extra = divmod(remaining, calls)
    # With extra > 0 the LAST remainder slice is `base`; refuse tails < K_MIN.
    if base < SELECTION_K_MIN:
        # Rebalance everything over ceil(count / k) calls.
        total_calls = -(-count // k)
        base, extra = divmod(count, total_calls)
        sizes = [base + 1] * extra + [base] * (total_calls - extra)
        return tuple(sizes)
    remainder = [base + 1] * extra + [base] * (calls - extra)
    return tuple([first] + remainder)


def plan_selection_calls(
    candidate_count: int,
    response_token_budget: int | None = None,
) -> SelectionPlan:
    """Plan how many bounded calls a window needs and how big each slice is.

    Approved rule:

    1. K_FIRST = max_k_for_budget(196). If candidate_count <= K_FIRST the
       window is planned as a single call under the 196 budget.
    2. Otherwise escalate to the approved 219 budget: K_EXTENDED =
       max_k_for_budget(219), and the candidates are split greedy-first with
       no tiny tails (see _balanced_slice_sizes).
    3. Every candidate is offered exactly once across disjoint slices.

    ``response_token_budget`` may be passed explicitly to force a specific
    budget (testing / future tuning); by default the approved 196/219
    escalation applies.
    """
    if candidate_count <= 0:
        return SelectionPlan(
            candidate_count=0,
            first_k=0,
            needs_continuation=False,
            total_calls=0,
            slice_sizes=(),
            num_predict=(
                response_token_budget
                if response_token_budget is not None
                else SELECTION_RESPONSE_TOKEN_BUDGET
            ),
        )

    if response_token_budget is None:
        k_first = max_k_for_budget(SELECTION_RESPONSE_TOKEN_BUDGET)
        if candidate_count <= k_first:
            budget = SELECTION_RESPONSE_TOKEN_BUDGET
            k = k_first
        else:
            budget = SELECTION_RESPONSE_TOKEN_BUDGET_EXTENDED
            k = max_k_for_budget(SELECTION_RESPONSE_TOKEN_BUDGET_EXTENDED)
    else:
        budget = response_token_budget
        k = max_k_for_budget(budget)

    k = min(max(k, SELECTION_K_MIN), SELECTION_K_HARD_CAP)
    slice_sizes = _balanced_slice_sizes(candidate_count, k)

    return SelectionPlan(
        candidate_count=candidate_count,
        first_k=slice_sizes[0],
        needs_continuation=len(slice_sizes) > 1,
        total_calls=len(slice_sizes),
        slice_sizes=slice_sizes,
        num_predict=budget,
    )


# --------------------------------------------------------------------------
# Truncation retry (safety net; the budget above should prevent this)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryDecision:
    """What to do after a call came back truncated (done_reason == 'length')."""

    #: Whether to retry the same candidate slice.
    should_retry: bool

    #: K to use for the retry (halved, floored).
    retry_k: int

    #: How many retries remain before giving up on this slice.
    attempts_left: int

    #: Human-readable reason, for logs.
    reason: str = ""


def truncation_retry_plan(
    attempted_k: int,
    attempts_left: int = _TRUNCATION_RETRY_MAX_ATTEMPTS,
) -> RetryDecision:
    """Plan the retry after a call hit num_predict (done_reason == 'length').

    Halve K and retry the SAME candidate slice under the same budget. The
    floor prevents an infinite loop: at the floor the envelope dominates and
    we give up rather than spin.
    """
    if attempts_left <= 0:
        return RetryDecision(
            should_retry=False,
            retry_k=0,
            attempts_left=0,
            reason="no attempts left",
        )

    if attempted_k <= _TRUNCATION_RETRY_K_FLOOR:
        return RetryDecision(
            should_retry=False,
            retry_k=attempted_k,
            attempts_left=0,
            reason=(
                f"K={attempted_k} already at floor {_TRUNCATION_RETRY_K_FLOOR}; "
                "halving cannot shrink output further"
            ),
        )

    new_k = max(_TRUNCATION_RETRY_K_FLOOR, attempted_k // 2)
    return RetryDecision(
        should_retry=True,
        retry_k=new_k,
        attempts_left=attempts_left - 1,
        reason=f"truncated at num_predict; retrying same slice with K={new_k}",
    )


# --------------------------------------------------------------------------
# Splitting an offered candidate set into per-call slices
# --------------------------------------------------------------------------


def split_candidate_ids(
    candidate_ids: list[str],
    slice_sizes: tuple[int, ...],
) -> list[list[str]]:
    """Split offered candidate ids into disjoint per-call slices.

    Slices are consecutive, disjoint, and their concatenation is exactly
    candidate_ids (no drops, no duplicates). The last slice may be smaller
    than its nominal size if candidate_ids ran out.
    """
    if not candidate_ids:
        return []
    slices: list[list[str]] = []
    idx = 0
    for size in slice_sizes:
        if idx >= len(candidate_ids):
            break
        take = min(size, len(candidate_ids) - idx)
        slices.append(candidate_ids[idx : idx + take])
        idx += take
    if idx < len(candidate_ids):
        # Defensive: if slice_sizes under-counted (should not happen when
        # produced by plan_selection_calls), put the remainder in a final
        # slice so nothing is silently dropped.
        slices.append(candidate_ids[idx:])
    return slices
