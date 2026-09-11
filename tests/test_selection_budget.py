"""
Unit tests for the approved adaptive selection budget (src/selection_budget.py).

The approved budget logic under test:

  - B_FIRST = 196 tokens for single-call windows (num_predict=196).
  - B_EXTENDED = 219 tokens once continuations are needed (num_predict=219).
  - K = clamp((B - envelope) // per-selection cost, 3, 12)
        -> K_FIRST = 5 (from 196), K_EXTENDED = 6 (from 219).
  - Greedy first call, continuation splitting with disjoint ids,
    no silent drops, no tiny tail calls (rebalanced within K),
    truncation retry (halve K, floor 2, max 2 attempts).

Synthetic dense and sparse candidate lists only; nothing here touches the
LLM, Ollama, or the pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure we import the project copy, not any installed shadow.
_project_root = Path(__file__).resolve().parents[1]
_sys_path_inserted = False
for _p in sys.path:
    if str(_project_root) == _p:
        _sys_path_inserted = True
        break
if not _sys_path_inserted:
    sys.path.insert(0, str(_project_root))

from src.selection_budget import (
    SELECTION_K_HARD_CAP,
    SELECTION_K_MIN,
    SELECTION_RESPONSE_TOKEN_BUDGET,
    SELECTION_RESPONSE_TOKEN_BUDGET_EXTENDED,
    max_k_for_budget,
    plan_selection_calls,
    split_candidate_ids,
    truncation_retry_plan,
    RetryDecision,
    SelectionPlan,
)


# ---------------------------------------------------------------------------
# Approved constants are the ones on disk
# ---------------------------------------------------------------------------


def test_approved_budgets_are_196_and_219() -> None:
    assert SELECTION_RESPONSE_TOKEN_BUDGET == 196
    assert SELECTION_RESPONSE_TOKEN_BUDGET_EXTENDED == 219


def test_approved_k_values() -> None:
    # K = clamp((196 - 120) // 15, 3, 12) = clamp(5, 3, 12) = 5
    assert max_k_for_budget(196) == 5
    # K = clamp((219 - 120) // 15, 3, 12) = clamp(6, 3, 12) = 6
    assert max_k_for_budget(219) == 6


def test_k_boundary_around_approved_budgets() -> None:
    # Just below/above the approved budgets: boundaries behave monotonically.
    assert max_k_for_budget(195) == 5   # (195-120)//15 = 5
    assert max_k_for_budget(134) == 3   # (134-120)//15 = 0 -> clamp to min
    assert max_k_for_budget(120) == 3   # at envelope reservation -> min
    assert max_k_for_budget(220) == 6   # (220-120)//15 = 6
    assert max_k_for_budget(225) == 7   # (225-120)//15 = 7
    assert max_k_for_budget(10_000) == SELECTION_K_HARD_CAP


def test_k_monotonic_in_budget() -> None:
    budgets = [130, 150, 180, 196, 200, 219, 240, 300, 500]
    ks = [max_k_for_budget(b) for b in budgets]
    assert ks == sorted(ks)


# ---------------------------------------------------------------------------
# plan_selection_calls: sparse documents (single call at the 196 budget)
# ---------------------------------------------------------------------------


def test_sparse_3_candidates_single_call_at_196() -> None:
    plan = plan_selection_calls(candidate_count=3)
    assert plan.slice_sizes == (3,)
    assert plan.first_k == 3
    assert plan.total_calls == 1
    assert plan.needs_continuation is False
    assert plan.num_predict == 196


def test_5_candidates_exactly_k_first_single_call_at_196() -> None:
    plan = plan_selection_calls(candidate_count=5)
    assert plan.slice_sizes == (5,)
    assert plan.num_predict == 196


def test_2_candidates_single_small_call_is_allowed() -> None:
    # Sparse doc: one call, no continuation to balance against.
    plan = plan_selection_calls(candidate_count=2)
    assert plan.slice_sizes == (2,)
    assert plan.num_predict == 196


def test_empty_candidate_list_needs_no_calls() -> None:
    plan = plan_selection_calls(candidate_count=0)
    assert plan.total_calls == 0
    assert plan.slice_sizes == ()
    assert plan.first_k == 0
    assert plan.num_predict == 196


# ---------------------------------------------------------------------------
# plan_selection_calls: escalation to the 219 budget
# ---------------------------------------------------------------------------


def test_6_candidates_escalate_to_219_single_call() -> None:
    # 6 > K_FIRST=5 -> escalation budget, but still a single call at K_EXT=6.
    plan = plan_selection_calls(candidate_count=6)
    assert plan.slice_sizes == (6,)
    assert plan.total_calls == 1
    assert plan.needs_continuation is False
    assert plan.num_predict == 219


def test_8_candidates_no_tiny_tail() -> None:
    # Naive greedy would give (6, 2) with a useless 2-candidate tail call.
    # The approved rule rebalances to two healthy calls.
    plan = plan_selection_calls(candidate_count=8)
    assert plan.slice_sizes == (4, 4)
    assert plan.total_calls == 2
    assert plan.needs_continuation is True
    assert plan.num_predict == 219


def test_15_candidates_split() -> None:
    plan = plan_selection_calls(candidate_count=15)
    assert plan.slice_sizes == (6, 5, 4)
    assert plan.total_calls == 3
    assert plan.num_predict == 219


def test_25_candidates_minimum_calls_no_tiny_tail() -> None:
    # K_EXT=6 -> ceil(25/6)=5 calls is the provable minimum; slices must be
    # balanced (no tail below SELECTION_K_MIN).
    plan = plan_selection_calls(candidate_count=25)
    assert plan.slice_sizes == (6, 5, 5, 5, 4)
    assert plan.total_calls == 5
    assert sum(plan.slice_sizes) == 25
    assert min(plan.slice_sizes) >= SELECTION_K_MIN
    assert plan.num_predict == 219


def test_7_candidates_rebalanced_from_tiny_tail() -> None:
    # Naive greedy: (6, 1). Rebalanced: (4, 3).
    plan = plan_selection_calls(candidate_count=7)
    assert plan.slice_sizes == (4, 3)


def test_13_candidates_split() -> None:
    plan = plan_selection_calls(candidate_count=13)
    assert plan.slice_sizes == (6, 4, 3)


# ---------------------------------------------------------------------------
# Structural invariants across a sweep of candidate counts
# ---------------------------------------------------------------------------


def test_sweep_invariants_default_budgets() -> None:
    for count in range(1, 61):
        plan = plan_selection_calls(candidate_count=count)
        assert sum(plan.slice_sizes) == count, f"count={count}"
        assert plan.total_calls == len(plan.slice_sizes)
        assert plan.first_k == plan.slice_sizes[0]
        assert plan.needs_continuation == (plan.total_calls > 1)
        assert all(s >= 1 for s in plan.slice_sizes)
        # No slice exceeds the K in force for its budget.
        k = 5 if plan.num_predict == 196 else 6
        assert all(s <= k for s in plan.slice_sizes), f"count={count}"
        # Continuation windows never end in a tiny tail (single-call sparse
        # windows of 1-2 candidates are allowed by design).
        if plan.total_calls > 1:
            assert min(plan.slice_sizes) >= SELECTION_K_MIN, f"count={count}"


def test_escalation_budget_matches_candidate_count() -> None:
    # Escalation is decided by candidate_count > K_FIRST=5, before slicing.
    # A 6-candidate window keeps the 219 budget even though it fits in one
    # call -- 6 selections x 15 + 120 = 210 > 196, so escalation is required.
    for count in range(1, 61):
        plan = plan_selection_calls(candidate_count=count)
        if count <= 5:
            assert plan.num_predict == 196, f"count={count}"
        else:
            assert plan.num_predict == 219, f"count={count}"


def test_explicit_budget_override_still_covers_all() -> None:
    plan = plan_selection_calls(candidate_count=10, response_token_budget=150)
    assert sum(plan.slice_sizes) == 10
    assert all(s <= max_k_for_budget(150) for s in plan.slice_sizes)
    assert plan.num_predict == 150


def test_plan_is_immutable() -> None:
    plan = plan_selection_calls(candidate_count=25)
    assert isinstance(plan, SelectionPlan)
    try:
        plan.first_k = 99  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("SelectionPlan must be frozen")


# ---------------------------------------------------------------------------
# Output bound: every planned call provably fits its num_predict
# ---------------------------------------------------------------------------


def test_output_bound_guarantee_approved_budgets() -> None:
    envelope = 120
    cost = 15
    for count in (0, 1, 3, 5, 6, 7, 8, 15, 25, 40, 60):
        plan = plan_selection_calls(candidate_count=count)
        budget = plan.num_predict
        k = max_k_for_budget(budget)
        for size in plan.slice_sizes:
            assert size <= k
            assert size * cost + envelope <= budget, (
                f"count={count}: slice {size} can exceed budget {budget}"
            )


# ---------------------------------------------------------------------------
# split_candidate_ids: disjoint, complete coverage
# ---------------------------------------------------------------------------


def _ids(n: int) -> list[str]:
    return [f"c{i:02d}" for i in range(n)]


def test_split_produces_disjoint_slices() -> None:
    plan = plan_selection_calls(candidate_count=25)
    slices = split_candidate_ids(_ids(25), plan.slice_sizes)
    flattened = [i for s in slices for i in s]
    assert len(flattened) == len(set(flattened)), "slices must be disjoint"


def test_split_covers_all_candidates_no_silent_drop() -> None:
    plan = plan_selection_calls(candidate_count=25)
    slices = split_candidate_ids(_ids(25), plan.slice_sizes)
    flattened = sorted(i for s in slices for i in s)
    assert flattened == sorted(_ids(25))


def test_split_matches_planned_slice_sizes() -> None:
    plan = plan_selection_calls(candidate_count=15)
    slices = split_candidate_ids(_ids(15), plan.slice_sizes)
    assert [len(s) for s in slices] == list(plan.slice_sizes)


def test_split_handles_sparse_list() -> None:
    plan = plan_selection_calls(candidate_count=3)
    slices = split_candidate_ids(_ids(3), plan.slice_sizes)
    assert slices == [["c00", "c01", "c02"]]


def test_split_empty_ids() -> None:
    plan = plan_selection_calls(candidate_count=0)
    assert split_candidate_ids([], plan.slice_sizes) == []


def test_split_defensive_remainder_not_dropped() -> None:
    # If slice_sizes under-counts (defensive path), the remainder must still
    # be returned rather than silently discarded.
    slices = split_candidate_ids(_ids(10), (4, 4))
    assert sum(len(s) for s in slices) == 10


def test_split_no_drop_across_sweep() -> None:
    for count in range(1, 61):
        plan = plan_selection_calls(candidate_count=count)
        slices = split_candidate_ids(_ids(count), plan.slice_sizes)
        flattened = [i for s in slices for i in s]
        assert len(flattened) == len(set(flattened)), f"count={count}"
        assert sorted(flattened) == sorted(_ids(count)), f"count={count}"


# ---------------------------------------------------------------------------
# truncation_retry_plan: halve, floor, give up
# ---------------------------------------------------------------------------


def test_truncation_retries_with_halved_k() -> None:
    d = truncation_retry_plan(attempted_k=6, attempts_left=2)
    assert isinstance(d, RetryDecision)
    assert d.should_retry is True
    assert d.retry_k == 3
    assert d.attempts_left == 1


def test_truncation_retry_floors_at_minimum() -> None:
    d = truncation_retry_plan(attempted_k=5, attempts_left=2)
    assert d.should_retry is True
    assert d.retry_k == 2
    assert d.attempts_left == 1


def test_truncation_retry_stops_at_floor_k() -> None:
    d = truncation_retry_plan(attempted_k=2, attempts_left=2)
    assert d.should_retry is False
    assert d.attempts_left == 0
    assert "floor" in d.reason.lower()


def test_truncation_retry_gives_up_when_no_attempts_left() -> None:
    d = truncation_retry_plan(attempted_k=6, attempts_left=0)
    assert d.should_retry is False
    assert d.attempts_left == 0


def test_truncation_retry_chain_terminates_from_k_extended() -> None:
    # Worst case: an escalated call (K=6) truncates repeatedly. K must
    # strictly decrease and the chain must terminate.
    k = 6
    attempts = 2
    seen: list[int] = [k]
    while True:
        d = truncation_retry_plan(attempted_k=k, attempts_left=attempts)
        if not d.should_retry:
            break
        assert d.retry_k < k, "retry K must strictly decrease"
        k = d.retry_k
        attempts = d.attempts_left
        seen.append(k)
    assert seen[-1] <= 2
    assert len(seen) <= 4


# ---------------------------------------------------------------------------
# Exact plan shapes for the requested report (3, 8, 15, 25)
# ---------------------------------------------------------------------------


def test_report_shapes_3_8_15_25() -> None:
    assert plan_selection_calls(3).slice_sizes == (3,)
    assert plan_selection_calls(8).slice_sizes == (4, 4)
    assert plan_selection_calls(15).slice_sizes == (6, 5, 4)
    assert plan_selection_calls(25).slice_sizes == (6, 5, 5, 5, 4)


def run_all() -> None:
    tests = [
        test_approved_budgets_are_196_and_219,
        test_approved_k_values,
        test_k_boundary_around_approved_budgets,
        test_k_monotonic_in_budget,
        test_sparse_3_candidates_single_call_at_196,
        test_5_candidates_exactly_k_first_single_call_at_196,
        test_2_candidates_single_small_call_is_allowed,
        test_empty_candidate_list_needs_no_calls,
        test_6_candidates_escalate_to_219_single_call,
        test_8_candidates_no_tiny_tail,
        test_15_candidates_split,
        test_25_candidates_minimum_calls_no_tiny_tail,
        test_7_candidates_rebalanced_from_tiny_tail,
        test_13_candidates_split,
        test_sweep_invariants_default_budgets,
        test_escalation_budget_matches_candidate_count,
        test_explicit_budget_override_still_covers_all,
        test_plan_is_immutable,
        test_output_bound_guarantee_approved_budgets,
        test_split_produces_disjoint_slices,
        test_split_covers_all_candidates_no_silent_drop,
        test_split_matches_planned_slice_sizes,
        test_split_handles_sparse_list,
        test_split_empty_ids,
        test_split_defensive_remainder_not_dropped,
        test_split_no_drop_across_sweep,
        test_truncation_retries_with_halved_k,
        test_truncation_retry_floors_at_minimum,
        test_truncation_retry_stops_at_floor_k,
        test_truncation_retry_gives_up_when_no_attempts_left,
        test_truncation_retry_chain_terminates_from_k_extended,
        test_report_shapes_3_8_15_25,
    ]

    failed: list[str] = []

    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed.append(f"{test.__name__}: {exc}")
        except Exception as exc:
            failed.append(f"{test.__name__}: RAISED {type(exc).__name__}: {exc}")

    if failed:
        print("Selection budget checks FAILED:")
        for line in failed:
            print(" -", line)
        raise SystemExit(1)

    print(f"Selection budget checks PASSED: {len(tests)}")


if __name__ == "__main__":
    run_all()
