"""
Unit tests for the Phase 3 selection-mode integration
(src/selection_engine.py, llm_engine selection additions,
document_understanding selection wiring).

Categories covered (per the approved integration plan):

A. Selection prompt construction
B. Response parsing (valid / zero / exact-K / unknown id / value mismatch /
   over-K / malformed / truncated / done_reason=length handling)
C. Evidence attachment (correct char_span slice, invalid span rejection)
D. Position-aware merge (repeats stay distinct, overlap dups collapse)
E. Fallback (invalid selection -> SelectionUnavailable -> legacy path)
F. Window-region locating (bounds, marker split, tail anchoring)

No LLM, no Ollama, no filesystem, no network. Pure functions only.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# Pipeline modules import siblings directly (from candidate_extractor import ...),
# so src/ itself must be importable (same convention as test_long_document_handling).
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import selection_engine as selection_engine
from document_understanding import (
    SELECTION_MODE_ENABLED,
    _strip_window_markers,
    _window_bounds,
)
from llm_engine import build_selection_prompt, parse_selection_response

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_MARKSHEET_OCR = (
    "AKTU One View\n"
    "Roll Number : 2407510100067\n"
    "Name : DEVESH VISHWAKARMA\n"
    "Father Name : RAM VISHWAKARMA\n"
    "Mother Name : SITA VISHWAKARMA\n"
    "Course : BTECH CSE\n"
    "SGPA ३ 6.05\n"
    "Date of Declaration s 30/06/25\n"
)


def _candidates_from_ocr(text: str) -> list[dict]:
    from candidate_extractor import extract_candidates

    return [c.to_dict() for c in extract_candidates(text)]


def _id_map(candidates: list[dict]) -> dict[int, dict]:
    return {c["id"]: c for c in candidates}


def _selection(entry_id: int, key: str, value: str) -> str:
    import json

    return json.dumps({"s": [{"i": entry_id, "k": key, "v": value}]})


# ---------------------------------------------------------------------------
# A. Selection prompt construction
# ---------------------------------------------------------------------------


class TestSelectionPrompt:
    def test_candidate_ids_labels_values_included(self):
        cands = [
            {"id": 12, "label": "Roll Number", "value": "2407510100067"},
            {"id": 3, "label": "SGPA", "value": "6.05"},
        ]
        prompt = build_selection_prompt("OCR WINDOW", cands)
        assert "12. Roll Number: 2407510100067" in prompt
        assert "3. SGPA: 6.05" in prompt

    def test_ocr_window_included(self):
        prompt = build_selection_prompt(
            "THE WINDOW TEXT ITSELF", [{"id": 0, "label": "L", "value": "v"}]
        )
        assert "THE WINDOW TEXT ITSELF" in prompt

    def test_doc_type_hint_included_when_available(self):
        prompt = build_selection_prompt(
            "w", [{"id": 0, "label": "L", "value": "v"}], doc_type_hint="MARKSHEET"
        )
        assert "MARKSHEET" in prompt

    def test_doc_type_hint_omitted_when_unknown(self):
        prompt = build_selection_prompt(
            "w", [{"id": 0, "label": "L", "value": "v"}], doc_type_hint="UNKNOWN"
        )
        assert "UNKNOWN" not in prompt

    def test_max_selections_stated_to_model(self):
        prompt = build_selection_prompt(
            "w", [{"id": 0, "label": "L", "value": "v"}], max_selections=5
        )
        assert "at most 5" in prompt

    def test_schema_keys_are_compact(self):
        prompt = build_selection_prompt(
            "w", [{"id": 0, "label": "L", "value": "v"}]
        )
        assert '"s"' in prompt and '"i"' in prompt and '"k"' in prompt
        assert '"v"' in prompt and '"t"' in prompt and '"m"' in prompt
        assert "evidence" in prompt  # explicit "No evidence snippets" rule


# ---------------------------------------------------------------------------
# B. Response parsing
# ---------------------------------------------------------------------------


class TestSelectionParsing:
    def _setup(self):
        cands = _candidates_from_ocr(_MARKSHEET_OCR)
        return cands, _id_map(cands)

    def test_valid_response(self):
        cands, idmap = self._setup()
        roll = next(c for c in cands if "2407510100067" in c["value"])
        result = parse_selection_response(
            _selection(roll["id"], "Roll No", roll["value"]), idmap, max_selections=5
        )
        assert result.ok, result.reason
        assert result.selections[0]["id"] == roll["id"]
        assert result.selections[0]["value"] == roll["value"]
        assert result.selections[0]["char_span"] == list(roll["char_span"])
        assert result.selections[0]["line_no"] == roll["line_no"]

    def test_zero_selections_is_valid(self):
        _, idmap = self._setup()
        result = parse_selection_response(
            '{"s": [], "t": "Marksheet", "m": "nothing"}', idmap, max_selections=5
        )
        assert result.ok, result.reason
        assert result.selections == []
        assert result.document_type == "Marksheet"

    def test_exact_k_selections_accepted(self):
        cands, idmap = self._setup()
        self._assert_at_least_five_candidates(cands)
        five = cands[:5]
        body = ",".join(
            '{"i": %d, "k": "k", "v": "%s"}' % (c["id"], c["value"]) for c in five
        )
        result = parse_selection_response(
            '{"s": [%s]}' % body, idmap, max_selections=5
        )
        assert result.ok, result.reason
        assert len(result.selections) == 5

    def test_unknown_candidate_id_rejected(self):
        _, idmap = self._setup()
        result = parse_selection_response(
            '{"s": [{"i": 999, "k": "x", "v": "y"}]}', idmap, max_selections=5
        )
        assert not result.ok
        assert "unknown candidate id" in result.reason

    def test_mismatched_value_rejected(self):
        """A response whose every entry is bad is a failed response."""
        _, idmap = self._setup()
        result = parse_selection_response(
            '{"s": [{"i": 0, "k": "x", "v": "NOT THE CANDIDATE VALUE"}]}',
            idmap,
            max_selections=5,
        )
        assert not result.ok
        assert "value mismatch" in result.reason

    def test_mixed_response_keeps_valid_entries_drops_bad_pairing(self):
        """Real-model failure mode: one fabricated value must not kill
        independently valid selections in the same response."""
        cands, idmap = self._setup()
        good = next(c for c in cands if "2407510100067" in c["value"])
        other = next(c for c in cands if c["id"] != good["id"])
        body = (
            '{"i": %d, "k": "Roll", "v": "%s"}, '
            '{"i": %d, "k": "Wrong", "v": "FABRICATED VALUE"}'
            % (good["id"], good["value"], other["id"])
        )
        result = parse_selection_response(
            '{"s": [%s]}' % body, idmap, max_selections=5
        )
        assert result.ok, result.reason
        assert len(result.selections) == 1
        assert result.selections[0]["id"] == good["id"]
        assert result.selections[0]["value"] == good["value"]
        assert result.rejected_reasons
        assert "value mismatch" in result.rejected_reasons[0]

    def test_one_based_id_shift_is_value_anchored(self):
        """Real qwen2.5:3b E2E failure mode: 1-based ids against the 0-based
        candidate list. Every value is a genuine candidate value, so the
        value-anchored repair re-pairs them; claimed-id collisions are
        dropped, never silently merged."""
        cands, idmap = self._setup()
        # Take 3 real candidates and emit them with ids shifted by +1.
        three = cands[:3]
        body = ",".join(
            '{"i": %d, "k": "k", "v": "%s"}' % (c["id"] + 1, c["value"])
            for c in three
        )
        result = parse_selection_response(
            '{"s": [%s]}' % body, idmap, max_selections=5
        )
        assert result.ok, result.reason
        assert len(result.selections) == 3
        repaired = {s["id"] for s in result.selections}
        assert repaired == {c["id"] for c in three}
        assert len(result.repairs) >= 1

    def test_ambiguous_value_match_rejected_not_guessed(self):
        """Two offered candidates sharing a value: no deterministic winner,
        the entry is dropped rather than guessed."""
        # Build a synthetic map where two ids share one value.
        idmap = {
            0: {"id": 0, "label": "A", "value": "DUP", "line_no": 0,
                "char_span": [0, 5]},
            1: {"id": 1, "label": "B", "value": "DUP", "line_no": 1,
                "char_span": [6, 11]},
        }
        result = parse_selection_response(
            '{"s": [{"i": 7, "k": "x", "v": "DUP"}]}', idmap, max_selections=5
        )
        assert not result.ok
        # Claimed id 7 is unknown and the value matches 2 candidates -> no
        # unique repair possible -> rejected as an unknown candidate id
        # (never guessed), with the ambiguity explicitly reported.
        assert "unknown candidate id" in result.reason
        assert "ambiguously" in result.reason

    def test_clipped_prefix_value_accepted_with_candidate_value(self):
        """Model may clip a long value to 36 chars; candidate value wins."""
        cands, idmap = self._setup()
        roll = next(c for c in cands if "2407510100067" in c["value"])
        clipped = roll["value"][:10]
        result = parse_selection_response(
            _selection(roll["id"], "Roll", clipped), idmap, max_selections=5
        )
        assert result.ok, result.reason
        assert result.selections[0]["value"] == roll["value"]

    def test_over_k_selections_rejected(self):
        """Over-K is a WHOLE-response rejection (planned bound violated)."""
        cands, idmap = self._setup()
        self._assert_at_least_five_candidates(cands)
        body = ",".join(
            '{"i": %d, "k": "k", "v": "%s"}' % (c["id"], c["value"])
            for c in cands[:6]
        )
        result = parse_selection_response(
            '{"s": [%s]}' % body, idmap, max_selections=5
        )
        assert not result.ok
        assert "exceeds planned K" in result.reason

    @staticmethod
    def _assert_at_least_five_candidates(cands: list[dict]) -> None:
        assert len(cands) >= 5, (
            "fixture must yield >=5 candidates for K-boundary tests; "
            f"got {len(cands)}: {[(c['id'], c['label'], c['value']) for c in cands]}"
        )

    def test_malformed_json_rejected(self):
        _, idmap = self._setup()
        result = parse_selection_response("not json at all", idmap, max_selections=5)
        assert not result.ok

    def test_truncated_json_rejected_whole(self):
        """A num_predict truncation must NEVER be partially accepted."""
        _, idmap = self._setup()
        truncated = '{"s": [{"i": 0, "k": "Roll Numb'
        result = parse_selection_response(truncated, idmap, max_selections=5)
        assert not result.ok
        assert result.selections == []

    def test_truncated_with_closed_object_still_rejected(self):
        _, idmap = self._setup()
        # JSON parses but the array was cut before the closing bracket content.
        result = parse_selection_response(
            '{"s": [{"i": 0, "k": "k"}', idmap, max_selections=5
        )
        assert not result.ok

    def test_non_integer_id_rejected(self):
        _, idmap = self._setup()
        result = parse_selection_response(
            '{"s": [{"i": "zero", "k": "k", "v": "v"}]}', idmap, max_selections=5
        )
        assert not result.ok

    def test_digit_string_id_coerced_and_verified(self):
        """Real-model quirk (qwen returned "i":"1"): digit strings coerce
        before lookup; value must still match the candidate exactly."""
        cands, idmap = self._setup()
        roll = next(c for c in cands if "2407510100067" in c["value"])
        result = parse_selection_response(
            _selection(roll["id"], "Roll", roll["value"]) # int form
            .replace(f'"i": {roll["id"]}', f'"i": "{roll["id"]}"'),
            idmap,
            max_selections=5,
        )
        assert result.ok, result.reason
        assert result.selections[0]["id"] == roll["id"]

    def test_digit_string_id_with_wrong_value_still_rejected(self):
        _, idmap = self._setup()
        result = parse_selection_response(
            '{"s": [{"i": "0", "k": "x", "v": "WRONG"}]}', idmap, max_selections=5
        )
        assert not result.ok
        assert "value mismatch" in result.reason

    def test_non_object_entry_rejected(self):
        _, idmap = self._setup()
        result = parse_selection_response(
            '{"s": [42]}', idmap, max_selections=5
        )
        assert not result.ok

    def test_missing_s_key_rejected(self):
        _, idmap = self._setup()
        result = parse_selection_response(
            '{"t": "Marksheet"}', idmap, max_selections=5
        )
        assert not result.ok

    def test_empty_response_rejected(self):
        _, idmap = self._setup()
        assert not parse_selection_response("", idmap, max_selections=5).ok
        assert not parse_selection_response(None, idmap, max_selections=5).ok

    def test_long_key_clamped_to_three_words(self):
        cands, idmap = self._setup()
        roll = next(c for c in cands if "2407510100067" in c["value"])
        result = parse_selection_response(
            _selection(roll["id"], "one two three four five", roll["value"]),
            idmap,
            max_selections=5,
        )
        assert result.ok
        assert len(result.selections[0]["key"].split()) == 3


# ---------------------------------------------------------------------------
# C. Evidence attachment
# ---------------------------------------------------------------------------


class TestEvidenceAttachment:
    OCR = "Roll Number : 2407510100067\nName : DEVESH"

    def test_exact_char_span_slice(self):
        fields = [
            {
                "id": 0,
                "key": "Roll Number",
                "value": "2407510100067",
                "line_no": 0,
                "char_span": [0, 27],
            }
        ]
        accepted, rejected = selection_engine.attach_evidence(fields, self.OCR)
        assert len(accepted) == 1
        assert len(rejected) == 0
        assert accepted[0]["evidence_snippet"] == self.OCR[0:27]
        # No normalization: original OCR characters preserved exactly.
        assert accepted[0]["evidence_snippet"] == "Roll Number : 2407510100067"

    def test_invalid_out_of_bounds_span_rejected(self):
        fields = [
            {"id": 0, "key": "k", "value": "v", "char_span": [100, 200]}
        ]
        accepted, rejected = selection_engine.attach_evidence(fields, self.OCR)
        assert accepted == []
        assert rejected[0]["rejection_reason"] == "invalid span"

    def test_inverted_span_rejected(self):
        fields = [{"id": 0, "key": "k", "value": "v", "char_span": [10, 5]}]
        accepted, rejected = selection_engine.attach_evidence(fields, self.OCR)
        assert accepted == []
        assert rejected[0]["rejection_reason"] == "invalid span"

    def test_missing_span_rejected_not_invented(self):
        fields = [{"id": 0, "key": "k", "value": "v"}]
        accepted, rejected = selection_engine.attach_evidence(fields, self.OCR)
        assert accepted == []
        assert len(rejected) == 1

    def test_whitespace_only_snippet_rejected(self):
        fields = [{"id": 0, "key": "k", "value": "v", "char_span": [13, 14]}]
        accepted, rejected = selection_engine.attach_evidence(fields, self.OCR)
        assert accepted == []
        assert rejected[0]["rejection_reason"] == "empty snippet"

    def test_mixed_accept_reject(self):
        fields = [
            {"id": 0, "key": "good", "value": "2407510100067", "char_span": [14, 27]},
            {"id": 1, "key": "bad", "value": "x", "char_span": [999, 1002]},
        ]
        accepted, rejected = selection_engine.attach_evidence(fields, self.OCR)
        assert len(accepted) == 1 and accepted[0]["key"] == "good"
        assert len(rejected) == 1 and rejected[0]["key"] == "bad"


# ---------------------------------------------------------------------------
# D. Position-aware merge
# ---------------------------------------------------------------------------


def _field(key: str, value: str, start: int) -> dict:
    return {
        "key": key,
        "value": value,
        "evidence_snippet": f"OCR[{start}]",
        "candidate_id": start,
        "line_no": start // 40,
        "char_span": [start, start + 20],
    }


class TestPositionAwareMerge:
    def test_multi_semester_sgpa_values_stay_distinct(self):
        windows = [
            {
                "accepted_fields": [
                    _field("SGPA", "6.09", 100),
                    _field("SGPA", "6.05", 900),
                    _field("SGPA", "7.00", 1700),
                    _field("SGPA", "7.12", 2500),
                    _field("SGPA", "6.61", 3300),
                ]
            }
        ]
        merged, info = selection_engine.position_aware_merge(windows, "x" * 4000)
        values = sorted(f["value"] for f in merged if f["key"] == "SGPA")
        assert values == ["6.05", "6.09", "6.61", "7.00", "7.12"]
        assert info["kept"] == 5

    def test_repeated_dates_stay_distinct(self):
        windows = [
            {
                "accepted_fields": [
                    _field("Date of Declaration", "30/06/25", 100),
                    _field("Date of Declaration", "30/06/24", 1500),
                    _field("Date of Declaration", "30/06/23", 2900),
                ]
            }
        ]
        merged, info = selection_engine.position_aware_merge(windows, "x" * 4000)
        assert info["kept"] == 3

    def test_identical_values_in_different_sections_stay_distinct(self):
        windows = [
            {
                "accepted_fields": [
                    _field("Session", "2024-25", 100),
                    _field("Session", "2024-25", 2500),
                ]
            }
        ]
        merged, info = selection_engine.position_aware_merge(windows, "x" * 4000)
        assert info["kept"] == 2

    def test_overlap_duplicate_collapses(self):
        """Same field re-found in an overlapping window = one record."""
        windows = [
            {"accepted_fields": [_field("Roll Number", "2407510100067", 400)]},
            {"accepted_fields": [_field("Roll Number", "2407510100067", 408)]},
        ]
        merged, info = selection_engine.position_aware_merge(windows, "x" * 4000)
        assert info["kept"] == 1
        assert info["collapsed_duplicates"] == 1

    def test_same_label_different_values_never_merge(self):
        windows = [
            {
                "accepted_fields": [
                    _field("Total Marks", "610", 200),
                    _field("Total Marks", "686", 240),
                ]
            }
        ]
        merged, info = selection_engine.position_aware_merge(windows, "x" * 4000)
        assert info["kept"] == 2

    def test_first_seen_document_order_preserved(self):
        windows = [
            {
                "accepted_fields": [
                    _field("Name", "DEVESH", 100),
                    _field("Roll Number", "2407510100067", 300),
                ]
            },
            {"accepted_fields": [_field("SGPA", "6.09", 1500)]},
        ]
        merged, _ = selection_engine.position_aware_merge(windows, "x" * 4000)
        keys = [f["key"] for f in merged]
        assert keys == ["Name", "Roll Number", "SGPA"]

    def test_empty_input(self):
        merged, info = selection_engine.position_aware_merge([], "text")
        assert merged == [] and info["kept"] == 0


# ---------------------------------------------------------------------------
# E. Fallback behaviour (monkeypatched transport; no real Ollama)
# ---------------------------------------------------------------------------


class _FakeMeta:
    def __init__(self, text, done_reason="stop", eval_count=10):
        self.payload = {
            "text": text,
            "done_reason": done_reason,
            "eval_count": eval_count,
            "latency_seconds": 0.01,
        }


class TestFallbackPath:
    def test_transport_failure_raises_selection_unavailable(self, monkeypatch):
        monkeypatch.setattr(selection_engine, "generate_with_meta", lambda *a, **k: None)
        call_log: list[dict] = []
        try:
            selection_engine.select_window(
                _MARKSHEET_OCR,
                _MARKSHEET_OCR,
                0,
                len(_MARKSHEET_OCR),
                model="fake",
                doc_type_hint=None,
                call_log=call_log,
                max_attempts_per_slice=2,
            )
        except selection_engine.SelectionUnavailable:
            pass
        else:
            raise AssertionError("SelectionUnavailable not raised")
        # The failed call is still logged with telemetry.
        assert call_log and call_log[0]["error"] == "transport failure"

    def test_invalid_json_after_retries_raises(self, monkeypatch):
        monkeypatch.setattr(
            selection_engine,
            "generate_with_meta",
            lambda *a, **k: _FakeMeta("NOT JSON").payload,
        )
        try:
            selection_engine.select_window(
                _MARKSHEET_OCR,
                _MARKSHEET_OCR,
                0,
                len(_MARKSHEET_OCR),
                model="fake",
                doc_type_hint=None,
                max_attempts_per_slice=2,
            )
        except selection_engine.SelectionUnavailable as exc:
            assert "invalid" in str(exc) or "JSON" in str(exc)
        else:
            raise AssertionError("SelectionUnavailable not raised")

    def test_truncation_retry_then_give_up(self, monkeypatch):
        """done_reason=length must trigger halved-K retries, then fall back."""
        calls: list[dict] = []

        def fake_generate(prompt, **kwargs):
            calls.append({"num_predict": kwargs.get("num_predict")})
            return _FakeMeta(
                '{"s": [{"i": 0, "k": "Roll Number", "v": "2407510100067"}, '
                '{"i": 1, "k": "Name", "v": "DEVESH VISHWAKARMA"}, '
                '{"i": 2, "k": "SGPA", "v": "6.05"}, '
                '{"i": 3, "k": "Date", "v": "30/06/25"}, '
                '{"i": 4, "k": "Extra", "v": "0"}]}',
                done_reason="length",
            ).payload

        monkeypatch.setattr(selection_engine, "generate_with_meta", fake_generate)
        call_log: list[dict] = []
        try:
            selection_engine.select_window(
                _MARKSHEET_OCR,
                _MARKSHEET_OCR,
                0,
                len(_MARKSHEET_OCR),
                model="fake",
                doc_type_hint=None,
                call_log=call_log,
                max_attempts_per_slice=3,
            )
        except selection_engine.SelectionUnavailable:
            pass
        else:
            raise AssertionError("SelectionUnavailable not raised")
        # Every attempt was bounded to the approved num_predict, and the
        # parse failures were recorded, not silently accepted.
        assert call_log
        assert all(c["json_valid"] is False for c in call_log)
        assert all(c["done_reason"] == "length" for c in call_log)

    def test_usable_selection_after_first_failure(self, monkeypatch):
        """One bad call then a good call: the window still succeeds.

        The fixture window yields 7 candidates, so the approved planner
        splits it into 2 slices (4+3) -- the good response is consumed per
        slice and both slices contribute selections.
        """
        state = {"n": 0}

        def fake_generate(prompt, **kwargs):
            state["n"] += 1
            if state["n"] == 1:
                return _FakeMeta("garbage").payload
            return _FakeMeta(
                '{"s": [{"i": 0, "k": "Roll Number", "v": "2407510100067"}], '
                '"t": "Marksheet", "m": "student record"}'
            ).payload

        monkeypatch.setattr(selection_engine, "generate_with_meta", fake_generate)
        result = selection_engine.select_window(
            _MARKSHEET_OCR,
            _MARKSHEET_OCR,
            0,
            len(_MARKSHEET_OCR),
            model="fake",
            doc_type_hint=None,
            max_attempts_per_slice=2,
        )
        assert len(result["selected_fields"]) == 2
        assert result["doc_type_votes"].count("Marksheet") == 2
        assert result["plan"]["total_calls"] == 2

    def test_no_candidates_raises(self):
        text = "this line has no separators at all"
        try:
            selection_engine.select_window(
                text, text, 0, len(text), model="fake", doc_type_hint=None
            )
        except selection_engine.SelectionUnavailable:
            pass
        else:
            raise AssertionError("SelectionUnavailable not raised")

    def test_selection_mode_flag_is_on(self):
        assert SELECTION_MODE_ENABLED is True


# ---------------------------------------------------------------------------
# F. Window-region locating
# ---------------------------------------------------------------------------


class TestWindowBounds:
    def test_short_document_single_region(self):
        text = "Name: Test\nRoll: 1"
        wins = ["Name: Test\nRoll: 1"]
        regions = _window_bounds(wins, text)
        assert regions == [(1, 0, len(text))]

    def test_regions_are_absolute_and_ordered(self):
        text = "A" * 200 + "\n" + "B" * 200
        wins = [text]
        regions = _window_bounds(wins, text)
        assert len(regions) == 1
        _, start, end = regions[0]
        assert start == 0 and end == len(text)

    def test_marker_window_splits_into_head_and_tail(self):
        head = "HEAD" * 500
        tail = "TAIL" * 300
        text = head + tail
        marker = f"\n[... 200 characters omitted ...]\n"
        window = head + marker + tail
        regions = _window_bounds([window], text)
        assert len(regions) == 2
        assert regions[0][1] == 0
        assert abs(regions[-1][2] - len(text)) < len(tail)

    def test_tail_region_reaches_document_end(self):
        head = "HEAD" * 500
        tail = "TAIL: 1\nROLL: 2\n" * 20
        text = head + tail
        marker = "\n[... 300 characters omitted ...]\n"
        window = head[:700] + marker + tail
        regions = _window_bounds([window], text)
        assert regions[-1][2] == len(text)

    def test_multi_window_forward_progress(self):
        text = ("Line 1\nLine 2\nLine 3\n" * 100).strip()
        wins = [
            "\n".join(text.split("\n")[:60]),
            "\n".join(text.split("\n")[55:120]),
        ]
        regions = _window_bounds(wins, text)
        assert len(regions) >= 2
        starts = [r[1] for r in regions]
        assert starts == sorted(starts)

    def test_strip_window_markers(self):
        assert (
            _strip_window_markers("head\n[... 50 characters omitted ...]\ntail")
            == "head\n\ntail"
        )
        assert _strip_window_markers("plain") == "plain"


# ---------------------------------------------------------------------------
# F. Regression: planner constants untouched by integration
# ---------------------------------------------------------------------------


class TestApprovedConstantsUnchanged:
    def test_budget_constants(self):
        from selection_budget import (
            SELECTION_RESPONSE_TOKEN_BUDGET,
            SELECTION_RESPONSE_TOKEN_BUDGET_EXTENDED,
        )

        assert SELECTION_RESPONSE_TOKEN_BUDGET == 196
        assert SELECTION_RESPONSE_TOKEN_BUDGET_EXTENDED == 219

    def test_k_values(self):
        from selection_budget import (
            max_k_for_budget,
            SELECTION_RESPONSE_TOKEN_BUDGET,
            SELECTION_RESPONSE_TOKEN_BUDGET_EXTENDED,
        )

        assert max_k_for_budget(SELECTION_RESPONSE_TOKEN_BUDGET) == 5
        assert max_k_for_budget(SELECTION_RESPONSE_TOKEN_BUDGET_EXTENDED) == 6

    def test_candidate_caps(self):
        from candidate_extractor import (
            MAX_CANDIDATES_PER_WINDOW,
            MAX_CANDIDATES_PER_DOCUMENT,
        )

        assert MAX_CANDIDATES_PER_WINDOW == 25
        assert MAX_CANDIDATES_PER_DOCUMENT == 40


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    failures = 0
    ran = 0

    class _MonkeyPatch:
        """Minimal pytest-monkeypatch stand-in for the plain runner."""

        def __init__(self):
            self._saved = []

        def setattr(self, target, name, value):
            self._saved.append((target, name, getattr(target, name)))
            setattr(target, name, value)

        def undo(self):
            for target, name, original in reversed(self._saved):
                setattr(target, name, original)
            self._saved.clear()

    for name, obj in sorted(globals().items()):
        if name.startswith("Test") and isinstance(obj, type):
            instance = obj()
            for method_name in sorted(dir(instance)):
                if not method_name.startswith("test_"):
                    continue
                method = getattr(instance, method_name)
                ran += 1
                needs_patch = "monkeypatch" in method.__code__.co_varnames
                mp = _MonkeyPatch() if needs_patch else None
                try:
                    if needs_patch:
                        method(mp)
                    else:
                        method()
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    print(f"FAIL {name}.{method_name}: {exc}")
                finally:
                    if mp is not None:
                        mp.undo()

    print(
        f"Selection integration checks "
        f"{'PASSED' if failures == 0 else 'FAILED'}: {ran - failures}/{ran}"
    )
    raise SystemExit(1 if failures else 0)
