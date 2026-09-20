"""Code-hygiene regression guard: no accidental CJK text in the codebase.

Background: an IME/clipboard glitch once introduced a Japanese katakana
character into a Hindi regex in ``src/heuristics.py`` (``रोल ヶバー`` style
corruption), silently killing the Hindi marksheet classification path.  This
suite makes that class of mistake fail CI instead of shipping.

Scope rules (deliberately asymmetric):

* **Production code** (``src/*.py`` + ``run_ui.py``): CJK ideographs, kana and
  hangul are forbidden outright, and fullwidth forms are forbidden except the
  full-width colon ``U+FF1A`` which is an *intentional* document separator in
  ``candidate_extractor`` (Chinese/Japanese-labelled documents use
  ``Student Name：...``).
* **Test code**: only kana and hangul are forbidden.  Chinese ideographs are
  allowed in fixtures because multilingual document parsing is deliberately
  exercised with Chinese-labelled documents (e.g. ``学号：2407510100067``).

This test never inspects ``data/`` — stored PDFs/JSONs are user documents and
may legitimately contain any script.
"""

import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CJK ranges: kana, CJK ideographs + extensions, compatibility ideographs,
# hangul syllables/jamo/compatibility jamo.
_KANA_HANGUL = re.compile(
    "[\u3040-\u30ff\u31f0-\u31ff\u1100-\u11ff\u3130-\u318f"
    "\uac00-\ud7af\ud7b0-\ud7ff]"
)
_IDEOGRAPH = re.compile(
    "[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)
# Fullwidth forms block: forbid everything except the full-width colon,
# which candidate_extractor intentionally treats as a label separator.
_FULLWIDTH_ALLOWED = {"\uff1a"}
_FULLWIDTH = re.compile("[\uff01-\uff60\uffe0-\uffe6]")

PRODUCTION_FILES = sorted(
    list((PROJECT_ROOT / "src").glob("*.py")) + [PROJECT_ROOT / "run_ui.py"]
)
TEST_FILES = sorted((PROJECT_ROOT / "tests").glob("*.py"))


def _hits(pattern, path):
    out = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        for m in pattern.finditer(line):
            out.append((lineno, m.group(), line.strip()))
    return out


class TestNoAccidentalCJK(unittest.TestCase):
    def test_production_files_exist(self):
        self.assertGreater(len(PRODUCTION_FILES), 5)
        self.assertIn(PROJECT_ROOT / "run_ui.py", PRODUCTION_FILES)

    def test_no_kana_or_hangul_in_production(self):
        problems = []
        for path in PRODUCTION_FILES:
            for lineno, ch, ctx in _hits(_KANA_HANGUL, path):
                problems.append(f"{path.name}:{lineno}: U+{ord(ch):04X} {ch!r} in {ctx}")
        self.assertEqual(problems, [], "kana/hangul leaked into production code:\n" + "\n".join(problems))

    def test_no_cjk_ideographs_in_production(self):
        problems = []
        for path in PRODUCTION_FILES:
            for lineno, ch, ctx in _hits(_IDEOGRAPH, path):
                problems.append(f"{path.name}:{lineno}: U+{ord(ch):04X} {ch!r} in {ctx}")
        self.assertEqual(problems, [], "Chinese ideographs leaked into production code:\n" + "\n".join(problems))

    def test_only_approved_fullwidth_chars_in_production(self):
        problems = []
        for path in PRODUCTION_FILES:
            for lineno, ch, ctx in _hits(_FULLWIDTH, path):
                if ch in _FULLWIDTH_ALLOWED:
                    continue
                problems.append(f"{path.name}:{lineno}: U+{ord(ch):04X} {ch!r} in {ctx}")
        self.assertEqual(problems, [], "unexpected fullwidth chars in production code:\n" + "\n".join(problems))

    def test_intentional_fullwidth_colon_still_supported(self):
        # The multilingual separator support must not be accidentally deleted
        # by future "cleanup" of the only allowed fullwidth character.
        src = (PROJECT_ROOT / "src" / "candidate_extractor.py").read_text(encoding="utf-8")
        self.assertIn("\uff1a", src, "full-width colon separator support was removed")

    def test_no_kana_or_hangul_in_tests(self):
        # Kana/hangul have no legitimate use in tests; ideographs are allowed
        # for Chinese document fixtures.
        problems = []
        for path in TEST_FILES:
            if path.name == Path(__file__).name:
                continue
            for lineno, ch, ctx in _hits(_KANA_HANGUL, path):
                problems.append(f"{path.name}:{lineno}: U+{ord(ch):04X} {ch!r} in {ctx}")
        self.assertEqual(problems, [], "kana/hangul leaked into tests:\n" + "\n".join(problems))


if __name__ == "__main__":
    unittest.main(verbosity=2)
