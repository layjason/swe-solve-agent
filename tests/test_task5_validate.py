"""Tests for Task 5: test-execution validation + ranking."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config import AgentConfig, ValidationMode
from src.validate import (
    ValidationResult,
    _as_list,
    _backfill_missing_passes,
    _parse_pytest,
    _pytest_summary_counts,
    _sample_regression,
    _score,
    _summary_nonpass_count,
    validate,
)


class TestHelpers(unittest.TestCase):
    def test_as_list_from_list(self):
        self.assertEqual(_as_list(["a", "b"]), ["a", "b"])

    def test_as_list_from_json_string(self):
        self.assertEqual(_as_list('["a", "b"]'), ["a", "b"])

    def test_as_list_from_garbage(self):
        self.assertEqual(_as_list("not json"), [])
        self.assertEqual(_as_list(None), [])

    def test_sample_regression_deterministic(self):
        pool = [f"t{i}" for i in range(50)]
        a = _sample_regression(pool, "inst-1", 10)
        b = _sample_regression(pool, "inst-1", 10)
        self.assertEqual(a, b)            # seeded by instance_id
        self.assertEqual(len(a), 10)

    def test_sample_regression_small_pool(self):
        self.assertEqual(_sample_regression(["a", "b"], "x", 10), ["a", "b"])

    def test_pytest_summary_counts_all_passed(self):
        output = "=== 10 passed in 1.88s ==="
        counts = _pytest_summary_counts(output)
        self.assertEqual(counts, {"passed": 10})

    def test_pytest_summary_counts_mixed(self):
        output = "=== 1 failed, 2 passed in 2.50s ==="
        counts = _pytest_summary_counts(output)
        self.assertEqual(counts, {"failed": 1, "passed": 2})

    def test_pytest_summary_counts_with_errors(self):
        output = "=== 1 failed, 1 error, 3 passed in 0.56s ==="
        counts = _pytest_summary_counts(output)
        self.assertEqual(counts, {"failed": 1, "errors": 1, "passed": 3})

    def test_pytest_summary_counts_quiet(self):
        output = "1 passed in 0.12s"
        counts = _pytest_summary_counts(output)
        self.assertEqual(counts, {"passed": 1})

    def test_summary_nonpass_count_empty(self):
        self.assertEqual(_summary_nonpass_count({}), 0)

    def test_summary_nonpass_count_all_passed(self):
        self.assertEqual(_summary_nonpass_count({"passed": 10}), 0)

    def test_summary_nonpass_count_with_failures(self):
        self.assertEqual(_summary_nonpass_count({"failed": 2, "passed": 3}), 2)

    def test_summary_nonpass_count_with_errors(self):
        self.assertEqual(_summary_nonpass_count({"errors": 1, "passed": 2}), 1)

    def test_summary_nonpass_count_mixed(self):
        self.assertEqual(_summary_nonpass_count({"failed": 1, "errors": 1, "skipped": 1, "passed": 2}), 3)

    def test_backfill_all_pass(self):
        status = {}
        counts = {"passed": 3}
        _backfill_missing_passes(status, ["a", "b", "c"], counts)
        self.assertEqual(status, {"a": "PASSED", "b": "PASSED", "c": "PASSED"})

    def test_backfill_partial_with_failure(self):
        status = {"a": "FAILED"}
        counts = {"failed": 1, "passed": 2}
        _backfill_missing_passes(status, ["a", "b", "c"], counts)
        self.assertEqual(status["a"], "FAILED")
        self.assertEqual(status["b"], "PASSED")
        self.assertEqual(status["c"], "PASSED")

    def test_backfill_skips_when_unknown_errors(self):
        status = {"a": "FAILED"}
        counts = {"failed": 1, "errors": 1, "passed": 1}
        _backfill_missing_passes(status, ["a", "b", "c"], counts)
        self.assertEqual(status["a"], "FAILED")
        self.assertNotIn("b", status)
        self.assertNotIn("c", status)

    def test_backfill_skips_when_error_not_in_status(self):
        """If summary says 1 error but we didn't parse the ERROR line, don't guess."""
        status = {}
        counts = {"errors": 1, "passed": 2}
        _backfill_missing_passes(status, ["a", "b", "c"], counts)
        self.assertEqual(status, {})

    def test_backfill_no_passed_in_summary(self):
        status = {}
        counts = {"failed": 2}
        _backfill_missing_passes(status, ["a", "b"], counts)
        self.assertEqual(status, {})

    def test_parse_pytest(self):
        out = (
            "PASSED path/test_a.py::test_one\n"
            "FAILED path/test_a.py::test_two - AssertionError: nope\n"
            "ERROR path/test_b.py::test_three\n"
            "random noise line\n"
        )
        m = _parse_pytest(out)
        self.assertEqual(m["path/test_a.py::test_one"], "PASSED")
        self.assertEqual(m["path/test_a.py::test_two"], "FAILED")
        self.assertEqual(m["path/test_b.py::test_three"], "ERROR")

    def test_run_tests_pytest3_all_passed_fallback(self):
        """pytest 3.x emits no per-test PASSED lines; only 'N passed' in summary."""
        from src.validate import _run_tests
        env = MagicMock()
        env.run.return_value = SimpleNamespace(
            exit_code=0,
            stdout="collected 3 items\n...=== 3 passed in 1.23s ===\n",
            stderr="",
        )
        node_ids = ["t.py::a", "t.py::b", "t.py::c"]
        status, _ = _run_tests(env, node_ids, timeout=60)
        for nid in node_ids:
            self.assertEqual(status[nid], "PASSED")

    def test_run_tests_pytest3_mixed_backfills(self):
        """pytest 3.x: 1 failed, 2 passed — backfills the 2 passed tests."""
        from src.validate import _run_tests
        env = MagicMock()
        env.run.return_value = SimpleNamespace(
            exit_code=1,
            stdout=(
                "FAILED t.py::a - AssertionError\n"
                "...=== 1 failed, 2 passed in 1.23s ===\n"
            ),
            stderr="",
        )
        node_ids = ["t.py::a", "t.py::b", "t.py::c"]
        status, _ = _run_tests(env, node_ids, timeout=60)
        self.assertEqual(status["t.py::a"], "FAILED")
        self.assertEqual(status["t.py::b"], "PASSED")
        self.assertEqual(status["t.py::c"], "PASSED")

    def test_run_tests_pytest3_error_backfills(self):
        """pytest 3.x: 1 error, 2 passed — backfills the 2 passed tests."""
        from src.validate import _run_tests
        env = MagicMock()
        env.run.return_value = SimpleNamespace(
            exit_code=1,
            stdout=(
                "ERROR t.py::a - ImportError\n"
                "...=== 1 error, 2 passed in 1.23s ===\n"
            ),
            stderr="",
        )
        node_ids = ["t.py::a", "t.py::b", "t.py::c"]
        status, _ = _run_tests(env, node_ids, timeout=60)
        self.assertEqual(status["t.py::a"], "ERROR")
        self.assertEqual(status["t.py::b"], "PASSED")
        self.assertEqual(status["t.py::c"], "PASSED")


class TestScoreOfficialGrading(unittest.TestCase):
    """Scoring must match official grading: broken regression is a hard gate."""

    def test_full_resolution_is_max(self):
        # all targets pass, no breaks -> 2.0
        s = _score(used_targets=True, target_total=3, target_passed=3, regression_total=10, broken=0)
        self.assertEqual(s, 2.0)

    def test_clean_partial_beats_breaking_full(self):
        # 2/3 targets, 0 breaks  vs  3/3 targets, 5 breaks
        clean = _score(used_targets=True, target_total=3, target_passed=2, regression_total=10, broken=0)
        breaking = _score(used_targets=True, target_total=3, target_passed=3, regression_total=10, broken=5)
        self.assertGreater(clean, breaking)
        self.assertGreaterEqual(clean, 1.0)
        self.assertLess(breaking, 1.0)

    def test_any_break_below_any_clean(self):
        # even a do-nothing clean patch (0 targets, 0 breaks) outranks a target-fixing breaker
        clean_zero = _score(used_targets=True, target_total=3, target_passed=0, regression_total=10, broken=0)
        breaker = _score(used_targets=True, target_total=3, target_passed=3, regression_total=10, broken=1)
        self.assertGreaterEqual(clean_zero, 1.0)
        self.assertLess(breaker, 1.0)
        self.assertGreater(clean_zero, breaker)

    def test_fallback_regression_fraction(self):
        s = _score(used_targets=False, target_total=0, target_passed=0, regression_total=10, broken=2)
        self.assertAlmostEqual(s, 0.8)


class TestValidationResult(unittest.TestCase):
    def test_is_full_true(self):
        r = ValidationResult(patch="p", score=2.0, used_targets=True, target_total=2, target_passed=2)
        self.assertTrue(r.is_full)

    def test_is_full_false_on_broken_regression(self):
        r = ValidationResult(patch="p", score=1.9, used_targets=True, target_total=2,
                             target_passed=2, broken_regressions=["t1"])
        self.assertFalse(r.is_full)

    def test_is_full_false_when_no_targets(self):
        r = ValidationResult(patch="p", score=1.0, used_targets=False)
        self.assertFalse(r.is_full)


def _ctx():
    return SimpleNamespace(
        instance_id="astropy__astropy-12907",
        fail_to_pass=["test_mod.py::test_target"],
        pass_to_pass=["test_mod.py::test_reg1", "test_mod.py::test_reg2"],
    )


class TestValidateRanking(unittest.TestCase):
    def test_ranks_clean_above_breaker(self):
        """validate() must sort a clean partial patch above a target-fixing regression-breaker."""
        config = AgentConfig(validation_mode=ValidationMode.TARGET_WITH_TEST_PATCH)
        env = MagicMock()

        breaker = ValidationResult(
            patch="A", score=_score(True, 3, 3, 10, 5), used_targets=True,
            target_total=3, target_passed=3, broken_regressions=["r1", "r2", "r3", "r4", "r5"],
        )
        clean = ValidationResult(
            patch="B", score=_score(True, 3, 2, 10, 0), used_targets=True,
            target_total=3, target_passed=2,
        )
        # _validate_one returns breaker for "A", clean for "B"
        mapping = {"A": breaker, "B": clean}
        with patch("src.validate._validate_one", side_effect=lambda p, c, e, cfg, tp, bo="": mapping[p]):
            results = validate(["A", "B"], _ctx(), env, config, test_patch="TP")

        self.assertEqual(results[0].patch, "B")   # clean wins despite fewer targets
        self.assertGreaterEqual(results[0].score, 1.0)
        self.assertLess(results[1].score, 1.0)

    def test_full_patch_ranks_first(self):
        config = AgentConfig()
        env = MagicMock()
        full = ValidationResult(patch="F", score=_score(True, 2, 2, 10, 0), used_targets=True,
                                target_total=2, target_passed=2)
        partial = ValidationResult(patch="P", score=_score(True, 2, 1, 10, 0), used_targets=True,
                                   target_total=2, target_passed=1)
        mapping = {"F": full, "P": partial}
        with patch("src.validate._validate_one", side_effect=lambda p, c, e, cfg, tp, bo="": mapping[p]):
            results = validate(["P", "F"], _ctx(), env, config, test_patch="TP")
        self.assertEqual(results[0].patch, "F")
        self.assertTrue(results[0].is_full)

    def test_apply_failure_scores_lowest(self):
        config = AgentConfig()
        env = MagicMock()
        env.run.return_value = SimpleNamespace(exit_code=0, stdout="", stderr="")
        with patch("src.validate._apply_patch", return_value=False):
            results = validate(["some patch"], _ctx(), env, config, test_patch="TP")
        self.assertEqual(results[0].score, -1.0)
        self.assertFalse(results[0].applied)

class TestApplyPatch(unittest.TestCase):
    """Regression for the 'corrupt patch at line N' failure: a missing trailing newline
    must be added, and a lenient `patch` fallback must run when git apply fails."""

    def test_trailing_newline_added_before_write(self):
        from src.validate import _apply_patch
        env = MagicMock()
        env.run.return_value = SimpleNamespace(exit_code=0, stdout="", stderr="")
        _apply_patch(env, "diff --git a/f b/f\n@@ -1 +1 @@\n-x\n+y", "candidate")  # no final \n
        # the base64-encoded payload must decode to a string ending in newline
        import base64
        write_cmd = next(c[0][0] for c in env.run.call_args_list if "base64" in c[0][0])
        b64 = write_cmd.split()[-1].strip("'")
        self.assertTrue(base64.b64decode(b64).decode().endswith("\n"))

    def test_patch_fallback_when_git_apply_fails(self):
        from src.validate import _apply_patch
        env = MagicMock()
        calls = {"git_apply": 0}

        def run_side(cmd, timeout=60):
            if "git apply" in cmd:
                calls["git_apply"] += 1
                return SimpleNamespace(exit_code=1, stdout="", stderr="corrupt patch")
            if cmd.startswith("patch "):
                return SimpleNamespace(exit_code=0, stdout="", stderr="")
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        env.run.side_effect = run_side
        ok = _apply_patch(env, "diff\n@@ -1 +1 @@\n-x\n+y\n", "candidate")
        self.assertTrue(ok)                    # patch fallback succeeded
        self.assertEqual(calls["git_apply"], 2)  # tried plain + 3way first


if __name__ == "__main__":
    unittest.main()
