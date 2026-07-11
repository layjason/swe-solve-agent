"""Tests for Task 6: refinement loop."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config import AgentConfig
from src.refine import _context_from, _relocalize_feedback, refine
from src.validate import ValidationResult


def _vr(patch="p", score=1.0, full=False, target_total=3, target_passed=1,
        failing=None, broken=None):
    r = ValidationResult(
        patch=patch, score=score, used_targets=True,
        target_total=target_total, target_passed=target_passed,
        failing_tests=failing or [], broken_regressions=broken or [],
        test_output="combined pytest output",
    )
    if full:
        r = ValidationResult(patch=patch, score=2.0, used_targets=True,
                             target_total=target_total, target_passed=target_total)
    return r


def _config():
    return AgentConfig(max_refine_iters=3, token_budget=999_999)


def _ctx():
    return SimpleNamespace(instance_id="x", problem_statement="bug", fail_to_pass=[], pass_to_pass=[])


def _model():
    m = MagicMock()
    m.get_usage.return_value = SimpleNamespace(total_tokens=0)
    return m


_LOC = {"files": ["f.py"], "functions": [], "regions": {}, "root_cause": "code"}


class TestContextBuilders(unittest.TestCase):
    def test_context_from_no_regression_output_duplication(self):
        best = _vr(failing=["t1"], broken=["r1"])
        ctx = _context_from(best)
        self.assertEqual(ctx.current_patch, best.patch)
        self.assertEqual(ctx.failing_tests, ["t1"])
        self.assertEqual(ctx.broken_regressions, ["r1"])
        self.assertEqual(ctx.test_output, "combined pytest output")
        # regression_output field was removed entirely (would duplicate test_output)
        self.assertFalse(hasattr(ctx, "regression_output"))

    def test_relocalize_feedback_mentions_prior_files(self):
        best = _vr(failing=["t1"], broken=["r1"])
        fb = _relocalize_feedback(best, ["a.py", "b.py"])
        self.assertIn("a.py", fb)
        self.assertIn("t1", fb)
        self.assertIn("r1", fb)


class TestRefine(unittest.TestCase):
    def test_empty_results_returns_none(self):
        self.assertIsNone(refine([], _ctx(), MagicMock(), _model(), _config(), _LOC, ""))

    def test_already_full_returns_without_generating(self):
        full = _vr(full=True)
        with patch("src.refine.generate_candidates") as gen:
            out = refine([full], _ctx(), MagicMock(), _model(), _config(), _LOC, "")
        self.assertEqual(out, full)
        gen.assert_not_called()

    def test_refinement_finds_full_patch(self):
        initial = _vr(score=1.3, target_passed=1)
        full = _vr(patch="fixed", full=True)
        with patch("src.refine.generate_candidates", return_value=["fixed"]), \
             patch("src.refine.validate", return_value=[full]):
            out = refine([initial], _ctx(), MagicMock(), _model(), _config(), _LOC, "")
        self.assertTrue(out.is_full)
        self.assertEqual(out.patch, "fixed")

    def test_no_progress_triggers_relocalize_once(self):
        initial = _vr(score=1.3, target_passed=1)
        stagnant = _vr(patch="same", score=1.3, target_passed=1)  # no improvement
        with patch("src.refine.generate_candidates", return_value=["c"]), \
             patch("src.refine.validate", return_value=[stagnant]), \
             patch("src.refine.localize", return_value=_LOC) as loc, \
             patch("src.refine.analyze_refinement_feedback", return_value="analysis note"):
            refine([initial], _ctx(), MagicMock(), _model(), _config(), _LOC, "")
        # re-localized exactly once despite 3 iters of no progress
        self.assertEqual(loc.call_count, 1)

    def test_budget_exhaustion_returns_best_so_far(self):
        initial = _vr(score=1.3, target_passed=1)
        model = _model()
        with patch("src.refine.budget_exceeded", return_value=True), \
             patch("src.refine.generate_candidates") as gen:
            out = refine([initial], _ctx(), MagicMock(), model, _config(), _LOC, "")
        self.assertEqual(out, initial)
        gen.assert_not_called()

    def test_keeps_better_batch_result(self):
        initial = _vr(patch="init", score=1.2, target_passed=1)
        better = _vr(patch="better", score=1.6, target_passed=2)
        # one iter improves, subsequent iters return nothing new
        with patch("src.refine.generate_candidates", side_effect=[["better"], [], []]), \
             patch("src.refine.validate", return_value=[better]), \
             patch("src.refine.analyze_refinement_feedback", return_value=""), \
             patch("src.refine.localize", return_value=_LOC):
            out = refine([initial], _ctx(), MagicMock(), _model(), _config(), _LOC, "")
        self.assertEqual(out.patch, "better")

    def test_no_progress_triggers_analysis_once(self):
        """Analysis should be called exactly once after first stall."""
        initial = _vr(score=1.3, target_passed=1)
        stagnant = _vr(patch="same", score=1.3, target_passed=1)
        with patch("src.refine.generate_candidates", return_value=["c"]), \
             patch("src.refine.validate", return_value=[stagnant]), \
             patch("src.refine.localize", return_value=_LOC), \
             patch("src.refine.analyze_refinement_feedback", return_value="analysis") as analysis:
            refine([initial], _ctx(), MagicMock(), _model(), _config(), _LOC, "")
        # analysis called exactly once despite 3 iters of no progress
        self.assertEqual(analysis.call_count, 1)

    def test_analysis_not_called_on_progress(self):
        """Analysis should NOT be called when refinement finds a full patch immediately."""
        initial = _vr(score=1.0, target_passed=1)
        full = _vr(patch="full", score=2.0, target_passed=2, full=True)
        with patch("src.refine.generate_candidates", return_value=["full"]), \
             patch("src.refine.validate", return_value=[full]), \
             patch("src.refine.analyze_refinement_feedback") as analysis:
            refine([initial], _ctx(), MagicMock(), _model(), _config(), _LOC, "")
        analysis.assert_not_called()

    def test_budget_prevents_analysis(self):
        """budget_exceeded should prevent analysis from running."""
        initial = _vr(score=1.3, target_passed=1)
        stagnant = _vr(patch="same", score=1.3, target_passed=1)
        with patch("src.refine.generate_candidates", return_value=["c"]), \
             patch("src.refine.validate", return_value=[stagnant]), \
             patch("src.refine.localize", return_value=_LOC), \
             patch("src.refine.budget_exceeded", return_value=True), \
             patch("src.refine.analyze_refinement_feedback") as analysis:
            refine([initial], _ctx(), MagicMock(), _model(), _config(), _LOC, "")
        analysis.assert_not_called()

    def test_analysis_note_passed_to_next_generation(self):
        """Next generate_candidates should receive RefinementContext with analysis_note."""
        initial = _vr(score=1.3, target_passed=1)
        stagnant = _vr(patch="same", score=1.3, target_passed=1)
        captured_refinement = []

        def capture_gen(*args, **kwargs):
            if kwargs.get("refinement"):
                captured_refinement.append(kwargs["refinement"])
            return ["c"]

        with patch("src.refine.generate_candidates", side_effect=capture_gen), \
             patch("src.refine.validate", return_value=[stagnant]), \
             patch("src.refine.localize", return_value=_LOC), \
             patch("src.refine.analyze_refinement_feedback", return_value="my analysis note"):
            refine([initial], _ctx(), MagicMock(), _model(), _config(), _LOC, "")
        # After first stall, next iteration should have analysis_note
        self.assertTrue(any(r.analysis_note == "my analysis note" for r in captured_refinement))

    def test_analysis_enriches_relocalization_feedback(self):
        """localize feedback should contain 'Refinement analysis:' when analysis exists."""
        initial = _vr(score=1.3, target_passed=1)
        stagnant = _vr(patch="same", score=1.3, target_passed=1)
        captured_feedback = []

        def capture_loc(*args, **kwargs):
            if kwargs.get("feedback"):
                captured_feedback.append(kwargs["feedback"])
            return _LOC

        with patch("src.refine.generate_candidates", return_value=["c"]), \
             patch("src.refine.validate", return_value=[stagnant]), \
             patch("src.refine.localize", side_effect=capture_loc), \
             patch("src.refine.analyze_refinement_feedback", return_value="deep insight"):
            refine([initial], _ctx(), MagicMock(), _model(), _config(), _LOC, "")
        self.assertTrue(any("Hypothesis from refinement analysis" in f for f in captured_feedback))
        self.assertTrue(any("deep insight" in f for f in captured_feedback))


if __name__ == "__main__":
    unittest.main()
