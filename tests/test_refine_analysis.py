"""Tests for refinement analysis on escalation."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config import AgentConfig
from src.refine import AttemptLedger
from src.refine_analysis import _extract_analysis, analyze_refinement_feedback
from src.validate import ValidationResult


def _vr(patch="p", score=1.0, target_total=3, target_passed=1):
    return ValidationResult(
        patch=patch, score=score, used_targets=True,
        target_total=target_total, target_passed=target_passed,
        failing_tests=["t1"], broken_regressions=[],
        test_output="pytest output here",
        target_frames=["f.py::func:10"],
        regression_frames=[],
    )


def _config():
    return AgentConfig(max_refine_iters=3, token_budget=999_999)


def _ctx():
    return SimpleNamespace(instance_id="x", problem_statement="bug description", fail_to_pass=[], pass_to_pass=[])


def _model():
    m = MagicMock()
    m.get_usage.return_value = SimpleNamespace(total_tokens=0)
    return m


_LOC = {"files": ["f.py"], "functions": [], "regions": {"f.py::func": "code here"}, "root_cause": "root cause code"}


class TestExtractAnalysis(unittest.TestCase):
    def test_extracts_from_tags(self):
        text = "<analysis>\nFailure summary: blah\nLikely cause: xyz\n</analysis>"
        result = _extract_analysis(text)
        self.assertIn("Failure summary", result)
        self.assertIn("Likely cause", result)

    def test_fallback_to_stripped_text(self):
        text = "Just some analysis without tags"
        result = _extract_analysis(text)
        self.assertEqual(result, text)

    def test_caps_output(self):
        text = "<analysis>\n" + "x" * 5000 + "\n</analysis>"
        result = _extract_analysis(text)
        self.assertLessEqual(len(result), 1800)


class TestAnalyzeRefinementFeedback(unittest.TestCase):
    def test_returns_empty_on_budget_exceeded(self):
        with patch("src.refine_analysis.budget_exceeded", return_value=True):
            result = analyze_refinement_feedback(
                _ctx(), MagicMock(), _model(), _config(), _LOC, _vr(), AttemptLedger(), 0
            )
        self.assertEqual(result, "")

    def test_returns_analysis_on_success(self):
        model = _model()
        model.generate.return_value = SimpleNamespace(
            content="<analysis>\nFailure summary: test failure\n</analysis>"
        )
        with patch("src.refine_analysis.budget_exceeded", return_value=False), \
             patch("src.refine_analysis.run_tools", return_value=("", False)):
            result = analyze_refinement_feedback(
                _ctx(), MagicMock(), model, _config(), _LOC, _vr(), AttemptLedger(), 0
            )
        self.assertIn("Failure summary", result)

    def test_tool_loop_on_tool_calls(self):
        model = _model()
        # First call returns tool call, second returns analysis
        model.generate.side_effect = [
            SimpleNamespace(content='run_python("print(1)")'),
            SimpleNamespace(content="<analysis>\nFinal analysis\n</analysis>"),
        ]
        with patch("src.refine_analysis.budget_exceeded", return_value=False), \
             patch("src.refine_analysis.run_tools", side_effect=[("[run_python]\n1", True), ("", False)]):
            result = analyze_refinement_feedback(
                _ctx(), MagicMock(), model, _config(), _LOC, _vr(), AttemptLedger(), 0
            )
        self.assertIn("Final analysis", result)
        self.assertEqual(model.generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
