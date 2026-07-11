"""Tests for Task 1: AgentConfig, budget_exceeded, solve_task pipeline parity."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config import AgentConfig, ValidationMode, budget_exceeded


class TestAgentConfig(unittest.TestCase):
    def test_defaults(self):
        c = AgentConfig()
        self.assertEqual(c.n_candidates, 3)
        self.assertEqual(c.max_refine_iters, 3)
        self.assertEqual(c.token_budget, 200_000)
        self.assertEqual(c.validation_mode, ValidationMode.TARGET_WITH_TEST_PATCH)
        self.assertEqual(c.regression_sample_size, 10)
        self.assertEqual(c.localization_top_k, 5)

    def test_override(self):
        c = AgentConfig(n_candidates=4, token_budget=50_000)
        self.assertEqual(c.n_candidates, 4)
        self.assertEqual(c.token_budget, 50_000)
        # other fields keep defaults
        self.assertEqual(c.max_refine_iters, 3)


class TestBudgetExceeded(unittest.TestCase):
    def _model(self, total):
        m = MagicMock()
        m.get_usage.return_value = SimpleNamespace(total_tokens=total)
        return m

    def test_not_exceeded(self):
        config = AgentConfig(token_budget=100_000)
        self.assertFalse(budget_exceeded(self._model(50_000), config, baseline_tokens=0))

    def test_exactly_at_budget(self):
        config = AgentConfig(token_budget=100_000)
        self.assertTrue(budget_exceeded(self._model(100_000), config, baseline_tokens=0))

    def test_baseline_offset(self):
        # only tokens spent in this instance count
        config = AgentConfig(token_budget=10_000)
        self.assertFalse(budget_exceeded(self._model(110_000), config, baseline_tokens=105_000))
        self.assertTrue(budget_exceeded(self._model(115_001), config, baseline_tokens=105_000))


class TestSolveTaskParity(unittest.TestCase):
    """solve_task must return a non-empty string (patch) when _generate returns a patch."""

    def _make_env(self):
        env = MagicMock()
        env.run.return_value = SimpleNamespace(exit_code=0, stdout="", stderr="")
        return env

    def _make_model(self, patch_content="diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x\n+y"):
        model = MagicMock()
        # First call returns a final_patch action; get_usage returns 0 always
        model.generate.return_value = SimpleNamespace(
            content=f"<final_patch>\n{patch_content}\n</final_patch>",
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )
        model.get_usage.return_value = SimpleNamespace(total_tokens=0)
        return model

    def _make_context(self):
        return SimpleNamespace(
            instance_id="test-1",
            problem_statement="Fix the bug.",
            hints_text="",
            fail_to_pass=[],
            pass_to_pass=[],
            test_patch="",
        )

    def test_returns_string(self):
        from src.agent import solve_task
        with patch("src.agent.localize", return_value={"root_cause": "some context"}):
            result = solve_task(self._make_context(), self._make_env(), self._make_model())
        self.assertIsInstance(result, str)

    def test_returns_patch_content(self):
        from src.agent import solve_task
        # Mock localize to return valid root_cause (so guard passes)
        # and generate_candidates to return a patch directly
        with patch("src.agent.localize", return_value={"root_cause": "some context"}), \
             patch("src.agent.generate_candidates", return_value=["diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x\n+y"]):
            result = solve_task(self._make_context(), self._make_env(), self._make_model())
        self.assertIn("diff --git", result)

    def test_returns_empty_on_no_candidates(self):
        from src.agent import solve_task
        model = self._make_model()
        # When generate_candidates returns empty (e.g., budget exhausted internally),
        # and no prior results exist, solve_task returns ""
        with patch("src.agent.localize", return_value={"root_cause": "some context"}), \
             patch("src.agent.generate_candidates", return_value=[]):
            result = solve_task(self._make_context(), self._make_env(), model)
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
