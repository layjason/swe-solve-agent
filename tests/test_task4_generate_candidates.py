"""Tests for Task 4: generate_candidates plan-then-generate."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from src.config import AgentConfig, RefinementContext
from src.generate import generate_candidates


def _mock_model(plan="Fix by changing return value in foo().", gen_output=""):
    model = MagicMock()
    model.get_usage.return_value = SimpleNamespace(total_tokens=0)
    responses = [
        SimpleNamespace(content=plan),   # plan call
        SimpleNamespace(content=gen_output),  # gen call
    ] * 10  # enough for multiple candidates
    model.generate.side_effect = responses
    return model


def _mock_env(diff="diff --git a/f.py b/f.py\n"):
    env = MagicMock()
    env.run.return_value = SimpleNamespace(exit_code=0, stdout=diff, stderr="")
    env.read_file.return_value = "def foo():\n    return 1\n"
    return env


def _config(n=2):
    return AgentConfig(n_candidates=n, token_budget=999_999)


def _context():
    return SimpleNamespace(problem_statement="Fix the bug in foo().")


def _loc():
    return {
        "files": ["f.py"],
        "functions": [{"file": "f.py", "name": "foo"}],
        "regions": {"f.py::foo": "def foo():\n    return 1\n"},
        "root_cause": "def foo():\n    return 1\n",
    }


_VALID_EDIT = (
    "f.py\n"
    "<<<<<<< SEARCH\n"
    "    return 1\n"
    "=======\n"
    "    return 2\n"
    ">>>>>>> REPLACE\n"
)


class TestGenerateCandidates(unittest.TestCase):
    def test_returns_n_patches(self):
        gen = _VALID_EDIT
        model = _mock_model(gen_output=gen)
        env = _mock_env()
        with patch("src.generate.apply_edits") as mock_apply, \
             patch("src.generate.parse_edits") as mock_parse, \
             patch("src.generate.revert"):
            mock_parse.return_value = [MagicMock()]
            mock_apply.return_value = SimpleNamespace(diff="diff --git a/f.py b/f.py\n", applied=["f.py[0]"], failed=[])
            patches = generate_candidates(_context(), env, model, _config(n=2), _loc())
        self.assertEqual(len(patches), 2)

    def test_short_plan_skipped(self):
        """A plan shorter than _MIN_PLAN_CHARS is skipped without reverting."""
        model = _mock_model(plan="x")  # too short
        env = _mock_env()
        with patch("src.generate.revert") as mock_revert:
            patches = generate_candidates(_context(), env, model, _config(n=1), _loc())
        self.assertEqual(patches, [])
        mock_revert.assert_not_called()

    def test_no_revert_on_empty_edits(self):
        """When parse_edits returns [], revert must NOT be called."""
        model = _mock_model()
        env = _mock_env()
        with patch("src.generate.parse_edits", return_value=[]), \
             patch("src.generate.revert") as mock_revert:
            generate_candidates(_context(), env, model, _config(n=1), _loc())
        mock_revert.assert_not_called()

    def test_budget_exceeded_stops_loop(self):
        model = _mock_model()
        model.get_usage.return_value = SimpleNamespace(total_tokens=999_999_999)
        patches = generate_candidates(_context(), _mock_env(), model, _config(n=3), _loc())
        self.assertEqual(patches, [])

    def test_first_batch_uses_diverse_suffixes(self):
        """First batch of N=3 should produce 3 different plan user prompts."""
        model = _mock_model()
        env = _mock_env()
        with patch("src.generate.parse_edits", return_value=[]), \
             patch("src.generate.revert"):
            generate_candidates(_context(), env, model, _config(n=3), _loc(), first_batch=True)
        # calls alternate: plan(0), gen(1), plan(2), gen(3), plan(4), gen(5)
        all_calls = model.generate.call_args_list
        plan_user_prompts = [all_calls[i][0][1] for i in range(0, len(all_calls), 2)]
        # 3 plan prompts should differ (different suffixes applied)
        self.assertEqual(len(plan_user_prompts), 3)
        self.assertEqual(len(set(plan_user_prompts)), 3)

    def test_refinement_context_in_prompt(self):
        """RefinementContext fields appear in the plan prompt."""
        model = _mock_model()
        env = _mock_env()
        ref = RefinementContext(
            current_patch="diff --git a/f.py b/f.py\n",
            failing_tests=["test_foo"],
            test_output="FAILED test_foo",
        )
        with patch("src.generate.parse_edits", return_value=[]), \
             patch("src.generate.revert"):
            generate_candidates(_context(), env, model, _config(n=1), _loc(), refinement=ref)
        first_plan_prompt = model.generate.call_args_list[0][0][1]
        self.assertIn("PREVIOUS PATCH", first_plan_prompt)
        self.assertIn("test_foo", first_plan_prompt)
        self.assertIn("FAILED test_foo", first_plan_prompt)


if __name__ == "__main__":
    unittest.main()
