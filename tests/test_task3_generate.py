"""Tests for Task 3: SEARCH/REPLACE parser and deterministic applier."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from src.generate import Edit, apply_edits, parse_edits, revert


def _mock_env(file_content="def foo():\n    return 1\n", write_ok=True,
              syntax_ok=True, lint_ok=True, diff_out="diff --git a/f.py b/f.py\n"):
    env = MagicMock()

    def run_side(cmd, timeout=60):
        if "git diff" in cmd:
            return SimpleNamespace(exit_code=0, stdout=diff_out, stderr="")
        if "ast.parse" in cmd:
            return SimpleNamespace(exit_code=0 if syntax_ok else 1, stdout="", stderr="")
        if "flake8" in cmd:
            return SimpleNamespace(exit_code=0 if lint_ok else 1, stdout="", stderr="")
        if "base64" in cmd:
            return SimpleNamespace(exit_code=0 if write_ok else 1, stdout="", stderr="")
        # git checkout revert
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    env.run.side_effect = run_side
    env.read_file.return_value = file_content
    return env


class TestParseEdits(unittest.TestCase):
    def test_single_block(self):
        text = (
            "f.py\n"
            "<<<<<<< SEARCH\n"
            "return 1\n"
            "=======\n"
            "return 2\n"
            ">>>>>>> REPLACE\n"
        )
        edits = parse_edits(text)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0].filepath, "f.py")
        self.assertEqual(edits[0].search, "return 1\n")
        self.assertEqual(edits[0].replace, "return 2\n")

    def test_fenced_block(self):
        text = (
            "f.py\n"
            "```python\n"
            "<<<<<<< SEARCH\n"
            "old\n"
            "=======\n"
            "new\n"
            ">>>>>>> REPLACE\n"
            "```"
        )
        edits = parse_edits(text)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0].search, "old\n")

    def test_no_block_returns_empty(self):
        self.assertEqual(parse_edits("no edits here"), [])

    def test_multiple_blocks(self):
        text = (
            "a.py\n<<<<<<< SEARCH\nold_a\n=======\nnew_a\n>>>>>>> REPLACE\n"
            "b.py\n<<<<<<< SEARCH\nold_b\n=======\nnew_b\n>>>>>>> REPLACE\n"
        )
        edits = parse_edits(text)
        self.assertEqual(len(edits), 2)
        self.assertEqual(edits[0].filepath, "a.py")
        self.assertEqual(edits[1].filepath, "b.py")


class TestApplyEdits(unittest.TestCase):
    def test_successful_apply(self):
        env = _mock_env(file_content="def foo():\n    return 1\n")
        edits = [Edit("f.py", "return 1\n", "return 42\n")]
        result = apply_edits(edits, env)
        self.assertEqual(len(result.applied), 1)
        self.assertEqual(len(result.failed), 0)
        self.assertIn("diff --git", result.diff)

    def test_no_match_skipped(self):
        env = _mock_env(file_content="def foo():\n    return 1\n")
        edits = [Edit("f.py", "return 999\n", "return 0\n")]
        result = apply_edits(edits, env)
        self.assertEqual(result.applied, [])
        self.assertEqual(len(result.failed), 1)
        self.assertIn("not found", result.failed[0])

    def test_syntax_fail_reverts_file(self):
        env = _mock_env(syntax_ok=False)
        edits = [Edit("f.py", "return 1\n", "return 2\n")]
        result = apply_edits(edits, env)
        self.assertEqual(result.applied, [])
        self.assertIn("syntax", result.failed[0])
        # revert call must have been made
        revert_calls = [str(c) for c in env.run.call_args_list if "checkout" in str(c)]
        self.assertTrue(len(revert_calls) >= 1)

    def test_lint_fail_reverts_file(self):
        env = _mock_env(syntax_ok=True, lint_ok=False)
        edits = [Edit("f.py", "return 1\n", "return 2\n")]
        result = apply_edits(edits, env)
        self.assertEqual(result.applied, [])
        self.assertIn("lint", result.failed[0])


class TestRevert(unittest.TestCase):
    def test_revert_calls_checkout_and_clean(self):
        env = MagicMock()
        env.run.return_value = SimpleNamespace(exit_code=0, stdout="", stderr="")
        revert(env)
        cmds = [c[0][0] for c in env.run.call_args_list]
        self.assertTrue(any("checkout" in c for c in cmds))
        self.assertTrue(any("clean" in c for c in cmds))


class TestDiffNotStripped(unittest.TestCase):
    """Regression: apply_edits must NOT strip the git diff — stripping drops the trailing
    newline and final context line, corrupting the patch for `git apply`."""

    def test_trailing_newline_and_blank_context_preserved(self):
        # diff ends with a blank context line ' ' then a final newline.
        diff = (
            "diff --git a/f.py b/f.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-x\n"
            "+y\n"
            " \n"
        )
        env = _mock_env(diff_out=diff)
        result = apply_edits([Edit("f.py", "x\n", "y\n")], env)
        self.assertEqual(result.diff, diff)          # byte-for-byte, not stripped
        self.assertTrue(result.diff.endswith("\n"))  # trailing newline intact

    def test_empty_diff_collapses_to_empty_string(self):
        env = _mock_env(diff_out="\n  \n")
        result = apply_edits([Edit("f.py", "x\n", "y\n")], env)
        self.assertEqual(result.diff, "")


class TestResolvePath(unittest.TestCase):
    """Regression (Bug B): a bare filename must resolve to its tracked repo-relative path."""

    def _env_with_tracked(self, tracked, exists_as_given=False):
        env = MagicMock()

        def run_side(cmd, timeout=60):
            if cmd.startswith("test -f"):
                return SimpleNamespace(exit_code=0 if exists_as_given else 1, stdout="", stderr="")
            if "git ls-files" in cmd:
                return SimpleNamespace(exit_code=0, stdout="\n".join(tracked) + "\n", stderr="")
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        env.run.side_effect = run_side
        return env

    def test_bare_filename_resolves_to_full_path(self):
        from src.generate import _resolve_path
        env = self._env_with_tracked(["astropy/io/fits/fitsrec.py"])
        self.assertEqual(_resolve_path(env, "fitsrec.py"), "astropy/io/fits/fitsrec.py")

    def test_existing_path_returned_as_is(self):
        from src.generate import _resolve_path
        env = self._env_with_tracked([], exists_as_given=True)
        self.assertEqual(_resolve_path(env, "astropy/io/fits/fitsrec.py"), "astropy/io/fits/fitsrec.py")

    def test_ambiguous_basename_returns_original(self):
        from src.generate import _resolve_path
        env = self._env_with_tracked(["a/utils.py", "b/utils.py"])
        self.assertEqual(_resolve_path(env, "utils.py"), "utils.py")  # ambiguous → no guess


if __name__ == "__main__":
    unittest.main()
