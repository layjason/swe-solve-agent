"""Tests for Task 2: hierarchical localization helpers."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call

from src.localize import _parse_json_list, _extract_region
from src.tools import run_tools as _run_tools, extract_call_arg as _extract_call_arg, parse_string_arg as _parse_string_arg


class TestParseJsonList(unittest.TestCase):
    def test_plain_array(self):
        self.assertEqual(_parse_json_list('["a.py", "b.py"]'), ["a.py", "b.py"])

    def test_with_leading_text(self):
        # leading [...] fragment must not fool the parser
        self.assertEqual(
            _parse_json_list('[1 file found] result: ["x.py"]'),
            ["x.py"],
        )

    def test_object_array(self):
        result = _parse_json_list('[{"file": "f.py", "name": "foo"}]')
        self.assertEqual(result, [{"file": "f.py", "name": "foo"}])

    def test_malformed_returns_empty(self):
        self.assertEqual(_parse_json_list("not json at all"), [])

    def test_nested_brackets(self):
        # Should parse the outer array even with nested brackets inside
        self.assertEqual(_parse_json_list('[["a", "b"], ["c"]]'), [["a", "b"], ["c"]])


class TestRunTools(unittest.TestCase):
    def _env(self, stdout="astropy/modeling/separable.py"):
        env = MagicMock()
        env.run.return_value = SimpleNamespace(exit_code=0, stdout=stdout, stderr="")
        return env

    def test_search_func_def_uses_extended_regex(self):
        env = self._env()
        _run_tools('search_func_def("separability_matrix")', env)
        cmd = env.run.call_args[0][0]
        self.assertIn("grep -rnE", cmd)
        self.assertIn("def", cmd)
        self.assertIn("separability_matrix", cmd)

    def test_search_class_def_uses_extended_regex(self):
        env = self._env()
        _run_tools('search_class_def("CompoundModel")', env)
        cmd = env.run.call_args[0][0]
        self.assertIn("grep -rnE", cmd)
        self.assertIn("class", cmd)
        self.assertIn("CompoundModel", cmd)

    def test_search_string_uses_fixed_string(self):
        env = self._env()
        _run_tools('search_string("foo")', env)
        cmd = env.run.call_args[0][0]
        self.assertIn("grep -rlnF", cmd)
        self.assertIn("foo", cmd)

    def test_arg_with_single_quote_rejected_as_invalid_identifier(self):
        env = self._env()
        # Single quote in identifier is invalid - should be rejected
        result, had_calls = _run_tools('search_func_def("it\'s_func")', env)
        self.assertTrue(had_calls)
        self.assertIn("Invalid identifier", result)
        self.assertFalse(env.run.called)

    def test_empty_stdout_excluded_from_results(self):
        env = self._env(stdout="")
        result, had_calls = _run_tools('search_func_def("foo")', env)
        self.assertEqual(result, "")
        self.assertTrue(had_calls)

    def test_no_tool_calls_returns_empty(self):
        env = self._env()
        result, had_calls = _run_tools("Here is the file list: ['a.py']", env)
        self.assertEqual(result, "")
        self.assertFalse(had_calls)
        env.run.assert_not_called()

    def test_run_python_basic(self):
        """run_python tool should execute Python code and return output."""
        env = self._env(stdout="['1.0D10']")
        result, had_calls = _run_tools(
            'run_python("print([\'1.0E10\'.replace(\'E\', \'D\')])")', env
        )
        self.assertTrue(had_calls)
        cmd = env.run.call_args[0][0]
        self.assertIn("python3 -I -c", cmd)
        self.assertIn("replace", cmd)
        self.assertIn("['1.0D10']", result)

    def test_run_python_with_parentheses_in_strings(self):
        """run_python should handle code with ')' inside strings (quote-aware parsing)."""
        env = self._env(stdout="test output")
        result, had_calls = _run_tools(
            'run_python("print(\'hello ) world\')")', env
        )
        self.assertTrue(had_calls)
        cmd = env.run.call_args[0][0]
        self.assertIn("python3 -I -c", cmd)
        self.assertIn("hello ) world", cmd)
        self.assertIn("test output", result)

    def test_run_python_with_nested_parentheses(self):
        """run_python should handle nested parentheses in function calls."""
        env = self._env(stdout="result")
        result, had_calls = _run_tools(
            'run_python("print(len([1, 2, 3]))")', env
        )
        self.assertTrue(had_calls)
        cmd = env.run.call_args[0][0]
        self.assertIn("python3 -I -c", cmd)
        self.assertIn("print(len([1, 2, 3]))", cmd)

    def test_run_python_with_escaped_quotes(self):
        """run_python should handle escaped quotes inside the string."""
        env = self._env(stdout="hi")
        result, had_calls = _run_tools(
            'run_python("print(\\"hi\\")")', env
        )
        self.assertTrue(had_calls)
        self.assertIn("hi", result)

    def test_run_python_with_triple_quotes(self):
        """run_python should handle triple-quoted strings."""
        env = self._env(stdout="ok")
        result, had_calls = _run_tools(
            'run_python("""print("ok")""")', env
        )
        self.assertTrue(had_calls)
        cmd = env.run.call_args[0][0]
        self.assertIn("python3 -I -c", cmd)

    def test_run_python_invalid_arg_returns_error(self):
        """run_python with non-string argument should return error message."""
        env = self._env()
        result, had_calls = _run_tools("run_python(123)", env)
        self.assertTrue(had_calls)
        self.assertIn("Invalid argument", result)
        self.assertFalse(env.run.called)

    def test_run_python_with_comment_containing_paren(self):
        """run_python should handle comments with ) inside them."""
        env = self._env(stdout="ok")
        result, had_calls = _run_tools(
            'run_python("# )\\nprint(\'ok\')")', env
        )
        self.assertTrue(had_calls)
        self.assertIn("ok", result)

    def test_run_python_malformed_call_detected(self):
        """Malformed run_python( calls should set had_calls and return error."""
        env = self._env()
        # Missing closing paren
        result, had_calls = _run_tools('run_python("print(\'oops\')"', env)
        self.assertTrue(had_calls)
        self.assertIn("Malformed call", result)
        self.assertFalse(env.run.called)

    def test_run_python_includes_stderr(self):
        """run_python should include stderr in output."""
        env = MagicMock()
        env.run.return_value = SimpleNamespace(
            exit_code=0, stdout="stdout output", stderr="stderr warning"
        )
        result, had_calls = _run_tools('run_python("print(\'test\')")', env)
        self.assertTrue(had_calls)
        self.assertIn("stdout output", result)
        self.assertIn("[stderr]", result)
        self.assertIn("stderr warning", result)

    def test_run_python_includes_exit_code(self):
        """run_python should include non-zero exit code in output."""
        env = MagicMock()
        env.run.return_value = SimpleNamespace(
            exit_code=1, stdout="", stderr="error message"
        )
        result, had_calls = _run_tools('run_python("raise Exception()")', env)
        self.assertTrue(had_calls)
        self.assertIn("[exit code: 1]", result)
        self.assertIn("error message", result)

    def test_run_python_output_capped(self):
        """run_python output should be capped to prevent blowup."""
        env = MagicMock()
        # Simulate huge output
        huge_output = "x" * 5000
        env.run.return_value = SimpleNamespace(
            exit_code=0, stdout=huge_output, stderr=""
        )
        result, had_calls = _run_tools('run_python("print(\'x\' * 5000)")', env)
        self.assertTrue(had_calls)
        self.assertIn("truncated", result)
        self.assertIn("5000 chars total", result)


class TestExtractCallArg(unittest.TestCase):
    """Tests for the quote-aware call argument extractor."""

    def test_simple_string(self):
        args = _extract_call_arg('run_python("hello")', "run_python")
        self.assertEqual(args, ['"hello"'])

    def test_paren_in_string(self):
        args = _extract_call_arg('run_python("print(\')\')")', "run_python")
        self.assertEqual(args, ['"print(\')\')"'])

    def test_nested_parens_outside_string(self):
        args = _extract_call_arg('run_python("print(len([1,2]))")', "run_python")
        self.assertEqual(args, ['"print(len([1,2]))"'])

    def test_triple_quoted_string(self):
        args = _extract_call_arg('run_python("""code""")', "run_python")
        self.assertEqual(args, ['"""code"""'])

    def test_escaped_quotes(self):
        args = _extract_call_arg('run_python("print(\\"hi\\")")', "run_python")
        self.assertEqual(args, ['"print(\\"hi\\")"'])

    def test_comment_with_paren(self):
        args = _extract_call_arg('run_python("# )\\nprint(1)")', "run_python")
        self.assertEqual(args, ['"# )\\nprint(1)"'])


class TestParseStringArg(unittest.TestCase):
    """Tests for the string literal parser."""

    def test_simple_string(self):
        self.assertEqual(_parse_string_arg('"hello"'), "hello")

    def test_single_quoted(self):
        self.assertEqual(_parse_string_arg("'hello'"), "hello")

    def test_triple_quoted(self):
        self.assertEqual(_parse_string_arg('"""hello"""'), "hello")

    def test_non_string_returns_none(self):
        self.assertIsNone(_parse_string_arg("123"))

    def test_invalid_literal_returns_none(self):
        self.assertIsNone(_parse_string_arg("not a literal"))

    def test_escaped_quotes(self):
        self.assertEqual(_parse_string_arg(r'"hello \"world\""'), 'hello "world"')


class TestSearchIdentifierValidation(unittest.TestCase):
    """Tests for identifier validation in search tools."""

    def _env(self, stdout="result"):
        env = MagicMock()
        env.run.return_value = SimpleNamespace(exit_code=0, stdout=stdout, stderr="")
        return env

    def test_valid_identifier_accepted(self):
        env = self._env()
        result, _ = _run_tools('search_func_def("my_func")', env)
        self.assertIn("result", result)

    def test_regex_chars_rejected(self):
        env = self._env()
        result, _ = _run_tools('search_func_def("foo.*")', env)
        self.assertIn("Invalid identifier", result)
        self.assertFalse(env.run.called)

    def test_identifier_with_numbers_accepted(self):
        env = self._env()
        result, _ = _run_tools('search_func_def("func123")', env)
        self.assertIn("result", result)

    def test_identifier_starting_with_number_rejected(self):
        env = self._env()
        result, _ = _run_tools('search_func_def("123func")', env)
        self.assertIn("Invalid identifier", result)


class TestExtractRegion(unittest.TestCase):
    def _env(self, grep_out="5:def separability_matrix(model):", sed_out="line1\nline2"):
        env = MagicMock()
        def run_side(cmd, timeout=60):
            if "grep" in cmd:
                return SimpleNamespace(exit_code=0, stdout=grep_out, stderr="")
            return SimpleNamespace(exit_code=0, stdout=sed_out, stderr="")
        env.run.side_effect = run_side
        return env

    def test_returns_region_on_match(self):
        env = self._env()
        result = _extract_region(env, "astropy/modeling/separable.py", "separability_matrix")
        self.assertEqual(result, "line1\nline2")

    def test_returns_empty_on_no_grep_match(self):
        env = self._env(grep_out="")
        result = _extract_region(env, "f.py", "nonexistent")
        self.assertEqual(result, "")

    def test_filepath_is_quoted(self):
        env = self._env()
        _extract_region(env, "path with spaces/file.py", "func")
        grep_cmd = env.run.call_args_list[0][0][0]
        self.assertIn("'path with spaces/file.py'", grep_cmd)

    def test_class_qualified_name_greps_method_only(self):
        """Bug A: a class-qualified name (Class.method) must grep `def method`, not
        `def Class.method` (which never matches a method definition → empty root cause)."""
        env = self._env()
        _extract_region(env, "f.py", "NDArithmeticMixin._arithmetic_mask")
        grep_cmd = env.run.call_args_list[0][0][0]
        self.assertIn("def _arithmetic_mask", grep_cmd)
        self.assertNotIn("NDArithmeticMixin._arithmetic_mask", grep_cmd)


if __name__ == "__main__":
    unittest.main()
