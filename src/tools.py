"""
Shared tool execution for LLM agents.

Provides tool parsing and execution for localization and refinement analysis.
Tools: search_func_def, search_class_def, search_string, run_python.
"""
from __future__ import annotations

import ast
import re
import shlex

from utils.docker_env import DockerEnv

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
OUTPUT_CAP = 2000
SEARCH_STRING_MAX = 500


def extract_call_arg(text: str, func_name: str) -> list[str]:
    """Extract raw argument strings from function calls, handling strings/parens/comments.

    Quote-aware tokenizer that correctly handles:
    - Parentheses inside strings: run_python("print(')')")
    - Multi-line strings: run_python('''code''')
    - Escaped quotes: run_python("print(\"hi\")")
    - Comments: run_python("# )\nprint('ok')")
    """
    pattern = re.compile(rf"{re.escape(func_name)}\(")
    args: list[str] = []

    for match in pattern.finditer(text):
        i = match.end()
        start = i
        depth = 1
        quote: str | None = None
        triple = False
        escape = False
        in_comment = False

        while i < len(text):
            ch = text[i]
            nxt3 = text[i:i+3]

            if in_comment:
                if ch == "\n":
                    in_comment = False
                i += 1
                continue

            if quote:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif triple and nxt3 == quote * 3:
                    quote = None
                    triple = False
                    i += 2
                elif not triple and ch == quote:
                    quote = None
                i += 1
                continue

            if ch == "#":
                in_comment = True
            elif nxt3 in ("'''", '"""'):
                quote = ch
                triple = True
                i += 2
            elif ch in ("'", '"'):
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    args.append(text[start:i].strip())
                    break

            i += 1

    return args


def parse_string_arg(arg_src: str) -> str | None:
    """Parse a Python string literal using ast.literal_eval for safety."""
    try:
        value = ast.literal_eval(arg_src)
    except (ValueError, SyntaxError, TypeError):
        return None
    return value if isinstance(value, str) else None


def run_tools(model_output: str, env: DockerEnv) -> tuple[str, bool]:
    """Execute tool calls found in model output; return (results, had_tool_calls).

    Tools: search_func_def, search_class_def, search_string, run_python.
    All tool args are parsed with extract_call_arg() for quote-aware extraction
    and parse_string_arg() for safe string literal parsing.
    Returns whether tool calls were found so the caller can send a follow-up
    even when tools returned empty results.
    """
    results: list[str] = []
    had_calls = False

    has_run_python = "run_python(" in model_output

    for arg_src in extract_call_arg(model_output, "search_func_def"):
        had_calls = True
        arg = parse_string_arg(arg_src)
        if arg is None:
            arg = arg_src.strip().strip("'\"")

        if not IDENTIFIER.fullmatch(arg):
            results.append(f"[search_func_def({arg!r})]\nInvalid identifier: must match [A-Za-z_][A-Za-z0-9_]*")
            continue

        pattern = rf"^[[:space:]]*(async[[:space:]]+)?def[[:space:]]+{arg}[[:space:]]*\("
        cmd = f"grep -rnE --include='*.py' {shlex.quote(pattern)} . | head -5"
        r = env.run(cmd, timeout=15)
        if r.stdout.strip():
            results.append(f"[search_func_def({arg!r})]\n{r.stdout.strip()}")

    for arg_src in extract_call_arg(model_output, "search_class_def"):
        had_calls = True
        arg = parse_string_arg(arg_src)
        if arg is None:
            arg = arg_src.strip().strip("'\"")

        if not IDENTIFIER.fullmatch(arg):
            results.append(f"[search_class_def({arg!r})]\nInvalid identifier: must match [A-Za-z_][A-Za-z0-9_]*")
            continue

        pattern = rf"^[[:space:]]*class[[:space:]]+{arg}([[:space:]]*[\(:]|$)"
        cmd = f"grep -rnE --include='*.py' {shlex.quote(pattern)} . | head -5"
        r = env.run(cmd, timeout=15)
        if r.stdout.strip():
            results.append(f"[search_class_def({arg!r})]\n{r.stdout.strip()}")

    for arg_src in extract_call_arg(model_output, "search_string"):
        had_calls = True
        arg = parse_string_arg(arg_src)
        if arg is None:
            results.append("[search_string]\nInvalid argument: expected a Python string literal")
            continue
        if not arg:
            results.append("[search_string]\nInvalid argument: empty search string")
            continue
        if len(arg) > SEARCH_STRING_MAX:
            results.append("[search_string]\nInvalid argument: search string too long")
            continue

        cmd = f"grep -rlnF --include='*.py' {shlex.quote(arg)} . | head -5"
        r = env.run(cmd, timeout=15)
        if r.stdout.strip():
            results.append(f"[search_string({arg!r})]\n{r.stdout.strip()}")

    extracted_args = extract_call_arg(model_output, "run_python")

    if has_run_python and not extracted_args:
        had_calls = True
        results.append("[run_python]\nMalformed call: missing closing parenthesis or invalid string literal")
    else:
        for arg_src in extracted_args:
            had_calls = True
            code = parse_string_arg(arg_src)

            if code is None:
                results.append("[run_python]\nInvalid argument: expected a Python string literal")
                continue

            cmd = f"python3 -I -c {shlex.quote(code)}"
            r = env.run(cmd, timeout=15)

            parts: list[str] = []
            if r.stdout.strip():
                parts.append(r.stdout.strip())
            if r.stderr.strip():
                parts.append("[stderr]\n" + r.stderr.strip())
            if r.exit_code != 0:
                parts.append(f"[exit code: {r.exit_code}]")

            output = "\n".join(parts) if parts else ""

            if len(output) > OUTPUT_CAP:
                output = output[:OUTPUT_CAP] + f"\n... (truncated, {len(output)} chars total)"

            if output:
                results.append(f"[run_python({code!r})]\n{output}")

    return "\n\n".join(results), had_calls
