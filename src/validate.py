"""
Test-execution validation + ranking (no LLM calls — protects the cost score).

# PatchPilot §3.2 + App A eq.(1): run the issue-fixing tests and the functionality
# (regression) tests on each candidate, then rank.
#
# RANKING ALIGNED WITH OFFICIAL SWE-BENCH GRADING (swebench/harness/grading.py):
#   FULL (resolved): all FAIL_TO_PASS pass AND all PASS_TO_PASS pass
#   PARTIAL:         0 < FAIL_TO_PASS pass-ratio < 1 AND all PASS_TO_PASS pass
#   NO:              anything else — INCLUDING any broken PASS_TO_PASS
# Therefore preserving regressions (p2p == 1.0) is a HARD GATE, not a tiebreaker: a patch
# that breaks any regression can never be "resolved", so it must rank below every patch that
# breaks none — regardless of how many targets it fixes.  PatchPilot's "prioritize PoC over
# functionality" heuristic is overridden here by the grader's hard p2p requirement (our
# decision; see DEVELOPMENT_LOG §3a).
#
# Our mapping (DEVELOPMENT_LOG §3, §3a):
#   PoC / issue-fixing tests  -> FAIL_TO_PASS, materialized via the hidden test_patch
#   functionality tests       -> sampled PASS_TO_PASS (existing regression tests)
# Modes:
#   target_with_test_patch (default) — apply candidate + test_patch, run FAIL_TO_PASS +
#                                       sampled PASS_TO_PASS. Strongest signal.
#   regression_only (pure)          — run sampled PASS_TO_PASS only; never sees hidden tests.
#   poc                             — reproduction phase not implemented; falls back to
#                                       regression_only (documented deviation).
"""
from __future__ import annotations

import base64
import json
import random
import re
import shlex
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.config import AgentConfig, ValidationMode
from src.generate import revert
from utils.docker_env import DockerEnv

if TYPE_CHECKING:
    from utils.tasks import TaskContext

# pytest outcome keyword that counts as a pass.
_PASS = "PASSED"
# Task 8b: single source of truth for the feedback cap (generate.py no longer re-truncates).
# Raised to 6000 now that 8a/8c strip ANSI and drop the preamble, leaving room for the
# actual traceback. When regressions break, _validate_one stacks target + regression focuses.
_OUTPUT_CAP = 5000


@dataclass
class ValidationResult:
    """Per-candidate validation outcome; richer than a bare score so Task 6 can build
    a RefinementContext directly from it."""
    patch: str
    score: float
    applied: bool = True
    used_targets: bool = False                                   # did we run FAIL_TO_PASS?
    target_total: int = 0
    target_passed: int = 0
    failing_tests: list[str] = field(default_factory=list)       # FAIL_TO_PASS still failing
    broken_regressions: list[str] = field(default_factory=list)  # PASS_TO_PASS newly broken
    test_output: str = ""                                        # combined pytest output (capped)
    no_effect: bool = False                                      # Task 10: patch had no runtime effect vs baseline
    target_frames: list[str] = field(default_factory=list)       # Task 17: frames from target test only
    regression_frames: list[str] = field(default_factory=list)   # Task 17: frames from regression tests only
    test_source: str = ""                                        # Task 17: source code of failing test functions

    @property
    def regressions_ok(self) -> bool:
        return not self.broken_regressions

    @property
    def is_full(self) -> bool:
        """True iff this patch would be graded FULL/resolved by the official harness."""
        return (
            self.applied
            and self.used_targets
            and self.target_total > 0
            and self.target_passed == self.target_total
            and self.regressions_ok
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _as_list(value) -> list[str]:
    """FAIL_TO_PASS / PASS_TO_PASS may be a JSON-encoded string or a real list."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return [str(v) for v in parsed] if isinstance(parsed, list) else []
        except (ValueError, TypeError):
            return []
    return []


def _sample_regression(pass_to_pass: list[str], instance_id: str, size: int) -> list[str]:
    """Deterministic, spread-out sample of regression tests (seeded by instance_id)."""
    if len(pass_to_pass) <= size:
        return list(pass_to_pass)
    rng = random.Random(instance_id)
    return rng.sample(sorted(pass_to_pass), size)


def _apply_patch(env: DockerEnv, patch_text: str, name: str) -> bool:
    """Write a unified diff to a temp file (base64 argv — no stdin) and apply it.

    Tries `git apply`, then `git apply --3way`, then `patch -p1` (lenient fuzz — mirrors the
    official SWE-bench harness). Guarantees a trailing newline: a patch whose last line lacks
    one makes git report "corrupt patch at line N".
    """
    if not patch_text.strip():
        return False
    if not patch_text.endswith("\n"):
        patch_text += "\n"
    path = f"/tmp/{name}.patch"
    encoded = base64.b64encode(patch_text.encode("utf-8")).decode("ascii")
    write = (
        "python3 -c "
        "'import base64,sys; open(sys.argv[1],\"w\").write(base64.b64decode(sys.argv[2]).decode())' "
        + shlex.quote(path) + " " + shlex.quote(encoded)
    )
    if env.run(write, timeout=30).exit_code != 0:
        return False
    qpath = shlex.quote(path)
    # plain apply → 3-way (context drift) → patch -p1 (lenient fuzz, like the official harness).
    if env.run("git apply --whitespace=nowarn " + qpath, timeout=60).exit_code == 0:
        return True
    if env.run("git apply --3way --whitespace=nowarn " + qpath, timeout=60).exit_code == 0:
        return True
    return env.run("patch --batch --fuzz=5 -p1 -i " + qpath, timeout=60).exit_code == 0


def _parse_pytest(output: str) -> dict[str, str]:
    """Map node_id -> outcome from pytest `-rA` short-summary lines (e.g. 'PASSED x::y')."""
    statuses = {"PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS"}
    result: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) >= 2 and parts[0] in statuses:
            node_id = parts[1]
            result.setdefault(node_id, parts[0])
    return result


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_FRAME_RE = re.compile(r"^(\S+):(\d+): in (\S+)")
_SUMMARY_COUNT_RE = re.compile(
    r"(?P<count>\d+)\s+"
    r"(?P<kind>"
    r"failed|passed|error|errors|skipped|xfailed|xpassed|"
    r"deselected|warning|warnings"
    r")\b"
)


def _pytest_summary_counts(output: str) -> dict[str, int]:
    """Parse counts from pytest's final summary line.

    Handles quiet pytest output such as:
        1 passed in 0.12s
        1 failed, 2 passed in 0.34s
        1 failed, 1 error, 3 passed in 0.56s

    Also handles non-quiet lines wrapped in === ... ===.
    """
    for raw_line in reversed(output.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        line = line.strip("= ").strip()
        if " in " not in f" {line} ":
            continue
        if not _SUMMARY_COUNT_RE.search(line):
            continue
        counts: dict[str, int] = {}
        for match in _SUMMARY_COUNT_RE.finditer(line):
            kind = match.group("kind")
            count = int(match.group("count"))
            if kind == "error":
                kind = "errors"
            elif kind == "warning":
                kind = "warnings"
            counts[kind] = counts.get(kind, 0) + count
        return counts
    return {}


def _summary_nonpass_count(counts: dict[str, int]) -> int:
    """Count per-test outcomes that are not equivalent to PASSED."""
    return (
        counts.get("failed", 0)
        + counts.get("errors", 0)
        + counts.get("skipped", 0)
        + counts.get("xfailed", 0)
        + counts.get("xpassed", 0)
    )


def _backfill_missing_passes(
    status: dict[str, str],
    node_ids: list[str],
    counts: dict[str, int],
) -> None:
    """
    Backfill PASSED entries omitted by old pytest -rA output.

    Safe only when all declared non-pass outcomes are already represented
    in parsed status lines.
    """
    declared_passed = counts.get("passed", 0)
    if declared_passed <= 0:
        return

    explicit_passed = sum(
        1 for node_id in node_ids
        if status.get(node_id) == _PASS
    )

    missing_passes = declared_passed - explicit_passed
    if missing_passes <= 0:
        return

    explicit_nonpass = sum(
        1 for node_id in node_ids
        if node_id in status and status[node_id] != _PASS
    )

    declared_nonpass = _summary_nonpass_count(counts)

    # If pytest reported more non-pass outcomes than we parsed, at least one
    # unknown test may be ERROR/SKIPPED/XFAIL/XPASS. Do not guess.
    if declared_nonpass > explicit_nonpass:
        return

    unknown = [node_id for node_id in node_ids if node_id not in status]

    for node_id in unknown[:missing_passes]:
        status[node_id] = _PASS


def _parse_frames(output: str) -> list[str]:
    """Task 17: extract file:function:line frames from --tb=short traceback.

    # Format: `file:line: in function` followed by indented source line.
    # Returns list of "file::function:line" strings (compact for prompt rendering).
    # Pure additive anchor — does NOT constrain localization (RGFL: over-narrowing hurts).
    """
    frames: list[str] = []
    for line in output.splitlines():
        m = _FRAME_RE.match(line)
        if m:
            file, lineno, func = m.groups()
            frames.append(f"{file}::{func}:{lineno}")
    return frames


def extract_test_source(test_patch: str, test_names: list[str]) -> str:
    """Extract the source code of failing test functions from the test_patch.

    # Parses the unified diff to find added lines in test functions.
    # Returns the extracted test source code (without diff markers).
    # Only extracts functions whose names appear in test_names.
    """
    if not test_patch or not test_names:
        return ""

    # Extract function/method names from test_names
    # Handles multiple formats:
    # - pytest: "path::test_func" or "path::TestClass::method"
    # - Django: "test_func (module.ClassName)"
    # - Parameterized: "test_func[param1-param2]"
    target_names = set()
    for name in test_names:
        # Strip parameterized part: "test_func[param]" -> "test_func"
        if "[" in name:
            name = name.split("[")[0]
        
        # Handle Django format: "test_func (module.ClassName)"
        if " (" in name:
            name = name.split(" (")[0]
        
        # Handle pytest format: take the last part after ::
        if "::" in name:
            parts = name.split("::")
            name = parts[-1]
        
        target_names.add(name.strip())

    # Parse the diff to extract added lines, tracking which are added vs context
    # Group lines by hunk function
    hunk_groups = []  # list of (hunk_func, lines)
    current_hunk_func = None
    current_hunk_lines = []
    in_hunk = False
    
    for line in test_patch.splitlines():
        if line.startswith("@@"):
            # Save previous hunk if any
            if current_hunk_lines:
                hunk_groups.append((current_hunk_func, current_hunk_lines))
            
            in_hunk = True
            # Extract function name from hunk header: @@ ... @@ def func_name():
            hunk_match = re.search(r"@@.*@@\s*(?:async\s+def|def|class)\s+(test_\w+|Test\w+)", line)
            current_hunk_func = hunk_match.group(1) if hunk_match else None
            current_hunk_lines = []
            continue
        if in_hunk:
            if line.startswith("+") and not line.startswith("+++"):
                current_hunk_lines.append(line[1:])  # strip the + marker
            elif line.startswith(" ") and not line.startswith("---"):
                current_hunk_lines.append(line[1:])  # context line, strip space
            elif line.startswith("-"):
                continue  # skip removed lines
            elif line.startswith("\\"):
                continue  # skip "\ No newline at end of file"
            else:
                in_hunk = False  # end of hunk
    
    # Save last hunk
    if current_hunk_lines:
        hunk_groups.append((current_hunk_func, current_hunk_lines))

    # Extract test function bodies from each hunk
    result = []
    
    for hunk_func, lines in hunk_groups:
        current_func = []
        current_func_name = None
        indent_level = None

        # If hunk has a function in header and it's a target, start that function context
        if hunk_func and hunk_func in target_names:
            current_func_name = hunk_func
            current_func = [f"def {hunk_func}():"]  # placeholder
            indent_level = 0

        for line in lines:
            # Check if this is a test function definition (allow leading whitespace for methods)
            match = re.match(r"^\s*(def|class)\s+(test_\w+|Test\w+)", line)
            if match:
                # Save previous function if any and if it's in target_names
                if current_func and current_func_name in target_names:
                    result.append("\n".join(current_func))
                current_func = [line]
                current_func_name = match.group(2)
                indent_level = len(line) - len(line.lstrip())
            elif current_func:
                # Check if we're still in the function (same or deeper indent, or empty line)
                if line.strip() == "":
                    current_func.append(line)
                elif line.startswith(" " * (indent_level + 1)) or line.startswith("\t"):
                    current_func.append(line)
                else:
                    # End of function
                    # Only include if it's in target_names
                    if current_func_name in target_names:
                        result.append("\n".join(current_func))
                    current_func = []
                    current_func_name = None

        # Don't forget the last function in this hunk
        if current_func and current_func_name:
            if current_func_name in target_names:
                result.append("\n".join(current_func))

    # Fallback: if no test functions found, return assert lines (modified tests)
    if not result:
        # Catch bare assert, self.assert* methods, pytest.raises, and fnmatch_lines
        assert_lines = []
        for hunk_func, lines in hunk_groups:
            for line in lines:
                if line.strip().startswith("assert") or \
                   re.search(r'self\.assert\w+\(', line) or \
                   re.search(r'\braises\(', line) or \
                   re.search(r'\bfnmatch_lines\(', line):
                    assert_lines.append(line)
        return "\n".join(assert_lines) if assert_lines else ""

    return "\n\n".join(result)


def _focus(output: str) -> str:
    """Task 8c: keep the FAILURES/ERRORS traceback + summary, not the platform/plugin preamble.

    The output cap is taken from the start, so without this the preamble eats the budget
    before the actual error is reached. Collection/import failures appear under "= ERRORS ="
    rather than "= FAILURES =", so key on whichever comes first. If neither, return as-is.
    """
    markers = [output.find("= FAILURES ="), output.find("= ERRORS =")]
    starts = [i for i in markers if i != -1]
    text = output[min(starts):] if starts else output
    return text[:_OUTPUT_CAP]


# Task 10: volatile tokens that vary run-to-run even for identical code — must be ignored when
# deciding whether a patch produced the SAME output as the no-fix baseline (i.e. a no-op).
_NO_EFFECT_OUTPUT_CAP = 20000

_VOLATILE_RE = re.compile(
    r"0x[0-9a-fA-F]+"            # memory addresses
    r"|/tmp/[^\s'\"]+"          # temp file paths
    r"|pytest-of-\w+|pytest-\d+"  # pytest tmp dir names
    r"|in \d+\.\d+s"            # run durations
)

_TRACE_LINE_RE = re.compile(r"(^|\s)(\S+\.py):\d+:\s+in\s+(\S+)")


def _focus_for_no_effect(output: str) -> str:
    """Focus on FAILURES/ERRORS with larger cap for no-effect comparison."""
    markers = [output.find("= FAILURES ="), output.find("= ERRORS =")]
    starts = [i for i in markers if i != -1]
    text = output[min(starts):] if starts else output
    return text[:_NO_EFFECT_OUTPUT_CAP]


def _norm(text: str) -> str:
    """Normalize pytest output for no-op comparison (drop volatile tokens, normalize line numbers, collapse ws)."""
    text = _VOLATILE_RE.sub("", text)
    text = _TRACE_LINE_RE.sub(r"\1\2:<LINE>: in \3", text)
    return re.sub(r"\s+", " ", text).strip()


def is_no_effect(focused_output: str, baseline_output: str) -> bool:
    """Task 10: True when a patch's (focused) target output matches the no-fix baseline,
    i.e. the edit had no runtime effect on the failing tests (e.g. 6938's reassignment)."""
    return bool(baseline_output) and _norm(focused_output) == _norm(baseline_output)


def _run_tests(env: DockerEnv, node_ids: list[str], timeout: int) -> tuple[dict[str, str], str]:
    """Run pytest on the given node ids; return (node->status map, raw output)."""
    if not node_ids:
        return {}, ""
    quoted = " ".join(shlex.quote(n) for n in node_ids)
    cmd = f"python -m pytest -rA --tb=short --color=no -p no:cacheprovider -q {quoted}"
    res = env.run(cmd, timeout=timeout)
    output = (res.stdout or "") + (("\n" + res.stderr) if res.stderr else "")
    output = _ANSI_RE.sub("", output)
    status = _parse_pytest(output)
    counts = _pytest_summary_counts(output)

    _backfill_missing_passes(status, node_ids, counts)

    return status, output


# ── core ──────────────────────────────────────────────────────────────────────

def _score(used_targets: bool, target_total: int, target_passed: int,
           regression_total: int, broken: int) -> float:
    """Composite score matching official grading (see module docstring).

    Bands (target mode):
        no regression broken : 1.0 + f2p_ratio    -> [1.0, 2.0], FULL == 2.0
        regression broken    : f2p_ratio - eps*broken -> strictly < 1.0  (hard gate)
    Fallback (no targets)   : fraction of regressions preserved -> [0, 1]
    Not applied (caller)    : -1.0
    """
    if used_targets:
        f2p_ratio = target_passed / target_total  # target_total > 0 guaranteed when used_targets
        if broken == 0:
            return 1.0 + f2p_ratio
        # Broken regressions => can never be FULL: keep strictly below the clean band [1.0, 2.0].
        # 0.001 * broken stays well under 1.0 (broken <= regression_sample_size), so a clean
        # patch with zero targets fixed (1.0) still outranks any regression-breaker.
        return f2p_ratio - 0.001 * broken
    # regression_only / poc fallback: fraction of sampled regressions preserved.
    return (regression_total - broken) / max(1, regression_total)


def _validate_one(
    patch: str,
    context: "TaskContext",
    env: DockerEnv,
    config: AgentConfig,
    test_patch: str,
    baseline_output: str = "",
) -> ValidationResult:
    """Validate a single candidate. Always reverts the working tree before returning."""
    mode = config.validation_mode
    targets = _as_list(context.fail_to_pass)
    regression = _sample_regression(
        _as_list(context.pass_to_pass), context.instance_id, config.regression_sample_size
    )

    use_targets = (
        mode == ValidationMode.TARGET_WITH_TEST_PATCH and bool(test_patch) and bool(targets)
    )

    revert(env)  # start from a clean base_commit

    if not _apply_patch(env, patch, "candidate"):
        revert(env)
        return ValidationResult(patch=patch, score=-1.0, applied=False)

    # Materialize the hidden issue-fixing tests (target mode only).
    if use_targets and not _apply_patch(env, test_patch, "test"):
        use_targets = False  # couldn't materialize; degrade to regression signal

    # Task 8d: run targets and regressions in SEPARATE pytest calls so the FAIL_TO_PASS
    # traceback is focused, not buried among up to 10 regression tests' output.
    t_status, t_output = _run_tests(env, targets, config.test_timeout) if use_targets else ({}, "")
    r_status, r_output = _run_tests(env, regression, config.test_timeout)
    revert(env)
    status = {**r_status, **t_status}

    target_passed = [t for t in targets if status.get(t) == _PASS] if use_targets else []
    failing = [t for t in targets if status.get(t) != _PASS] if use_targets else []
    broken = [r for r in regression if status.get(r) != _PASS]

    score = _score(
        used_targets=use_targets,
        target_total=len(targets),
        target_passed=len(target_passed),
        regression_total=len(regression),
        broken=len(broken),
    )

    # Prefer the target traceback for refinement; include regression output when
    # regressions are broken (so the model can see what it broke).
    if broken:
        feedback = _focus(t_output) + "\n" + _focus(r_output)
    elif t_output:
        feedback = _focus(t_output)
    else:
        feedback = _focus(r_output)

    # Task 10: flag a no-op — patch applied but the target tests behave exactly as with no fix.
    no_effect = use_targets and is_no_effect(
        _focus_for_no_effect(t_output),
        baseline_output,
    )

    # Task 17: extract frames separately for target and regression tests
    target_frames = _parse_frames(_focus(t_output)) if use_targets else []
    regression_frames = _parse_frames(_focus(r_output)) if broken else []

    # Task 17: extract test source from test_patch for failing tests
    test_source = extract_test_source(test_patch, failing) if use_targets and failing else ""

    return ValidationResult(
        patch=patch,
        score=score,
        applied=True,
        used_targets=use_targets,
        target_total=len(targets) if use_targets else 0,
        target_passed=len(target_passed),
        failing_tests=failing,
        broken_regressions=broken,
        test_output=feedback,
        no_effect=no_effect,
        target_frames=target_frames,
        regression_frames=regression_frames,
        test_source=test_source,
    )


def validate(
    candidates: list[str],
    context: "TaskContext",
    env: DockerEnv,
    config: AgentConfig,
    test_patch: str = "",
    baseline_output: str = "",
) -> list[ValidationResult]:
    """Validate and rank candidates best-first (no LLM calls).

    # Ranking matches official grading: regression-preservation is a hard gate (see module
    # docstring). Tiebreak (Task 11): normally prefer the smaller patch (minimal correct
    # patch, Req 9.2); but when NO candidate fixes any target, prefer the LARGER attempt so
    # refinement seeds from the richest partial fix rather than a do-nothing stub (the 14182
    # failure mode). Score remains primary, so the hard gate is untouched.
    # test_patch: hidden issue-fixing tests (loaded via src/dataset.py); "" -> regression signal.
    # baseline_output: no-fix target output for no-op detection (Task 10); "" disables it.
    """

    results = [
        _validate_one(p, context, env, config, test_patch, baseline_output)
        for p in candidates if p and p.strip()
    ]
    any_progress = any(r.used_targets and r.target_passed > 0 for r in results)
    results.sort(key=lambda r: (-r.score, r.no_effect, len(r.patch) if any_progress else -len(r.patch)))
    return results


def baseline_target_output(
    context: "TaskContext", env: DockerEnv, config: AgentConfig, test_patch: str
) -> str:
    """Task 10: capture the FAIL_TO_PASS output on the UNPATCHED tree (test_patch applied only
    to materialize the target tests). Used to detect no-op patches in refinement. Returns ""
    when targets can't be materialized (regression-only mode)."""
    targets = _as_list(context.fail_to_pass)
    if not (test_patch and targets
            and config.validation_mode == ValidationMode.TARGET_WITH_TEST_PATCH):
        return ""
    revert(env)
    if not _apply_patch(env, test_patch, "test"):
        revert(env)
        return ""
    _status, output = _run_tests(env, targets, config.test_timeout)
    revert(env)
    return _focus(output)
