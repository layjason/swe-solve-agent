"""
SEARCH/REPLACE edit format parser, deterministic applier, and revert helper.

PatchPilot App B (following Aider diff format):
- LLM emits edits as SEARCH/REPLACE blocks; we convert them to git diff ourselves.
- Per-edit: Python ast syntax check + Flake8 lint check in the Docker env.
- If either check fails, the edit is reverted and marked failed.

Aider diff format (aider.chat/docs/more/edit-formats.html):
    path/to/file.py
    <<<<<<< SEARCH
    <exact lines to find>
    =======
    <replacement lines>
    >>>>>>> REPLACE

Deviations from paper (our choices):
- Checks run in the Docker env (code lives in container, not locally).
- File writes use base64 argv — env.run has no stdin support.
- Revert uses git checkout -- . + git clean -fd.
"""
from __future__ import annotations

import base64
import re
import shlex
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from utils.docker_env import DockerEnv

if TYPE_CHECKING:
    from src.config import AgentConfig, RefinementContext
    from utils.models import ModelClient
    from utils.tasks import TaskContext

from src.config import budget_exceeded  # runtime import (no circular dependency)


_TEST_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)


def extract_test_files(test_patch: str) -> set[str]:
    """Extract file paths modified by the test_patch (ground-truth tests)."""
    if not test_patch:
        return set()
    return set(_TEST_FILE_RE.findall(test_patch))


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class Edit:
    filepath: str
    search: str
    replace: str


@dataclass
class ApplyResult:
    applied: list[str] = field(default_factory=list)   # edit keys that succeeded
    failed: list[str] = field(default_factory=list)    # edit keys that failed
    diff: str = ""                                      # unified diff from git diff


# ── session setup ─────────────────────────────────────────────────────────────

def ensure_tools(env: DockerEnv) -> None:
    """Install flake8 into the container once at session start.

    Called from solve_task before the pipeline begins so lint checks are
    unconditional. One-time cost: ~1-2s, no LLM tokens.
    """
    env.run("pip install --quiet flake8", timeout=60)


# ── parser ────────────────────────────────────────────────────────────────────

_BLOCK_RE = re.compile(
    r"^(?P<path>[^\n]+?\.py)\s*\n"   # filepath ending in .py
    r"(?:```[^\n]*\n)?"              # optional opening fence
    r"<{7} SEARCH\n"
    r"(?P<search>.*?)"
    r"={7}\n"
    r"(?P<replace>.*?)"
    r">{7} REPLACE\n?"
    r"(?:```)?",                     # optional closing fence
    re.DOTALL | re.MULTILINE,
)


def parse_edits(text: str) -> list[Edit]:
    """Extract all SEARCH/REPLACE blocks from model output.

    # PatchPilot App B + Aider diff format.
    Returns [] if no well-formed block found (Req 4.2).
    """
    return [
        Edit(filepath=m.group("path").strip(), search=m.group("search"), replace=m.group("replace"))
        for m in _BLOCK_RE.finditer(text)
    ]


# ── in-env checks ─────────────────────────────────────────────────────────────

def _syntax_ok(env: DockerEnv, filepath: str) -> bool:
    """ast syntax check inside the container. # PatchPilot App B."""
    cmd = "python3 -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' " + shlex.quote(filepath)
    return env.run(cmd, timeout=15).exit_code == 0


def _lint_ok(env: DockerEnv, filepath: str) -> bool:
    """Flake8 lint check inside the container. # PatchPilot App B."""
    return env.run(
        f"flake8 --max-line-length=120 --select=E9,F {shlex.quote(filepath)}", timeout=15
    ).exit_code == 0


# ── file write ────────────────────────────────────────────────────────────────

def _write_file(env: DockerEnv, filepath: str, content: str) -> bool:
    """Write content to filepath via base64 argv (env.run has no stdin).

    base64 chars [A-Za-z0-9+/=] are shell-safe. filepath is shlex.quote'd
    as a separate argv argument — never inside the -c string.
    """
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    cmd = (
        "python3 -c "
        "'import base64,sys; open(sys.argv[1],\"w\").write(base64.b64decode(sys.argv[2]).decode())' "
        + shlex.quote(filepath) + " " + shlex.quote(encoded)
    )
    return env.run(cmd, timeout=30).exit_code == 0


# ── applier ───────────────────────────────────────────────────────────────────

def _resolve_path(env: DockerEnv, filepath: str) -> str:
    """Resolve a possibly-bare filepath to a tracked repo-relative path.

    The model sometimes emits a basename ('fitsrec.py') instead of the full path
    ('astropy/io/fits/fitsrec.py') in a SEARCH/REPLACE block (Bug B). If the path as given
    doesn't exist, match tracked files by suffix and return the unique/best match; otherwise
    return the original (the caller then records a read error rather than editing a wrong file).
    """
    if env.run("test -f " + shlex.quote(filepath), timeout=10).exit_code == 0:
        return filepath
    base = filepath.rsplit("/", 1)[-1]
    r = env.run(
        "git ls-files | grep -E " + shlex.quote(f"(^|/){re.escape(base)}$") + " | head -20",
        timeout=15,
    )
    matches = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    if not matches:
        return filepath
    # Partial path given (contains '/'): prefer a tracked file that ends with it.
    if "/" in filepath:
        for m in matches:
            if m == filepath or m.endswith("/" + filepath):
                return m
    # Bare basename: only resolve when unambiguous, else leave as-is (don't guess).
    return matches[0] if len(matches) == 1 else filepath


def apply_edits(edits: list[Edit], env: DockerEnv, test_files: set[str] | None = None) -> ApplyResult:
    """Apply edits deterministically; return unified diff.

    For each edit:
      1. Exact-match search text in file; replace once.
      2. Write back via base64 argv.
      3. ast syntax check + Flake8; revert single file on failure.

    # PatchPilot App B. Req 5.1 (exact-match), 5.2 (no-match → skip), 5.3 (git diff).
    test_files: set of file paths from test_patch that must not be modified.
    """
    result = ApplyResult()
    blocked_files = test_files or set()

    for i, edit in enumerate(edits):
        filepath = _resolve_path(env, edit.filepath)  # normalize bare filenames (Bug B)
        key = f"{filepath}[{i}]"
        qfp = shlex.quote(filepath)

        if filepath in blocked_files:
            result.failed.append(f"{key}: blocked (test file from test_patch)")
            continue

        try:
            content = env.read_file(filepath, max_chars=200_000)
        except Exception as e:
            result.failed.append(f"{key}: read error: {e}")
            continue

        # Exact-match replace (Req 5.2: no-match → skip, no change to file)
        if edit.search not in content:
            result.failed.append(f"{key}: search text not found")
            continue

        new_content = content.replace(edit.search, edit.replace, 1)

        if not _write_file(env, filepath, new_content):
            result.failed.append(f"{key}: write failed")
            continue

        # Syntax check (PatchPilot App B)
        if not _syntax_ok(env, filepath):
            env.run("git checkout -- " + qfp, timeout=15)
            result.failed.append(f"{key}: syntax check failed")
            continue

        # Lint check (PatchPilot App B)
        if not _lint_ok(env, filepath):
            env.run("git checkout -- " + qfp, timeout=15)
            result.failed.append(f"{key}: lint check failed")
            continue

        result.applied.append(key)

    # Unified diff (Req 5.3). Capture verbatim: do NOT strip — stripping removes the trailing
    # newline and any final blank/whitespace context line, which corrupts the hunk and makes
    # `git apply` reject the patch ("corrupt patch at line N"). git diff already emits a valid
    # patch ending in a newline; only collapse to "" when there is genuinely no change.
    raw = env.run("git diff", timeout=30).stdout
    result.diff = raw if raw.strip() else ""
    return result


def revert(env: DockerEnv) -> None:
    """Revert working tree to base_commit state between candidates.

    # PatchPilot App B: revert before next candidate. Req 5.4.
    """
    env.run("git checkout -- .", timeout=30)
    env.run("git clean -fd", timeout=30)


# ── plan-then-generate ────────────────────────────────────────────────────────
# PatchPilot §3.4 + App D + repair.py (source-verified at github.com/ucsb-mlsec/PatchPilot):
# First batch = 3 greedy plans with distinct strategies: standard (planning_prompt),
# comprehensive (planning_prompt_general), minimal (planning_prompt_minimal).
# Paper text says "4 patches" but source confirms 3 distinct prompt types as first batch.
# Diversity comes from prompt TYPE not temperature (paper §3.4: "simply increasing the
# temperature still results in similar patches").
# Subsequent batches: standard only (caching + cost reduction).

_PLAN_SYSTEM = (
    "You are an experienced software maintainer responsible for analyzing and fixing "
    "repository issues. Your role is to:\n"
    "1. Thoroughly analyze bugs to identify underlying root causes.\n"
    "2. Provide clear, actionable repair plans with precise code modifications.\n\n"
    "Format your repair plans using:\n"
    "- <STEP> and </STEP> tags for each modification step.\n"
    "- <Actions to be Taken> and </Actions to be Taken> tags for specific actions.\n"
    "- Maximum {step_cap} steps, with each step containing exactly one code modification.\n"
    "- Only include steps that require code changes. Do not write code in the plan."
)


def _get_plan_system(n_sites: int) -> str:
    """Task 13: scale step cap to localized site count (gold 14182 = 4 edits / 2 files)."""
    step_cap = max(3, n_sites * 2)
    return _PLAN_SYSTEM.format(step_cap=step_cap)

_GEN_SYSTEM = (
    "You are an experienced software maintainer. Given a repair plan, generate the code edits "
    "to implement it. Use ONLY the Aider SEARCH/REPLACE format for every edit:\n\n"
    "path/to/file.py\n"
    "<<<<<<< SEARCH\n"
    "<exact lines to replace>\n"
    "=======\n"
    "<replacement lines>\n"
    ">>>>>>> REPLACE\n\n"
    "Rules:\n"
    "- One SEARCH/REPLACE block per file change.\n"
    "- SEARCH must match the file content exactly (whitespace included).\n"
    "- Use the exact repository-relative file path shown in the FILE headers "
    "(e.g. astropy/io/fits/fitsrec.py), never a bare filename.\n"
    "- No explanation outside the SEARCH/REPLACE blocks."
)

# PatchPilot App D + repair.py: three prompt strategies for first-batch diversity.
_PLAN_SUFFIXES = [
    "",   # standard — no extra instruction (planning_prompt)
    (     # comprehensive — PatchPilot App D §D.3 / planning_prompt_general
        "\n\nProduce a COMPREHENSIVE patch that not only fixes this issue but also "
        "prevents similar issues in related code paths. Make the fix as general as possible."
    ),
    (     # minimal — PatchPilot App D §D.2 / planning_prompt_minimal
        "\n\nProduce the MINIMAL patch — the smallest possible code change that fixes "
        "the issue without altering unrelated behaviour."
    ),
]


def _get_plan_suffixes(n_files: int) -> list[str]:
    """Task 13: only constrain to one file when localization found a single file."""
    suffixes = list(_PLAN_SUFFIXES)
    if n_files == 1:
        suffixes[2] += " Modify only one file."
    return suffixes

_MIN_PLAN_CHARS = 30  # guard: real plans cite a file+action (>77 chars); garbage is <20

_PREVIOUS_PATCH_CAP = 12000


def _summarize_patch_for_prompt(patch: str, max_chars: int = _PREVIOUS_PATCH_CAP) -> str:
    """Truncate patch at file boundaries to avoid breaking SEARCH/REPLACE blocks."""
    if len(patch) <= max_chars:
        return patch

    # Prefer keeping whole file-diff sections.
    sections = []
    current = []

    for line in patch.splitlines():
        if line.startswith("diff --git ") and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current))

    kept = []
    total = 0
    for section in sections:
        projected = total + len(section) + 2
        if projected > max_chars:
            break
        kept.append(section)
        total = projected

    if kept:
        return (
            "\n\n".join(kept)
            + "\n\n... [previous patch truncated at file boundary]"
        )

    # Fallback for a single huge file diff.
    lines = patch.splitlines()
    out = []
    total = 0
    for line in lines:
        if total + len(line) + 1 > max_chars:
            break
        out.append(line)
        total += len(line) + 1

    return (
        "\n".join(out)
        + "\n\n... [previous patch truncated; diff may be incomplete]"
    )


def _build_plan_user(
    context: "TaskContext",
    localization: dict,
    suffix: str,
    refinement: "RefinementContext | None",
    test_files: frozenset[str] = frozenset(),
) -> str:
    """Build plan prompt with structured refinement feedback if present.

    # PatchPilot repair.py: feedback = current_patch + test_output +
    # broken_regression_output. Structured with clear delimiters so the model
    # can reason about each section separately (PatchPilot §3.5).
    """
    parts: list[str] = []

    if refinement:
        is_escalated = bool(refinement.analysis_note)

        # Previous patch first — so the model sees what caused the current failures.
        if refinement.current_patch:
            parts.append(
                "--- BEGIN PREVIOUS PATCH ---\n"
                + _summarize_patch_for_prompt(refinement.current_patch, 6000)
                + "\n--- END PREVIOUS PATCH ---"
            )


        # Explicit failing target test names.
        if refinement.failing_tests:
            parts.append(
                "--- BEGIN FAILING TARGET TESTS ---\n"
                + "\n".join(refinement.failing_tests[:20])
                + "\n--- END FAILING TARGET TESTS ---"
            )

        # Target failure feedback.
        failing_test_parts = []

        # Test source is expensive/noisy. Include only after escalation.
        if is_escalated and refinement.test_source:
            failing_test_parts.append(
                "Test source:\n" + refinement.test_source[:2000]
            )

        if refinement.target_frames:
            localized_files = set(localization.get("files", []))
            visible_frames = [
                f.rsplit(":", 1)[0]  # strip :line → file::function
                for f in refinement.target_frames
                if f.split("::")[0] in localized_files
            ]
            if visible_frames:
                failing_test_parts.append(
                    "Target execution path:\n" + "\n".join(visible_frames[:15])
                )

        if refinement.test_output.strip():
            log_cap = 3000 if is_escalated else 1800
            failing_test_parts.append(
                "Failure logs:\n" + refinement.test_output[:log_cap]
            )

        if failing_test_parts:
            parts.append(
                "--- BEGIN FAILING TARGET FEEDBACK ---\n"
                + "\n\n".join(failing_test_parts)
                + "\n--- END FAILING TARGET FEEDBACK ---"
            )

        # Broken regressions are important because they mean the patch harmed existing behavior.
        if refinement.regression_frames or refinement.broken_regressions:
            regression_parts = []

            if refinement.broken_regressions:
                regression_parts.append(
                    "Broken regression tests:\n"
                    + "\n".join(refinement.broken_regressions[:20])
                )

            if refinement.regression_frames:
                frame_cap = 15 if is_escalated else 8
                frames_without_lines = [
                    f.rsplit(":", 1)[0]
                    for f in refinement.regression_frames[:frame_cap]
                ]
                regression_parts.append(
                    "Regression execution path:\n" + "\n".join(frames_without_lines)
                )

            parts.append(
                "--- BEGIN BROKEN REGRESSIONS ---\n"
                + "\n\n".join(regression_parts)
                + "\n--- END BROKEN REGRESSIONS ---"
            )

        # Previous attempts help prevent repeating the same failed mechanism.
        if refinement.ledger_text:
            parts.append(
                "--- BEGIN PREVIOUS ATTEMPTS ---\n"
                + refinement.ledger_text
                + "\n--- END PREVIOUS ATTEMPTS ---"
            )

        # Escalation-only analysis.
        if refinement.analysis_note:
            parts.append(
                "--- BEGIN REFINEMENT ANALYSIS ---\n"
                + refinement.analysis_note
                + "\n--- END REFINEMENT ANALYSIS ---"
            )

        # Final instruction.
        if refinement.note and not refinement.analysis_note:
            parts.append(refinement.note)
        elif refinement.analysis_note:
            parts.append(
                "Use the refinement analysis above as guidance, but treat the validation logs "
                "as the source of truth. Avoid repeating mechanisms listed in PREVIOUS ATTEMPTS."
            )
        else:
            parts.append(
                "This patch did not fully fix the issue. Make a focused refinement based on "
                "the failing target feedback above. Preserve behavior for passing/regression tests."
            )

    parts.append("--- BEGIN ISSUE ---\n" + context.problem_statement + "\n--- END ISSUE ---")

    root_cause = localization.get("root_cause", "")
    if root_cause:
        parts.append("--- BEGIN FILE ---\n" + root_cause + "\n--- END FILE ---")

    if test_files:
        parts.append(
            "IMPORTANT: Do NOT plan any modifications to the following test files — "
            "they are read-only ground truth used for evaluation. Only fix source code:\n"
            + ", ".join(sorted(test_files))
        )

    parts.append("Propose a plan to fix this issue." + suffix)
    return "\n\n".join(parts)


def _build_gen_user(
    context: "TaskContext",
    plan: str,
    localization: dict,
    refinement: "RefinementContext | None" = None,
    test_files: frozenset[str] = frozenset(),
) -> str:
    parts = ["--- BEGIN ISSUE ---\n" + context.problem_statement + "\n--- END ISSUE ---"]

    if refinement and refinement.analysis_note:
        parts.append(
            "--- BEGIN REFINEMENT ANALYSIS ---\n"
            + refinement.analysis_note
            + "\n--- END REFINEMENT ANALYSIS ---"
        )

    root_cause = localization.get("root_cause", "")
    if root_cause:
        parts.append("--- BEGIN FILE ---\n" + root_cause + "\n--- END FILE ---")

    parts.append("--- BEGIN PLAN ---\n" + plan + "\n--- END PLAN ---")

    if test_files:
        parts.append(
            "IMPORTANT: Do NOT include any SEARCH/REPLACE blocks for these test files "
            "(they are read-only ground truth):\n"
            + ", ".join(sorted(test_files))
        )

    parts.append("Generate the SEARCH/REPLACE edits that implement this plan.")
    return "\n\n".join(parts)


def generate_candidates(
    context: "TaskContext",
    env: DockerEnv,
    model: "ModelClient",
    config: "AgentConfig",
    localization: dict,
    refinement: "RefinementContext | None" = None,
    first_batch: bool = True,
    baseline_tokens: int = 0,
    force_diverse: bool = False,
    test_patch: str = "",
) -> list[str]:
    """Generate N candidate patches using plan-then-generate.

    baseline_tokens: global per-instance token snapshot from solve_task (shared cap).
    # PatchPilot §3.4 + App D + repair.py (source-verified):
    # First batch: standard / comprehensive / minimal suffixes, up to config.n_candidates.
    # Subsequent batches (first_batch=False): standard only.
    # Plan call forces CoT; generate call produces edits.
    # RefinementContext feeds structured feedback for refinement batches (PatchPilot §3.5).
    # Deviation: N=3 default (one of each strategy); paper source confirms 3 greedy plans.
    # force_diverse (Task 9): on a refine stall, reactivate the comprehensive/minimal
    # suffixes to break out of a single-mechanism loop (paper uses standard-only in refine
    # for caching; our proxy gets no caching benefit, so escalating diversity is free).
    test_patch: hidden issue-fixing tests; files modified by it are blocked from editing.
    """
    test_files = extract_test_files(test_patch)
    # Task 13: scale plan constraints to localization complexity.
    n_files = len(localization.get("files", []))
    n_sites = len(localization.get("regions", {})) or n_files

    if localization.get("expand_full"):
        n_sites = 4   # -> step_cap = max(3, 4 * 2) = 8
    plan_system = _get_plan_system(n_sites)
    suffixes = _get_plan_suffixes(n_files) if (first_batch or force_diverse) else [_PLAN_SUFFIXES[0]]
    patches: list[str] = []

    for idx in range(config.n_candidates):
        if budget_exceeded(model, config, baseline_tokens):
            break

        suffix = suffixes[idx % len(suffixes)]

        # Step 1 — plan  (PatchPilot §3.4: explicit plan forces CoT reasoning)
        plan = model.generate(
            plan_system,
            _build_plan_user(context, localization, suffix, refinement, test_files=frozenset(test_files)),
            temperature=0.8,
        ).content.strip()

        if len(plan) < _MIN_PLAN_CHARS:
            continue  # garbage output — nothing written, no revert needed

        if budget_exceeded(model, config, baseline_tokens):
            break

        # Step 2 — generate  (PatchPilot §3.4: generate edits following the plan)
        gen_out = model.generate(
            _GEN_SYSTEM,
            _build_gen_user(context, plan, localization, refinement=refinement,test_files=frozenset(test_files)),
            temperature=0.1,
        ).content

        edits = parse_edits(gen_out)
        if not edits:
            continue  # nothing parsed — no file touched, no revert needed

        result = apply_edits(edits, env, test_files=test_files)
        if result.diff:
            patches.append(result.diff)

        # Revert between candidates so each starts from base_commit  (Req 5.4)
        revert(env)

    return patches
