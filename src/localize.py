"""
Hierarchical fault localization: files → functions/line regions.

PatchPilot §3.3 + App A/B: three-step procedure (file → class/function → line)
followed by a review step, implemented with two bounded LLM calls.

Deviations from paper (our choices):
- No PoC-coverage filtering (ablation §4.3: PoC-only hurts −1.7%).
- No 4× majority-voting at file level (cost; single call sufficient).
- Tools executed as grep shell commands rather than a dynamic tool-calling loop.
- Budget checked between steps to prevent localization consuming all tokens.
- Function body extraction: 120-line window + 6000-char cap
- Signatures limited to 2000 chars (~60 lines); full content only for top-2 files.
  Files 3-5 get signatures only to save tokens.
- Python-only filtering (*.py); SWE-bench is Python-only, so no need for multi-language.
"""
from __future__ import annotations

import json
import re
import shlex

from src.config import AgentConfig, budget_exceeded
from src.tools import run_tools as _run_tools
from utils.docker_env import DockerEnv
from utils.models import ModelClient
from utils.tasks import TaskContext

# ── prompt templates (single strings — no implicit line-break truncation) ─────

_FILE_SYSTEM = (
    "You are a fault-localization assistant for a Python codebase. "
    "Given the repository file tree and an issue description, identify the files most likely "
    "to contain the bug. You may call tools by writing them exactly as shown:\n\n"
    '  search_func_def("name")   — find file defining function <name>\n'
    '  search_class_def("name")  — find file defining class <name>\n'
    '  search_string("text")     — find file containing <text> most often\n'
    '  run_python("code")        — run a Python snippet to verify runtime behavior\n\n'
    "All arguments must be quoted strings. After optionally calling tools, reply with ONLY a JSON array of file paths, best first:\n"
    '["path/to/file.py", ...]\n'
    "No explanation outside the JSON array."
)

_FUNC_SYSTEM = (
    "You are a fault-localization assistant for a Python codebase. "
    "Given file contents and an issue description, identify the functions/classes most likely "
    "to contain the bug.\n\n"
    "Reply with ONLY a JSON array of objects:\n"
    '[{"file": "path/to/file.py", "name": "function_or_class_name"}, ...]\n'
    "No explanation outside the JSON array."
)

_REVIEW_SYSTEM = (
    "You are reviewing a fault-localization result for a Python codebase. "
    "Given the root cause snippet and the issue, check whether context is missing. "
    "If the root cause is under 150 lines, add any missing related code. "
    "Reply with the final root cause as plain text (the code to fix). Do not truncate it."
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_json_list(text: str) -> list:
    """Try each '[' in order; return the first that parses as a valid JSON array."""
    for m in re.finditer(r"\[", text):
        candidate = text[m.start():]
        depth, end = 0, -1
        for i, ch in enumerate(candidate):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            continue
        try:
            return json.loads(candidate[: end + 1])
        except Exception:
            continue
    return []


def _repo_tree(env: DockerEnv, max_lines: int = 300) -> str:
    r = env.run("git ls-files '*.py' | head -400", timeout=20)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    return "\n".join(lines[:max_lines])


def _signatures(env: DockerEnv, filepath: str, max_chars: int = 2000) -> str:
    r = env.run(f"grep -n 'def \\|class ' {shlex.quote(filepath)} | head -60", timeout=15)
    return r.stdout.strip()[:max_chars]


def _content(env: DockerEnv, filepath: str, max_chars: int = 6000) -> str:
    try:
        return env.read_file(filepath, max_chars=max_chars)
    except Exception:
        return ""


def _extract_region(env: DockerEnv, filepath: str, name: str, ctx: int = 15) -> str:
    """Source lines of function/class ± ctx lines.

    # PatchPilot App B: localized lines + surrounding ±15-line range = root cause.
    """
    qfp = shlex.quote(filepath)
    # The function step may return class-qualified names ("NDArithmeticMixin._arithmetic_mask").
    # A method definition is `def _arithmetic_mask`, not `def NDArithmeticMixin._arithmetic_mask`,
    # so match on the final dotted segment (Bug A: class-qualified names found nothing → empty
    # root cause → no usable candidates).
    simple = name.split(".")[-1]
    def_pat = shlex.quote(f"def {simple}\\b")
    cls_pat = shlex.quote(f"class {simple}\\b")
    r = env.run(f"grep -n -e {def_pat} -e {cls_pat} {qfp} | head -1", timeout=10)
    if not r.stdout.strip():
        return ""
    try:
        lineno = int(r.stdout.strip().split(":")[0])
    except ValueError:
        return ""
    start = max(1, lineno - ctx)
    end = lineno + 120 + ctx  # 120-line function body window; covers most real-world methods
    r2 = env.run(f"sed -n '{start},{end}p' {qfp}", timeout=10)
    return r2.stdout.strip()[:6000]  # cap to avoid overwhelming generation prompt


def _expand_full_smart(
    env: DockerEnv,
    files: list[str],
    prior_regions: dict[str, str] | None = None,
    target_frames: list[str] | None = None,
    max_chars_per_file: int = 30000,
) -> dict[str, str]:
    """Smart full-file expansion using anchor points from prior analysis.

    Strategy:
      - Small files (≤750 lines): read entire file with `cat`.
      - Large files (>750 lines):
          a. Always include def/class signatures as structural overview.
          b. Extract ±80-line windows around each anchor line from target_frames.
          c. Merge overlapping windows.
          d. If no anchors exist for a file, fall back to reading from the
             beginning with a larger max_chars cap (30000).
          e. Cap total output per file at max_chars_per_file.
    """
    prior_regions = prior_regions or {}
    target_frames = target_frames or []
    regions: dict[str, str] = {}

    # ── Collect anchor lines per file ──────────────────────────────
    # target_frames format: "file::function:line"
    anchors: dict[str, list[int]] = {fp: [] for fp in files}
    _frame_re = re.compile(r"^(.+?)::(\w+):(\d+)$")
    for frame in target_frames:
        m = _frame_re.match(frame)
        if m:
            fp, lineno = m.group(1), int(m.group(3))
            if fp in anchors:
                anchors[fp].append(lineno)

    # Also derive anchors from prior_regions keys ("file::func").
    # We scan for the function name in the file to get an approximate line number.
    _def_re_template = r"^\s*(?:def|async def)\s+{}\s*\("
    for region_key in prior_regions:
        parts = region_key.split("::", 1)
        if len(parts) != 2:
            continue
        fp, func = parts
        if fp not in anchors:
            continue
        qfp = shlex.quote(fp)
        r = env.run(
            f"grep -n '{shlex.quote(func)}' {qfp} | head -5",
            timeout=10,
        )
        if r.stdout.strip():
            for line in r.stdout.strip().splitlines():
                try:
                    lineno = int(line.split(":")[0])
                    anchors[fp].append(lineno)
                except (ValueError, IndexError):
                    continue

    # ── Extract content per file ──────────────────────────────────
    for fp in files:
        qfp = shlex.quote(fp)

        # Get line count
        wc = env.run(f"wc -l < {qfp}", timeout=10)
        try:
            total_lines = int(wc.stdout.strip())
        except (ValueError, AttributeError):
            total_lines = 0

        # Small file → read everything
        if total_lines <= 750:
            body = env.run(f"cat {qfp}", timeout=15).stdout
            if body:
                regions[fp] = body[:max_chars_per_file]
            continue

        # Large file → anchored extraction
        parts: list[str] = []

        # (a) Structural overview: all def/class signatures
        sigs = _signatures(env, fp, max_chars=3000)
        if sigs:
            parts.append(
                f"=== Structural overview (def/class in {fp}) ===\n{sigs}"
            )

        # (b) Extract ±80-line windows around anchors
        file_anchors = sorted(set(anchors.get(fp, [])))
        if file_anchors:
            # Build and merge windows
            half_window = 80
            windows: list[tuple[int, int]] = []
            for line in file_anchors:
                start = max(1, line - half_window)
                end = min(total_lines, line + half_window)
                if windows and start <= windows[-1][1] + 10:
                    windows[-1] = (windows[-1][0], max(windows[-1][1], end))
                else:
                    windows.append((start, end))

            for start, end in windows:
                r = env.run(
                    f"sed -n '{start},{end}p' {qfp}", timeout=15
                )
                if r.stdout.strip():
                    parts.append(
                        f"=== Lines {start}-{end} of {fp} ===\n"
                        f"{r.stdout.strip()}"
                    )
        else:
            # No anchors for this file → fallback: read more from beginning
            body = _content(env, fp, max_chars=max_chars_per_file)
            if body:
                parts.append(body)

        combined = "\n\n".join(parts)
        if combined:
            regions[fp] = combined[:max_chars_per_file]

    return regions

# ── localization steps ────────────────────────────────────────────────────────

def _step_files(
    context: TaskContext,
    env: DockerEnv,
    model: ModelClient,
    config: AgentConfig,
    baseline_tokens: int,
    feedback: str = "",
) -> list[str]:
    """File-level localization — PatchPilot §3.3 + App B: top-K files.

    feedback: validation results from a failed refinement batch — PatchPilot §3.5
    re-localization ("rerun localization with the validation results").
    """
    tree = _repo_tree(env)
    user = (
        f"Repository files:\n{tree}\n\n"
        f"Issue:\n{context.problem_statement[:3000]}\n\n"
        + (feedback + "\n\n" if feedback else "")
        + "Call tools if needed, then give the JSON array."
    )
    resp = model.generate(_FILE_SYSTEM, user, temperature=0.0)

    # Up to 3 retries when model makes tool calls but gets empty results
    # or when response doesn't contain a valid JSON array.
    max_retries = 2
    accumulated_tool_out = ""
    for attempt in range(max_retries):
        if budget_exceeded(model, config, baseline_tokens):
            break

        tool_out, had_calls = _run_tools(resp.content, env)
        files = [f for f in _parse_json_list(resp.content) if isinstance(f, str) and f.endswith(".py")]

        # Success: got files
        if files:
            return files[: config.localization_top_k]

        # Accumulate tool results
        if tool_out:
            accumulated_tool_out += f"\n\n{tool_out}" if accumulated_tool_out else tool_out

        # Model made tool calls and got results — let it continue (may make more tool calls)
        if had_calls and accumulated_tool_out:
            is_last = attempt == max_retries - 1
            prompt_suffix = (
                "\n\nNow give the final JSON array." if is_last
                else "\n\nCall more tools if needed, or give the final JSON array."
            )
            resp = model.generate(
                _FILE_SYSTEM,
                user + f"\n\nTool results so far:{accumulated_tool_out}{prompt_suffix}",
                temperature=0.0,
            )
        # Model made tool calls but got empty results — let it try different tools
        elif had_calls:
            is_last = attempt == max_retries - 1
            prompt_suffix = (
                "\n\nNow give the final JSON array." if is_last
                else "\n\nPlease retry another tool calls."
            )
            resp = model.generate(
                _FILE_SYSTEM,
                user + f"\n\nYour tool calls returned no results.{prompt_suffix}",
                temperature=0.0,
            )
        # Model didn't make tool calls but also didn't produce valid JSON — nudge it
        else:
            resp = model.generate(
                _FILE_SYSTEM,
                user + "\n\nPlease give the final JSON array.",
                temperature=0.0,
            )

    # Final parse after retries
    files = [f for f in _parse_json_list(resp.content) if isinstance(f, str) and f.endswith(".py")]
    return files[: config.localization_top_k]


def _step_functions(
    context: TaskContext,
    env: DockerEnv,
    model: ModelClient,
    files: list[str],
) -> list[dict]:
    """Function/line-level localization — PatchPilot §3.3 + App B."""
    blocks: list[str] = []
    for i, fp in enumerate(files):
        sigs = _signatures(env, fp)
        body = _content(env, fp) if i < 2 else ""  # full content top-2 only (token budget)
        block = f"=== {fp} ===\nSignatures:\n{sigs}"
        if body:
            block += f"\n\nFull content:\n{body}"
        blocks.append(block)

    user = (
        f"Issue:\n{context.problem_statement[:2000]}\n\n"
        + "\n\n".join(blocks)
        + "\n\nGive the JSON array of functions/classes to fix."
    )
    resp = model.generate(_FUNC_SYSTEM, user, temperature=0.0)
    items = _parse_json_list(resp.content)
    return [i for i in items if isinstance(i, dict) and "file" in i and "name" in i]


# ── public entrypoint ─────────────────────────────────────────────────────────

def localize(
    context: TaskContext,
    env: DockerEnv,
    model: ModelClient,
    config: AgentConfig,
    baseline_tokens: int = 0,
    feedback: str = "",
    expand_full: bool = False,
    prior_regions: dict[str, str] | None = None,
    target_frames: list[str] | None = None,
) -> dict:
    """Hierarchical fault localization → root-cause context dict.

    Returns dict with keys: files, functions, regions, root_cause.
    Returns {} on complete failure so the pipeline degrades gracefully (Req 12.1).
    baseline_tokens: global per-instance token snapshot from solve_task.
    feedback: validation results for re-localization (PatchPilot §3.5).
    """
    # Step 1 — files  (PatchPilot §3.3)
    files = _step_files(context, env, model, config, baseline_tokens, feedback)
    if not files:
        return {}

    # Task 12: full-file escalation. When narrow ±15-line regions have already failed to fix
    # the bug (refine stall), give the model the WHOLE localized file(s) so it sees every
    # sibling function / call-site (e.g. 14365 needs both `_line_type` and
    # `_get_tables_from_qdp_file`; 14182 needs all the RST methods together).
    if expand_full:
        regions = _expand_full_smart(
            env, files,
            prior_regions=prior_regions,
            target_frames=target_frames,
        )
        root_cause = "\n\n".join(f"# {k}\n{v}" for k, v in regions.items())
        return {"files": files, "functions": [], "regions": regions,
                "root_cause": root_cause, "expand_full": True}

    # Step 2 — functions/lines  (PatchPilot §3.3)
    if budget_exceeded(model, config, baseline_tokens):
        return {"files": files, "functions": [], "regions": {}, "root_cause": ""}
    functions = _step_functions(context, env, model, files)

    # Extract ±15-line regions  (PatchPilot App B)
    regions: dict[str, str] = {}
    for item in functions:
        region = _extract_region(env, item["file"], item["name"])
        if region:
            regions[f"{item['file']}::{item['name']}"] = region

    root_cause = "\n\n".join(f"# {k}\n{v}" for k, v in regions.items())

    # Review step  (PatchPilot §3.3): expand context if < 150 lines
    if root_cause and not budget_exceeded(model, config, baseline_tokens):
        lines = root_cause.count("\n")
        user = (
            f"Issue:\n{context.problem_statement[:1500]}\n\n"
            f"Current root cause ({lines} lines):\n{root_cause}\n\n"
            + ("Add any missing context needed to fix this issue."
               if lines < 150 else "Confirm correctness or correct the root cause.")
        )
        reviewed = model.generate(_REVIEW_SYSTEM, user, temperature=0.0).content.strip()
        # Only replace if review produced a substantive result (guard against garbage output)
        if reviewed and len(reviewed) >= len(root_cause) // 2:
            root_cause = reviewed

    return {"files": files, "functions": functions, "regions": regions, "root_cause": root_cause}

