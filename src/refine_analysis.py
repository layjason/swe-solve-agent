"""
Refinement analysis on escalation.

Runs only when the refinement loop stalls. Diagnoses why the current best patch
and previous attempts are failing, then guides the next refinement generation batch
and the one-time re-localization escalation.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.config import AgentConfig, budget_exceeded
from src.generate import _summarize_patch_for_prompt, revert
from src.tools import run_tools
from src.validate import _apply_patch
from utils.docker_env import DockerEnv

if TYPE_CHECKING:
    from src.refine import AttemptLedger
    from src.validate import ValidationResult
    from utils.models import ModelClient
    from utils.tasks import TaskContext

_ANALYSIS_CAP = 1800
_TOOL_RESULTS_CAP = 4000
_MAX_TOOL_ITERS = 4

_REFINEMENT_ANALYSIS_SYSTEM = (
    "You are a debugging analyst for a Python repair agent.\n\n"
    "Goal: explain why the current best patch failed or why refinement is stuck, "
    "using tool evidence, validation feedback, and previous attempts.\n\n"
    "You have tools available. To use them, write tool calls as plain text, "
    "one per line, with no markdown fences.\n\n"

    "Available tools:\n"
    '  search_func_def("name")   — find file defining function <name>\n'
    '  search_class_def("name")  — find file defining class <name>\n'
    '  search_string("text")     — find files containing exact text\n'
    '  run_python("code")        — run a Python snippet to verify runtime behavior\n\n'

    "Valid tool-call examples:\n"
    'search_func_def("process_data")\n'
    'search_string("header_rows")\n'
    'run_python("from astropy.io.ascii import rst; print(rst.SimpleRSTHeader().start_line)")\n\n'
    
    "Tool-use protocol:\n"
    "- On your FIRST response, you MUST call at least one tool.\n"
    "- You may call one or more tools in the same response.\n"
    "- Search tools are only for locating code. They are not enough evidence by themselves.\n"
    "- Before final analysis, you should normally call run_python(...) at least once.\n"
    "- If you use search_func_def/search_class_def/search_string first, use the result "
    "to construct a run_python(...) probe next.\n"
    "- The run_python(...) probe should verify the failing behavior, candidate patch behavior, "
    "or relevant object state.\n"
    "- Only skip run_python(...) when the validation traceback already proves the root cause "
    "without runtime inspection.\n"
    "- After TOOL RESULTS, either call another useful tool or return final analysis.\n"
    "- Do not finalize just because one search/location tool result was provided.\n"
    "- Use another tool only when it will materially improve the diagnosis.\n"
    "- Do not repeat the same probe unless the previous result was empty or ambiguous.\n"
    "- When the prompt says FINAL REQUIRED, stop using tools and return only "
    "<analysis>...</analysis>.\n\n"
    "Rules:\n"
    "- Do not write a patch.\n"
    "- Do not produce SEARCH/REPLACE blocks.\n"
    "- Do not propose full code edits.\n"
    "- Use tool results as evidence, not guesses.\n"
    "- Focus on root cause, evidence, what to try next, and what to avoid.\n\n"
    "Final response format:\n\n"
    "<analysis>\n"
    "Failure summary: ...\n"
    "Confirmed evidence: ...\n"
    "Likely root cause: ...\n"
    "Next patch direction: ...\n"
    "Avoid: ...\n"
    "</analysis>"
)


def _extract_analysis(text: str) -> str:
    """Extract analysis from <analysis> tags or fallback to stripped text."""
    m = re.search(r"<analysis>\s*(.*?)\s*</analysis>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()[:_ANALYSIS_CAP]
    return text.strip()[:_ANALYSIS_CAP]


def _has_analysis(text: str) -> bool:
    """Return True when the model emitted a tagged final analysis."""
    return bool(re.search(r"<analysis>\s*.*?</analysis>", text, re.DOTALL | re.IGNORECASE))


def _contains_run_python_call(text: str) -> bool:
    """Return True if the model attempted a runtime probe."""
    return "run_python(" in text


def analyze_refinement_feedback(
    context: "TaskContext",
    env: DockerEnv,
    model: "ModelClient",
    config: AgentConfig,
    localization: dict,
    best: "ValidationResult",
    ledger: "AttemptLedger",
    baseline_tokens: int,
    test_patch: str = "",
) -> str:
    """Analyze why refinement is stuck and return a guidance note.

    Runs only on escalation (when the refinement loop stalls). Uses tools
    to verify assumptions about runtime behavior if needed.
    """
    if budget_exceeded(model, config, baseline_tokens):
        return ""

    user = "\n\n".join([
        "--- BEGIN ISSUE ---\n"
        + context.problem_statement[:3000]
        + "\n--- END ISSUE ---",

        "--- BEGIN CURRENT LOCALIZATION ---\n"
        + str({
            "files": localization.get("files", []),
            "regions": {
                k: v[:500] + "..." if len(v) > 500 else v
                for k, v in localization.get("regions", {}).items()
            },
        })[:2000]
        + "\n--- END CURRENT LOCALIZATION ---",

        "--- BEGIN ROOT CAUSE CONTEXT ---\n"
        + localization.get("root_cause", "")[:4000]
        + "\n--- END ROOT CAUSE CONTEXT ---",

        "--- BEGIN PREVIOUS PATCH ---\n"
        + _summarize_patch_for_prompt(best.patch)
        + "\n--- END PREVIOUS PATCH ---",

        "--- BEGIN FAILING TARGET TESTS ---\n"
        + ("\n".join(best.failing_tests[:20]) if best.failing_tests else "(none)")
        + "\n--- END FAILING TARGET TESTS ---",

        "--- BEGIN BROKEN REGRESSION TESTS ---\n"
        + ("\n".join(best.broken_regressions[:20]) if best.broken_regressions else "(none)")
        + "\n--- END BROKEN REGRESSION TESTS ---",

        "--- BEGIN VALIDATION FEEDBACK ---\n"
        + best.test_output[:4000]
        + "\n--- END VALIDATION FEEDBACK ---",

        "--- BEGIN FRAMES ---\n"
        + ("\n".join((best.target_frames + best.regression_frames)[:30])
        if best.target_frames or best.regression_frames else "(none)")
        + "\n--- END FRAMES ---",

        "--- BEGIN PREVIOUS ATTEMPTS ---\n"
        + (ledger.render() or "(none)")
        + "\n--- END PREVIOUS ATTEMPTS ---",

        "Analyze why refinement is stuck.\n"
        "First, call at least one tool. Prefer run_python(...) to verify runtime behavior "
        "against the localized code or failing test behavior. Do not return <analysis> "
        "until TOOL RESULTS are provided.",
    ])

    resp = model.generate(_REFINEMENT_ANALYSIS_SYSTEM, user, temperature=0.0)

    accumulated_tool_out = ""
    saw_tool_results = False
    saw_runtime_probe = False
    should_revert = bool(best.patch.strip() or test_patch.strip())

    try:
        # Analyze the same state validation saw: candidate patch plus optional hidden test patch.
        # Without this, run_python(...) probes the clean base repo, not the failed candidate.
        if best.patch.strip():
            _apply_patch(env, best.patch, "analysis_current_patch")
        if test_patch.strip():
            _apply_patch(env, test_patch, "analysis_test_patch")

        for i in range(_MAX_TOOL_ITERS):
            if budget_exceeded(model, config, baseline_tokens):
                break

            is_last = i == _MAX_TOOL_ITERS - 1
            tool_out, had_calls = run_tools(resp.content, env)

            if had_calls:
                saw_tool_results = True
                if _contains_run_python_call(resp.content):
                    saw_runtime_probe = True

                accumulated_tool_out += "\n\n" + (tool_out.strip() or "[tools returned no output]")

                if is_last:
                    next_instruction = "FINAL REQUIRED. Return ONLY the final <analysis> block."
                elif not saw_runtime_probe:
                    next_instruction = (
                        "You have only used search/location tools so far. "
                        "Before final analysis, run at least one run_python(...) probe that verifies "
                        "the failing behavior, candidate patch behavior, or relevant object state. "
                        "Use the located files/symbols to construct the probe."
                    )
                else:
                    next_instruction = (
                        "You now have runtime TOOL RESULTS. Either call another useful tool, "
                        "or return the final <analysis> block. Do not repeat already-used probes."
                    )

                resp = model.generate(
                    _REFINEMENT_ANALYSIS_SYSTEM,
                    user
                    + "\n\n--- BEGIN TOOL RESULTS ---\n"
                    + accumulated_tool_out[-_TOOL_RESULTS_CAP:]
                    + "\n--- END TOOL RESULTS ---\n"
                    + next_instruction,
                    temperature=0.0,
                )
                continue

            if _has_analysis(resp.content):
                if saw_runtime_probe or is_last:
                    return _extract_analysis(resp.content)

                resp = model.generate(
                    _REFINEMENT_ANALYSIS_SYSTEM,
                    user
                    + "\n\nYou returned <analysis> after only search/location tools. "
                    + "Call one run_python(...) probe first to verify runtime behavior.",
                    temperature=0.0,
                )
                continue

            resp = model.generate(
                _REFINEMENT_ANALYSIS_SYSTEM,
                user
                + "\n\nYour previous response had neither valid tool calls nor a valid "
                + "<analysis> block. "
                + (
                    "FINAL REQUIRED. Return ONLY the final <analysis> block."
                    if is_last and saw_tool_results
                    else "Call a useful tool, or return a valid <analysis> block if TOOL RESULTS are sufficient."
                ),
                temperature=0.0,
            )

        return _extract_analysis(resp.content)

    finally:
        if should_revert:
            revert(env)
