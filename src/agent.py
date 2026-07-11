"""
SWE-bench repair agent — PatchPilot-inspired multi-phase pipeline.

Pipeline (PatchPilot §3.1 rule-based workflow):
  localize → generate → validate → refine → select best patch

The original baseline agentic loop is preserved below as a commented block for
comparison.
"""
from __future__ import annotations

from src.config import AgentConfig
from src.dataset import get_test_patch
from src.generate import ensure_tools, generate_candidates
from src.localize import localize
from src.refine import refine
from src.validate import ValidationResult, validate
from utils.docker_env import DockerEnv
from utils.models import ModelClient
from utils.tasks import TaskContext


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE (original agentic loop) — kept for reference / comparison
# ══════════════════════════════════════════════════════════════════════════════
#
# from __future__ import annotations
#
# import re
#
# from utils.docker_env import DockerEnv
# from utils.models import ModelClient
# from utils.patches import extract_patch
# from utils.tasks import TaskContext
#
#
# SYSTEM_PROMPT = """You are an autonomous SWE-bench CLI agent running inside a repository container.
# Think through shell commands, inspect code, edit files, and validate with tests.
#
# Return EXACTLY ONE of the following formats per turn:
# 1) For taking an action:
# <command>
# ONE SINGLE SHELL COMMAND
# </command>
#
# 2) When you are done:
# <final_patch>
# UNIFIED DIFF PATCH
# </final_patch>
#
# Rules:
# - Use non-interactive shell commands only.
# - Prefer focused inspections (`ls`, `find`, `grep`, `sed`, `python -m pytest <target>`).
# - Never use destructive system-level commands (reboot, shutdown, mkfs, etc.).
# - If you edit files, produce the final answer as a unified diff patch.
# - Do not output markdown fences or any extra text outside the required tags.
# """
#
# FINAL_PATCH_PROMPT = """You have reached the finalization step.
# Return ONLY:
# <final_patch>
# ...unified diff patch...
# </final_patch>
# No extra text.
# """
#
# MAX_STEPS = 12
# MAX_OBS_CHARS = 1200
# FORBIDDEN_COMMAND_PATTERNS = (
#     "shutdown", "reboot", "poweroff", "halt", "mkfs", "fdisk",
#     "dd if=", "rm -rf /", ":(){:|:&};:",
# )
#
# def _truncate(text: str, max_chars: int = MAX_OBS_CHARS) -> str:
#     cleaned = text.strip()
#     if len(cleaned) <= max_chars:
#         return cleaned
#     return f"{cleaned[:max_chars]}... [truncated]"
#
# def _extract_tagged(text: str, tag: str) -> str:
#     pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
#     match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
#     if not match:
#         return ""
#     return match.group(1).strip()
#
# def _is_forbidden_command(command: str) -> bool:
#     lowered = command.lower()
#     return any(token in lowered for token in FORBIDDEN_COMMAND_PATTERNS)
#
# def _parse_model_action(content: str) -> tuple[str, str]:
#     final_patch = _extract_tagged(content, "final_patch")
#     if final_patch:
#         return "final_patch", final_patch
#     command = _extract_tagged(content, "command")
#     if command:
#         return "command", command
#     patch = extract_patch(content)
#     if any(patch.startswith(s) for s in ("diff --git", "--- ", "Index: ", "*** Begin Patch")):
#         return "final_patch", patch
#     return "invalid", content.strip()
#
# def _build_task_prompt(context, history, steps_remaining, final_only=False):
#     history_block = "\n\n".join(history[-8:]).strip() if history else "(none)"
#     instruction = (
#         "Decide the next best single CLI command."
#         if not final_only
#         else "Now stop acting and produce the final patch."
#     )
#     return f"""Problem statement:\n{
# problem_statement}\n\nInteraction history:\n{history_block}\n\nInstruction:\n{instruction}\n\nSteps remaining: {steps_remaining}"""
#
# def _format_observation(exit_code, stdout, stderr):
#     out = stdout.strip()
#     err = _truncate(stderr)
#     return f"exit_code: {exit_code}\nstdout:\n{out or '(empty)'}\nstderr:\n{err or '(empty)'}"
#
# def solve_task(context: TaskContext, env: DockerEnv, model: ModelClient) -> str:
#     history: list[str] = []
#     for step in range(1, MAX_STEPS + 1):
#         steps_remaining = MAX_STEPS - step + 1
#         user_prompt = _build_task_prompt(context, history, steps_remaining=steps_remaining)
#         response = model.generate(SYSTEM_PROMPT, user_prompt, temperature=0.2)
#         action_type, payload = _parse_model_action(response.content)
#         if action_type == "final_patch":
#             return extract_patch(payload)
#         if action_type != "command":
#             history.append(f"[step {step} invalid]\nraw:\n{_truncate(response.content)}")
#             continue
#         command = payload.strip()
#         if not command:
#             history.append(f"[step {step} invalid]\nEmpty command.")
#             continue
#         if _is_forbidden_command(command):
#             history.append(f"[step {step} blocked]\ncommand: {command}\nreason: blocked by safety policy")
#             continue
#         result = env.run(command, timeout=120)
#         observation = _format_observation(result.exit_code, result.stdout, result.stderr)
#         history.append(f"[step {step} command]\n{command}\n[step {step} observation]\n{observation}")
#     # Finalization pass
#     final_prompt = _build_task_prompt(context, history, steps_remaining=0, final_only=True)
#     final_response = model.generate(
#         SYSTEM_PROMPT + "\n\nYou have reached the finalization step.\nReturn ONLY:\n<final_patch>\n...unified diff patch...\n</final_patch>\nNo extra text.",
#         final_prompt, temperature=0.0,
#     )
#     final_type, final_payload = _parse_model_action(final_response.content)
#     if final_type == "final_patch":
#         return extract_patch(final_payload)
#     diff_result = env.run("git diff", timeout=60)
#     return extract_patch(diff_result.stdout)
#
# ══════════════════════════════════════════════════════════════════════════════
# END BASELINE
# ══════════════════════════════════════════════════════════════════════════════

def _localize(context: TaskContext, env: DockerEnv, model: ModelClient, config: AgentConfig, baseline_tokens: int = 0) -> dict:
    """Hierarchical localization — PatchPilot §3.3 + App A/B (implemented in Task 2)."""
    return localize(context, env, model, config, baseline_tokens)


def _generate(
    context: TaskContext,
    env: DockerEnv,
    model: ModelClient,
    config: AgentConfig,
    localization: dict,
    baseline_tokens: int = 0,
    test_patch: str = "",
) -> list[str]:
    """Plan-then-generate candidates — PatchPilot §3.4 (implemented in Task 4)."""
    return generate_candidates(context, env, model, config, localization, baseline_tokens=baseline_tokens, test_patch=test_patch)


def _validate(
    candidates: list[str],
    context: TaskContext,
    env: DockerEnv,
    config: AgentConfig,
    test_patch: str = "",
) -> list[ValidationResult]:
    """Test-execution validation + ranking — PatchPilot §3.2 (implemented in Task 5).

    Deterministic (no LLM calls). Ranking matches official SWE-bench grading: any broken
    regression is a hard gate. Returns ValidationResult list, best-first.
    """
    return validate(candidates, context, env, config, test_patch)


def _select(scored: list[ValidationResult]) -> str:
    """Return the best patch. validate() already sorts best-first with smaller-patch
    tie-break, so the head is the winner (Requirement 9.2)."""
    return scored[0].patch if scored else ""


# ── public entrypoint ─────────────────────────────────────────────────────────

def solve_task(context: TaskContext, env: DockerEnv, model: ModelClient) -> str:
    """Multi-phase repair pipeline (PatchPilot §3.1 rule-based workflow).

    Pipeline: localize → generate → validate → refine → select best patch.
    """
    # PatchPilot §3.1: rule-based planning — fixed phase order for all instances.
    config = AgentConfig()
    baseline_tokens = model.get_usage().total_tokens  # snapshot before this instance

    # Install tooling once at Docker session start (flake8 for lint checks — Task 3)
    ensure_tools(env)

    # Phase 1 — Localize
    localization = _localize(context, env, model, config, baseline_tokens)

    # Guard: if localization failed (no root cause), skip generation to save tokens.
    # Empty localization → generate_candidates would produce poor patches with no useful context.
    if not localization.get("root_cause"):
        return ""

    # Load test_patch early so generate_candidates can block test-file edits.
    test_patch = get_test_patch(context.instance_id)

    # Phase 2 — Generate candidates (budget checked internally per candidate)
    candidates = _generate(context, env, model, config, localization, baseline_tokens=baseline_tokens, test_patch=test_patch)

    # Phase 3 — Validate + rank (deterministic test execution — Task 5; no LLM tokens).
    scored = _validate(candidates, context, env, config, test_patch)

    # Phase 4 — Refine loop (PatchPilot §3.5 — Task 6).
    # Refine the top-ranked patch on validation feedback until a qualified patch is found,
    # max_refine_iters is reached, or the token budget is exhausted. Returns best-so-far.
    best = refine(scored, context, env, model, config, localization, test_patch, baseline_tokens)

    # Phase 5 — Select best
    return best.patch if best else _select(scored)
