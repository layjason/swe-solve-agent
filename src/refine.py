"""
Patch refinement loop (PatchPilot §3.5 — unique to PatchPilot).

# PatchPilot §3.5: refine the TOP-RANKED patch by feeding the current batch + its validation
# results back to generation (plan prompt guides correcting the failed tests). Continue until a
# qualified patch passes all validations, or the patch limit (our: max_refine_iters / token
# budget) is reached. If a WHOLE batch passes no NEW tests, rerun localization with the
# validation results to obtain a new root cause. We always return the best patch seen so far.
#
# Ablation §4.3: refinement contributes +3.7% resolved — refining beats re-generating from
# scratch because it reuses the partially-correct patch and concrete test feedback.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.config import AgentConfig, RefinementContext, budget_exceeded
from src.generate import generate_candidates
from src.localize import localize
from src.refine_analysis import analyze_refinement_feedback
from src.validate import ValidationResult, baseline_target_output, is_no_effect, validate
from utils.docker_env import DockerEnv

if TYPE_CHECKING:
    from utils.models import ModelClient
    from utils.tasks import TaskContext

_FEEDBACK_CAP = 3000
_LEDGER_MAX_ENTRIES = 8          # cap rendered attempts to protect token budget
_HUNK_RE = re.compile(r"@@ .*?@@\s*(.*)")


@dataclass
class AttemptRecord:
    """One row in the cross-loop attempt ledger (Reflexion-style verbal episodic memory).

    # Task 16: per-candidate record so the model stops repeating a dead mechanism.
    # `signature` is a structural summary (per-hunk `file::func: old => new`) — cheap
    # heuristic, no LLM call — that distinguishes different mechanisms on the same site.
    """
    patch_hash: str
    sites: list[str]
    signature: str
    still_failing: list[str]
    no_effect: bool


def _summarize_patch(patch: str) -> tuple[list[str], str]:
    """Structural summary of a unified diff for the attempt ledger.

    Returns (sites, signature):
      sites     — distinct ``file::func`` (or ``file``) touched, for reference.
      signature — per-hunk ``file::func: <old> => <new>`` joined with " | ", capturing the
                  ACTUAL change so different mechanisms on the same site are distinguishable
                  (e.g. 6938 `x = x.replace(...)` vs gold `x[:] = x.replace(...)`). A flat
                  token bag could not tell these apart.
    """
    cur_file = "?"
    site = "?"
    sites: list[str] = []
    hunks: list[str] = []
    removed: str | None = None
    added: str | None = None

    def flush() -> None:
        nonlocal removed, added
        if removed is not None or added is not None:
            old = (removed or "∅").strip()
            new = (added or "∅").strip()
            hunks.append(f"{site}: {old} => {new}")
        removed = added = None

    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            cur_file = line[6:].strip()
        elif line.startswith("@@"):
            flush()                              # flush prior hunk with its (old) site
            m = _HUNK_RE.search(line)
            func = m.group(1).strip() if m and m.group(1).strip() else ""
            site = f"{cur_file}::{func}" if func else cur_file
            if site not in sites:
                sites.append(site)
        elif line.startswith("-") and not line.startswith("---"):
            if removed is None:
                removed = line[1:]
            else:
                removed += "\n" + line[1:]
        elif line.startswith("+") and not line.startswith("+++"):
            if added is None:
                added = line[1:]
            else:
                added += "\n" + line[1:]
    flush()

    signature = " | ".join(hunks[:4]) if hunks else "no change"
    return sites, signature


class AttemptLedger:
    """Cross-loop attempt ledger (Reflexion-style verbal episodic memory).

    # replaces the bare `seen` set with a structured ledger that records
    # every tried candidate and renders a compact "PREVIOUS ATTEMPTS" section into
    # the refine plan prompt.  Grounded in Reflexion (arXiv 2511.00197): verbal
    # episodic memory of past failures guides future attempts.
    """

    def __init__(self) -> None:
        self._attempts: list[AttemptRecord] = []
        self._seen_hashes: set[str] = set()

    @property
    def seen_hashes(self) -> set[str]:
        return self._seen_hashes

    def record(self, patch: str, result: ValidationResult) -> bool:
        """Record a candidate attempt. Returns False if already recorded (dedup)."""
        h = hashlib.sha256(patch.encode()).hexdigest()[:12]
        if h in self._seen_hashes:
            return False
        self._seen_hashes.add(h)
        sites, signature = _summarize_patch(patch)
        self._attempts.append(AttemptRecord(
            patch_hash=h,
            sites=sites,
            signature=signature,
            still_failing=list(result.failing_tests),
            no_effect=result.no_effect,
        ))
        return True

    def render(self) -> str:
        """Render a compact PREVIOUS ATTEMPTS section for the refine plan prompt.

        # Task 16: collapse identical mechanism signatures with a ×count so the model sees
        # when a SITE/approach is exhausted (e.g. "qdp.py::_line_type: ... (×6)") rather than
        # six near-duplicate lines. Each signature already embeds file::func, so multi-file /
        # multi-function patches are represented faithfully. still_failing is not rendered:
        # every recorded attempt failed by definition, so it would be noise.
        """
        if not self._attempts:
            return ""

        order: list[str] = []
        agg: dict[str, dict] = {}
        for a in self._attempts:
            info = agg.get(a.signature)
            if info is None:
                agg[a.signature] = {"count": 1, "no_effect": a.no_effect}
                order.append(a.signature)
            else:
                info["count"] += 1
                info["no_effect"] = info["no_effect"] or a.no_effect

        lines = ["Do not repeat these approaches:"]
        for sig in order[:_LEDGER_MAX_ENTRIES]:
            info = agg[sig]
            mult = f" (repeated {info['count']} x)" if info["count"] > 1 else ""
            eff = " [NO EFFECT on tests — this edit changed nothing]" if info["no_effect"] else " [PARTIALLY CORRECT on tests]"
            lines.append(f"- {sig}{mult}{eff}")
        extra = len(order) - _LEDGER_MAX_ENTRIES
        if extra > 0:
            lines.append(f"  ... and {extra} more distinct attempt(s)")

        return "\n".join(lines)


def _context_from(best: ValidationResult, ledger: AttemptLedger | None = None) -> RefinementContext:
    """Build a structured RefinementContext from the top-ranked validation result.

    test_output is the COMBINED pytest run (targets + regressions in one invocation), and
    broken_regressions already names the regression failures — so RefinementContext has no
    separate regression_output field (it would duplicate the same text and waste prompt tokens).

    # Task 16: ledger_text carries the rendered "PREVIOUS ATTEMPTS" section into the
    # refine plan prompt (Reflexion-style verbal episodic memory).
    # Task 17: target_frames and regression_frames are separated so target frames
    # (what the test exercises) are shown separately from regression frames (what was broken).
    # Task 17: test_source carries the source code of failing test functions so the model
    # can trace the execution path and understand what the test actually does.
    """
    return RefinementContext(
        current_patch=best.patch,
        failing_tests=best.failing_tests,
        test_output=best.test_output,
        broken_regressions=best.broken_regressions,
        ledger_text=ledger.render() if ledger else "",
        target_frames=best.target_frames,
        regression_frames=best.regression_frames,
        test_source=best.test_source,
    )


def _relocalize_feedback(
    best: ValidationResult,
    prior_files: list[str],
    analysis_note: str = "",
) -> str:
    parts = ["A previous patch did not resolve the issue."]

    if prior_files:
        parts.append(
            "Files already edited without success: "
            + ", ".join(sorted(prior_files))
            + "."
        )

    if best.failing_tests or best.target_frames:
        target_info = []
        if best.failing_tests:
            target_info.append("Still failing target tests: " + ", ".join(best.failing_tests[:10]))
        if best.target_frames:
            target_info.append("Target execution path: " + ", ".join(best.target_frames[:15]))
        parts.append("; ".join(target_info) + ".")

    if best.broken_regressions or best.regression_frames:
        regression_info = []
        if best.broken_regressions:
            regression_info.append(
                "Broken regression tests: " + ", ".join(best.broken_regressions[:10])
            )
        if best.regression_frames:
            regression_info.append(
                "Regression execution path: " + ", ".join(best.regression_frames[:10])
            )
        parts.append("; ".join(regression_info) + ".")

    if analysis_note:
        parts.append(
            "Hypothesis from refinement analysis. Treat this as advisory, not ground truth:\n"
            + analysis_note
        )

    parts.append(
        "Reconsider the root cause. The bug may be in a different file/function, "
        "or the previous patch may have targeted the right area with the wrong mechanism."
    )

    return "\n".join(parts)[:_FEEDBACK_CAP]


def refine(
    results: list[ValidationResult],
    context: "TaskContext",
    env: DockerEnv,
    model: "ModelClient",
    config: AgentConfig,
    localization: dict,
    test_patch: str,
    baseline_tokens: int = 0,
) -> ValidationResult | None:
    """Iteratively refine the top-ranked candidate; return the best result seen.

    results: initial validation results (best-first). Returns None only if there was
    nothing to refine and nothing was produced.

    # Task 16: uses an AttemptLedger (Reflexion-style verbal episodic memory) instead
    # of a bare `seen` set.  Every candidate is recorded with its mechanism summary,
    # and the ledger is rendered into every refine plan prompt so the model sees which
    # approaches have been exhausted.
    """
    if not results:
        return None

    best = results[0]
    if best.is_full:
        return best  # already qualified — no refinement needed

    relocalized = False
    prior_files = list(localization.get("files", []))
    # Task 10/16: no-fix target output, captured once, used to flag no-op patches (e.g. 6938).
    baseline_output = baseline_target_output(context, env, config, test_patch)
    best.no_effect = is_no_effect(best.test_output, baseline_output)
    ledger = AttemptLedger()
    ledger.record(best.patch, best)       # seed with the initial best
    stalled = False            # force diverse prompts next batch after a no-progress stall
    note = ""                  # stall-breaker instruction injected into the plan prompt
    # Run refinement analysis at most once per target-progress plateau.
    # This fixes the failure mode where score improves only because regressions are repaired,
    # while the target tests remain stuck at the same passed count.
    analysis_count = 0
    analysis_done_at_target_passed: int | None = None
    max_analysis_calls = 2
    pending_analysis_note = "" # analysis produced by a previous stalled iteration

    for _ in range(config.max_refine_iters):
        if budget_exceeded(model, config, baseline_tokens):
            break

        prev_target_passed = best.target_passed
        prev_score = best.score
        target_improved = False
        candidate_target_improved = False
        score_improved = False

        # Refine the current best: feed its validation feedback + ledger into plan-then-generate.
        refinement = _context_from(best, ledger)
        refinement.note = note
        refinement.analysis_note = pending_analysis_note
        candidates = generate_candidates(
            context, env, model, config, localization,
            refinement=refinement, first_batch=False, baseline_tokens=baseline_tokens,
            force_diverse=stalled, test_patch=test_patch,
        )

        if candidates:
            # Dedup via ledger (exact-hash check); record every novel candidate.
            dedupped_candidates: list[str] = []
            for c in candidates:
                h = hashlib.sha256(c.encode()).hexdigest()[:12]
                if h not in ledger.seen_hashes:
                    dedupped_candidates.append(c)

            if dedupped_candidates:
                batch = validate(dedupped_candidates, context, env, config, test_patch,
                                baseline_output=baseline_output)

                if batch:
                    # Record every validated candidate into the ledger (even non-winners).
                    for vr in batch:
                        ledger.record(vr.patch, vr)

                    batch_best = batch[0]

                    if batch_best.is_full:
                        return batch_best

                    candidate_target_improved = batch_best.target_passed > prev_target_passed
                    score_improved = batch_best.score > prev_score

                    # Preserve official-style ranking: only replace the incumbent best
                    # when the composite validation score improves.
                    if score_improved:
                        best = batch_best
                        target_improved = best.target_passed > prev_target_passed

        if target_improved:
            # Real target progress means we reached a new plateau. Do not carry stale
            # analysis forward; if this new target count gets stuck later, the keyed
            # analysis guard below can run fresh analysis for that new plateau.
            stalled = False
            note = ""
            pending_analysis_note = ""
        else:
            # Escalate not only on total no-progress, but also on score-only progress.
            # The latter is the important failure mode: existing/regression tests improve,
            # yet the target tests remain stuck at the same pass count.
            stalled = True

            if score_improved:
                stall_note = (
                    "Validation score improved, but the target tests still did not pass. "
                    "Do not focus only on preserving existing tests. Use TARGET FEEDBACK, "
                    "PREVIOUS ATTEMPTS, and any REFINEMENT ANALYSIS to fix the remaining "
                    "target failure with a meaningfully different mechanism if needed."
                )
            elif candidate_target_improved:
                stall_note = (
                    "A candidate improved the target tests but was not selected because the "
                    "overall validation score did not improve, likely due to regressions. "
                    "Use the ledger and feedback to keep the target-fixing mechanism while "
                    "removing the regression breakage."
                )
            else:
                stall_note = (
                    "Your previous patch did not improve validation. "
                    "See PREVIOUS ATTEMPTS and avoid exact variations of the same mechanism. "
                    "Try a fundamentally different fix: different mechanism, function, or file."
                )

            analysis_note = ""

            should_run_analysis = (
                analysis_count < max_analysis_calls
                and analysis_done_at_target_passed != best.target_passed
                and not budget_exceeded(model, config, baseline_tokens)
            )

            if should_run_analysis:
                analysis_note = analyze_refinement_feedback(
                    context=context,
                    env=env,
                    model=model,
                    config=config,
                    localization=localization,
                    best=best,
                    ledger=ledger,
                    baseline_tokens=baseline_tokens,
                    test_patch=test_patch,
                )
                analysis_count += 1
                analysis_done_at_target_passed = best.target_passed

            if analysis_note:
                pending_analysis_note = analysis_note

            note = stall_note

            # Second escalation: re-localize once with validation feedback (PatchPilot §3.5).
            if pending_analysis_note and not relocalized and not budget_exceeded(model, config, baseline_tokens):
                relocalized = True

                feedback = _relocalize_feedback(
                    best,
                    prior_files,
                    analysis_note=pending_analysis_note,
                )

                new_loc = localize(
                    context, env, model, config, baseline_tokens,
                    feedback=feedback,
                    expand_full=True,   # full-file context covers all sibling sites
                    prior_regions=localization.get("regions", {}),
                    target_frames=(best.target_frames or []) + (best.regression_frames or []),
                )
                if new_loc and new_loc.get("root_cause"):
                    localization = new_loc

    return best
