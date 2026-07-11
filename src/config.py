"""
Agent configuration and budget management.

# PatchPilot §3.1: rule-based planning with a fixed workflow; the pipeline runs until a
# qualified patch is found or N_max total patches are exhausted (our: token budget).
# Deviation from paper: we use a token-budget early-exit instead of a patch-count ceiling,
# because token cost is an explicit grading criterion (20%) in this course.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.models import ModelClient


class ValidationMode(str, Enum):
    # PatchPilot §3.2: run PoC + functionality tests. We replace PoC with real FAIL_TO_PASS
    # tests materialized via the record's test_patch (our deviation — cheaper, stronger signal).
    TARGET_WITH_TEST_PATCH = "target_with_test_patch"
    # Pure variant: only existing PASS_TO_PASS regression tests (no hidden tests accessed).
    REGRESSION_ONLY = "regression_only"
    # PatchPilot §3.2 original: generate a PoC and run it (stub for now).
    POC = "poc"


@dataclass
class AgentConfig:
    # PatchPilot App B: N=4 candidates per batch, N_max=12 total.
    # Our deviation: N=3 (one of each strategy: standard/comprehensive/minimal)
    # to control token cost on deepseek-v3.2 while preserving diversity.
    n_candidates: int = 3
    # PatchPilot §3.5: iterative refinement until qualified or N_max reached.
    max_refine_iters: int = 3
    # Our addition (not in paper): per-instance token budget for early-exit cost control.
    token_budget: int = 200_000
    # Default: test-aware (option c — see docs/DEVELOPMENT_LOG.md §3 for tradeoff).
    validation_mode: ValidationMode = ValidationMode.TARGET_WITH_TEST_PATCH
    # PatchPilot App B: sample PASS_TO_PASS for functionality tests. We cap at 10.
    regression_sample_size: int = 10
    # PatchPilot §3.3 + App B: file-level localization returns top-5 files.
    localization_top_k: int = 5
    # Our addition: per test-run timeout (seconds) for validation (no LLM cost involved).
    test_timeout: int = 300


def budget_exceeded(model: "ModelClient", config: AgentConfig, baseline_tokens: int = 0) -> bool:
    """Return True if cumulative tokens since baseline have reached the token budget.

    # Our addition (not in paper): token-budget guard checked before each LLM-calling phase.
    """
    usage = model.get_usage()
    spent = usage.total_tokens - baseline_tokens
    return spent >= config.token_budget


@dataclass
class RefinementContext:
    """Structured feedback for the refinement loop.

    # PatchPilot §3.5 + repair.py: feedback = current_patch + failing_test_output +
    # newly_broken_regression_output. Structured here so generation builds a clean,
    # section-delimited prompt instead of opaque string concatenation.
    """
    current_patch: str          # best patch so far (what to refine, not replace)
    failing_tests: list[str]    # FAIL_TO_PASS test IDs that still fail
    test_output: str            # pytest stdout/stderr for failing tests (truncated)
    broken_regressions: list[str] = field(default_factory=list)   # PASS_TO_PASS tests newly broken
    note: str = ""              # Task 9/10: stall-breaker instruction (e.g. "try a different mechanism")
    analysis_note: str = ""     # escalation-only debugging analysis from analyze_refinement_feedback()
    ledger_text: str = ""       # Task 16: rendered "PREVIOUS ATTEMPTS" section (Reflexion-style)
    target_frames: list[str] = field(default_factory=list)   # Task 17: frames from target test only
    regression_frames: list[str] = field(default_factory=list)   # Task 17: frames from regression tests only
    test_source: str = ""       # Task 17: source code of failing test functions
