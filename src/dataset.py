"""
Framework-independent dataset loader for the hidden ``test_patch``.

# Constraint (DEVELOPMENT_LOG §3a): the course contract is "implement solve_task". The
# harness passes only the projected TaskContext (no raw record, no test_patch) and may reset
# utils/ at grading. So we keep utils/ read-only and load test_patch ourselves, keyed by
# instance_id, from the locally-downloaded dataset under data/*.
#
# Zero LLM tokens; load_from_disk is memory-mapped Arrow and cached per process so the dataset
# is read at most once regardless of how many instances solve_task handles.
"""
from __future__ import annotations

import glob
from functools import lru_cache


# Search roots, most-specific first. main.py's default_local_dataset_path puts the dataset at
# data/<dataset-name-with-slashes-as-__>, so data/* covers the standard workflow.
_SEARCH_GLOBS = ("data/*", "data")


@lru_cache(maxsize=1)
def _instance_index() -> dict[str, dict]:
    """Map instance_id -> record, scanning every dataset dir/split under data/.

    Cached: built once per process. Returns {} if nothing is discoverable so callers can
    fall back gracefully (e.g. to regression_only validation).
    """
    try:
        from datasets import DatasetDict, load_from_disk
    except Exception:
        return {}

    index: dict[str, dict] = {}
    seen_paths: set[str] = set()
    for pattern in _SEARCH_GLOBS:
        for path in sorted(glob.glob(pattern)):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                obj = load_from_disk(path)
            except Exception:
                continue
            splits = obj.values() if isinstance(obj, DatasetDict) else [obj]
            for split in splits:
                if "instance_id" not in split.column_names:
                    continue
                for record in split:
                    iid = record.get("instance_id")
                    if iid and iid not in index:
                        index[iid] = record
    return index


def get_test_patch(instance_id: str) -> str:
    """Return the hidden test_patch for instance_id, or "" if not discoverable."""
    record = _instance_index().get(instance_id)
    if not record:
        return ""
    return record.get("test_patch") or ""
