"""Which runs keep their retrieval windows, and which lose them.

A run's windows are what a reader debugs a retrieval with, and they are most
of what a stored run weighs. Old runs are rarely debugged again, so their
windows are cleared: by age, and then by volume while the store is over its
limit. What is concluded from the windows is kept before they go, by the
caller, so this decides only which runs.

Always kept: the last few runs of every realm, so that a person coming back
after a long break can carry on debugging where they stopped, and any run the
caller names as protected (the baseline every other run is compared with).

Pure: no I/O.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass(frozen=True)
class Retention:
    keep_last_per_realm: int = 2
    max_age_days: int = 30
    max_mib: int = 512


@dataclass(frozen=True)
class StoredRun:
    run_id: str
    realm_id: str
    started_at: str
    window_bytes: int
    cleared: bool


def _moment(started_at: str) -> datetime | None:
    """A run's start as an aware moment, or None when it cannot be read.

    Some runs record a moment without a zone, written by `utcnow()`, and they
    are read as UTC, which is what `utcnow()` wrote.
    """
    try:
        moment = datetime.fromisoformat(started_at)
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _oldest_first(runs: Iterable[StoredRun]) -> list[StoredRun]:
    # A run whose start cannot be read sorts as the oldest: its age is
    # unknown, so age alone never clears it, and volume reaches it first.
    floor = datetime.min.replace(tzinfo=UTC)
    return sorted(runs, key=lambda r: (_moment(r.started_at) or floor, r.run_id))


def protected_runs(runs: Iterable[StoredRun], keep_last_per_realm: int) -> set[str]:
    """The last runs of every realm; runs without a realm count as one realm."""
    by_realm: dict[str, list[StoredRun]] = {}
    for run in runs:
        by_realm.setdefault(run.realm_id or "", []).append(run)
    kept: set[str] = set()
    for realm_runs in by_realm.values():
        newest = list(reversed(_oldest_first(realm_runs)))
        kept.update(r.run_id for r in newest[:max(keep_last_per_realm, 0)])
    return kept


def runs_to_clear(
    runs: list[StoredRun],
    retention: Retention,
    *,
    stored_bytes: int,
    protected: Iterable[str] = (),
    now: datetime,
) -> list[tuple[str, str]]:
    """The runs whose windows go, oldest first, each with why: `age` or `volume`.

    `stored_bytes` is what the windows and the texts weigh together. Only the
    windows are subtracted as runs are chosen, because a text is freed only
    when no run at all refers to it any more, which this cannot know; the
    estimate is therefore high, and the store may reach its limit only on
    the next pass.
    """
    keep = protected_runs(runs, retention.keep_last_per_realm) | set(protected)
    candidates = [r for r in _oldest_first(runs) if not r.cleared and r.run_id not in keep]
    cutoff = now - timedelta(days=retention.max_age_days)

    chosen: list[tuple[str, str]] = []
    remaining = stored_bytes
    for run in candidates:
        moment = _moment(run.started_at)
        if moment is not None and moment < cutoff:
            chosen.append((run.run_id, "age"))
            remaining -= run.window_bytes

    limit = retention.max_mib * 1024 * 1024
    taken = {run_id for run_id, _ in chosen}
    for run in candidates:
        if remaining <= limit:
            break
        if run.run_id in taken:
            continue
        chosen.append((run.run_id, "volume"))
        remaining -= run.window_bytes
    return chosen
