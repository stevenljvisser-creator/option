"""Fail-closed point-in-time and temporal-overlap audits.

The functions in this module do not try to repair suspicious research data.
They produce machine-readable findings and mark a candidate invalid whenever a
future value, an unavailable source record, or an overlapping label window is
observed.  This keeps the rule ``information at t -> prediction for t+h`` an
executable contract instead of documentation only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import pandas as pd


FORBIDDEN_FEATURE_TOKENS = (
    "future_", "target_", "realized_payoff", "realised_payoff",
    "forward_return", "next_return", "outcome", "label_",
)


@dataclass
class LeakageAudit:
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def error(self, message: str) -> None:
        self.valid = False
        self.errors.append(str(message))

    def warning(self, message: str) -> None:
        self.warnings.append(str(message))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_feature_columns(columns: Iterable[str], allowed_targets: Iterable[str] = ()) -> LeakageAudit:
    audit = LeakageAudit()
    columns = list(columns)
    allowed = {str(value) for value in allowed_targets}
    bad = []
    for column in columns:
        name = str(column)
        low = name.lower()
        if name not in allowed and any(token in low for token in FORBIDDEN_FEATURE_TOKENS):
            bad.append(name)
    audit.checks["feature_columns"] = len(columns)
    audit.checks["forbidden_feature_columns"] = sorted(set(bad))
    if bad:
        audit.error("Toekomst- of labelkolommen gebruikt als feature: " + ", ".join(sorted(set(bad))))
    return audit


def audit_point_in_time(
    frame: pd.DataFrame,
    as_of_column: str = "minute",
    availability_columns: Iterable[str] = (
        "effective_available_utc", "record_available_utc", "vector_as_of_utc",
    ),
    target_time_column: str = "target_minute",
) -> LeakageAudit:
    """Check source availability and target ordering for every evaluated row."""
    audit = LeakageAudit()
    if frame is None or frame.empty:
        audit.error("Geen rijen beschikbaar voor point-in-time audit.")
        return audit
    if as_of_column not in frame:
        audit.error(f"As-of-kolom ontbreekt: {as_of_column}.")
        return audit
    as_of = pd.to_datetime(frame[as_of_column], utc=True, errors="coerce")
    missing_as_of = int(as_of.isna().sum())
    audit.checks["rows"] = int(len(frame))
    audit.checks["missing_as_of"] = missing_as_of
    if missing_as_of:
        audit.error(f"{missing_as_of} rijen missen een geldige as-of-tijd.")

    checked = 0
    future_counts: dict[str, int] = {}
    for column in availability_columns:
        if column not in frame:
            continue
        available = pd.to_datetime(frame[column], utc=True, errors="coerce")
        usable = available.notna() & as_of.notna()
        future = usable & (available > as_of)
        count = int(future.sum())
        checked += int(usable.sum())
        future_counts[column] = count
        if count:
            audit.error(f"{count} rijen gebruiken {column} na het voorspeltijdstip.")
    audit.checks["availability_values_checked"] = checked
    audit.checks["future_availability_by_column"] = future_counts

    if target_time_column in frame:
        target = pd.to_datetime(frame[target_time_column], utc=True, errors="coerce")
        comparable = target.notna() & as_of.notna()
        non_future = comparable & (target <= as_of)
        audit.checks["target_times_checked"] = int(comparable.sum())
        audit.checks["non_future_targets"] = int(non_future.sum())
        if non_future.any():
            audit.error(f"{int(non_future.sum())} labels liggen niet strikt na het voorspeltijdstip.")
    return audit


def audit_purged_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    as_of_column: str = "minute",
    label_end_column: str = "target_minute",
) -> LeakageAudit:
    """Require every training label to end before the first test observation."""
    audit = LeakageAudit()
    if train is None or train.empty or test is None or test.empty:
        audit.error("Train- of testdeel is leeg.")
        return audit
    if as_of_column not in train or as_of_column not in test:
        audit.error(f"Kolom {as_of_column} ontbreekt in train of test.")
        return audit
    train_as_of = pd.to_datetime(train[as_of_column], utc=True, errors="coerce")
    test_as_of = pd.to_datetime(test[as_of_column], utc=True, errors="coerce")
    label_end = (
        pd.to_datetime(train[label_end_column], utc=True, errors="coerce")
        if label_end_column in train else train_as_of
    )
    first_test = test_as_of.min()
    last_train_feature = train_as_of.max()
    last_train_label = label_end.max()
    audit.checks.update({
        "last_train_as_of": None if pd.isna(last_train_feature) else last_train_feature.isoformat(),
        "last_train_label_end": None if pd.isna(last_train_label) else last_train_label.isoformat(),
        "first_test_as_of": None if pd.isna(first_test) else first_test.isoformat(),
    })
    if pd.isna(first_test) or pd.isna(last_train_label):
        audit.error("Train-labelgrens of eerste testtijd is ongeldig.")
    elif last_train_label >= first_test:
        audit.error("Purged split ongeldig: een traininglabel overlapt de testperiode.")
    return audit


def audit_vector_provenance(metadata: dict[str, Any] | None, strict: bool = True) -> LeakageAudit:
    """Audit a stored daily-vector sidecar before it is accepted for research."""
    audit = LeakageAudit()
    meta = metadata or {}
    if not meta:
        audit.error("Vector-provenance ontbreekt.")
        return audit
    as_of = pd.to_datetime(meta.get("as_of_utc"), utc=True, errors="coerce")
    if pd.isna(as_of):
        audit.error("Vector-as-of ontbreekt of is ongeldig.")
    if bool(meta.get("future_values_used_as_features", True)):
        audit.error("Vector meldt toekomstwaarden in features.")
    news = meta.get("news") or {}
    basis = str(news.get("availability_basis") or "missing")
    audit.checks["news_availability_basis"] = basis
    if basis in {"missing", "legacy_publication_only"}:
        message = "Nieuws mist een sterke first-seen/effective-availability tijdsbasis."
        audit.error(message) if strict else audit.warning(message)
    earnings = meta.get("earnings") or {}
    if earnings.get("present") and not bool(earnings.get("point_in_time_enforced")):
        audit.error("Earningsdata is aanwezig zonder point-in-time handhaving.")
    audit.checks["as_of_utc"] = None if pd.isna(as_of) else as_of.isoformat()
    return audit


def merge_audits(*audits: LeakageAudit) -> LeakageAudit:
    merged = LeakageAudit()
    for index, audit in enumerate(audits):
        merged.valid = merged.valid and audit.valid
        merged.errors.extend(audit.errors)
        merged.warnings.extend(audit.warnings)
        merged.checks[f"audit_{index + 1}"] = audit.checks
    return merged


def require_valid(audit: LeakageAudit, context: str = "modelkandidaat") -> None:
    if not audit.valid:
        raise ValueError(f"{context} ongeldig wegens data leakage: " + " | ".join(audit.errors))
