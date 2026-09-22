"""
CSV loader that turns an XES-derived event log export into the Event objects and domain registry defined in event_log.py.
This loader is dataset-agnostic. It only requires three standard columns: case:concept:name, concept:name, and time:timestamp. 
All other columns are discovered automatically as either categorical or scalar features, and are attached to each Event's attributes dictionary.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from baseline.revised_plus.data import event_log


def _parse_timestamps(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(series, utc=True, errors="coerce")

    if parsed.isna().all() and series.notna().any():
        parsed = pd.to_datetime(series, utc=True, errors="coerce")

    if parsed.isna().any():
        bad_count = int(parsed.isna().sum())
        raise ValueError(f"Failed to parse {bad_count} timestamp rows in CSV.")
    return parsed


def _infer_case_id_like_columns(
    df: pd.DataFrame,
    candidate_cols: List[str],
    case_col: str = "case:concept:name",
    uniqueness_threshold: float = 0.95,
) -> List[str]:
    """
    Behaviorally detect columns (numeric or string) that are alternate
    case identifiers rather than genuine case-level features.

    Two conditions, both required:
      1. Constant within every case (doesn't vary event-to-event) —
         necessary but not sufficient, since real case-level attributes
         (e.g. AMOUNT_REQ) are also constant-per-case.
      2. Near-bijective with the case ID: distinct values in the column
         are close to the number of distinct cases
         (distinct_vals / n_cases >= uniqueness_threshold). A genuine
         attribute isn't structurally tied to case count; an ID column is
         (one ID per case, by definition).
    """
    n_cases = df[case_col].nunique()
    if n_cases == 0:
        return []

    id_like = []
    for col in candidate_cols:
        if col == case_col:
            continue
        per_case_nunique = df.groupby(case_col)[col].nunique(dropna=True)
        if not (per_case_nunique <= 1).all():
            continue  # varies within a case -> genuine event-level feature

        distinct_vals = df[col].nunique(dropna=True)
        if distinct_vals / n_cases >= uniqueness_threshold:
            id_like.append(col)

    return id_like


def _infer_timestamp_like_string_columns(
    df: pd.DataFrame,
    candidate_cols: List[str],
    parse_success_threshold: float = 0.95,
) -> List[str]:
    """
    Behaviorally detect string/object columns that hold timestamps
    rather than categorical values, without hardcoding a column name.

    A column qualifies if at least `parse_success_threshold` of its
    non-null values parse as datetimes via pandas. 
    """
    timestamp_like = []
    for col in candidate_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        parsed = pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
        success_rate = parsed.notna().mean()
        if success_rate >= parse_success_threshold:
            timestamp_like.append(col)
    return timestamp_like


def prepare_event_log_csv(input_path: str, output_path: str) -> None:
    """
    Precompute per-event remaining_time and write an enriched copy of the CSV.
    """
    df = pd.read_csv(input_path)
    if "time:timestamp" not in df.columns:
        raise ValueError("Missing time:timestamp column in CSV.")

    df["time:timestamp"] = _parse_timestamps(df["time:timestamp"])
    df = df.sort_values(["case:concept:name", "time:timestamp"])

    end_times = df.groupby("case:concept:name")["time:timestamp"].transform("max")
    remaining_hours = (end_times - df["time:timestamp"]).dt.total_seconds() / 3600.0
    df["remaining_time"] = remaining_hours.astype(float)

    df.to_csv(output_path, index=False)


def load_event_log_csv(
    path: str,
    compute_remaining_time: bool = True,
    sample_n_cases: Optional[int] = None,
    seed: int = 0,
    dataset_label: Optional[str] = None,
    force_keep_numeric: Optional[List[str]] = None,
    force_keep_string: Optional[List[str]] = None,
    id_uniqueness_threshold: float = 0.95,
    timestamp_parse_threshold: float = 0.95,
) -> List[event_log.Case]:
    """
    Load a CSV event log into DIFF-ERO's in-memory Case/Event representation.

    This is the function the rest of the pipeline actually calls. It:
      1. Parses timestamps and sorts events per case.
      2. Optionally subsamples to `sample_n_cases` cases (reproducible via `seed`).
      3. Computes remaining_time on the fly if `compute_remaining_time=True`
         (independent of whether prepare_event_log_csv was ever run).
      4. Auto-discovers categorical vocabularies and scalar (min, max)
         bounds from every non-reserved, non-structural, non-ID-like,
         non-timestamp-like column.
      5. Builds one Event per row (attributes dict) and groups into Case
         objects per case:concept:name.
      6. Registers the discovered activities/resources/vocabularies/bounds
         as global domain state via event_log.set_domain(...) and
         event_log.NUMERIC_MIN_MAX - required before encoder training or
         CF search can run.

    Args:
        force_keep_numeric: numeric column names to always treat as real
            scalar features even if flagged as case-ID-like (escape hatch
            for _infer_case_id_like_columns false positives).
        force_keep_string: string column names to always treat as real
            categorical features even if flagged as case-ID-like or
            timestamp-like (escape hatch for the same reason).
        id_uniqueness_threshold: passed through to _infer_case_id_like_columns.
        timestamp_parse_threshold: passed through to
            _infer_timestamp_like_string_columns.

    Note: if compute_remaining_time=False and the CSV has no
    remaining_time column of its own, remaining_time will be NaN for
    every event - this is silent, so check upstream if you rely on it.
    """
    df = pd.read_csv(path)
    if "time:timestamp" not in df.columns:
        raise ValueError("Missing time:timestamp column in CSV.")

    df["time:timestamp"] = _parse_timestamps(df["time:timestamp"])
    df = df.sort_values(["case:concept:name", "time:timestamp"])

    if sample_n_cases is not None:
        case_ids = df["case:concept:name"].dropna().unique().tolist()
        if sample_n_cases < len(case_ids):
            rng = np.random.default_rng(seed)
            sampled = rng.choice(case_ids, size=sample_n_cases, replace=False)
            df = df[df["case:concept:name"].isin(sampled)]

    if compute_remaining_time:
        end_times = df.groupby("case:concept:name")["time:timestamp"].transform("max")
        remaining_hours = (end_times - df["time:timestamp"]).dt.total_seconds() / 3600.0
        df["remaining_time"] = remaining_hours.astype(float)
    else:
        df["remaining_time"] = np.nan

    activities = sorted(df["concept:name"].dropna().unique().tolist())

    def _resource_label(row: pd.Series) -> str:
        actor = str(row.get("actor", "")).strip()
        org_res = str(row.get("org:resource", "")).strip()
        if actor and org_res:
            return f"{actor}:{org_res}"
        if actor:
            return actor
        if org_res:
            return org_res
        return "Resource_unknown"

    resources = sorted(df.apply(_resource_label, axis=1).unique().tolist())

    force_keep_numeric = set(force_keep_numeric or [])
    force_keep_string = set(force_keep_string or [])

    # --- structural exclusions: actively consumed elsewhere in this loader ---
    STRUCTURAL_NUMERIC_EXCLUSIONS = ["remaining_time", "time:timestamp", "case:concept:name"]
    STRUCTURAL_STRING_EXCLUSIONS = ["case:concept:name", "concept:name", "org:resource", "actor"]

    # --- numeric side: behaviorally detected case-ID-like columns ---
    all_numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_candidates = [c for c in all_numeric_cols if c not in STRUCTURAL_NUMERIC_EXCLUSIONS]
    id_like_numeric = [
        c for c in _infer_case_id_like_columns(df, numeric_candidates, uniqueness_threshold=id_uniqueness_threshold)
        if c not in force_keep_numeric
    ]
    if id_like_numeric:
        print(f"  Auto-excluded case-ID-like numeric columns: {id_like_numeric} "
              f"(pass force_keep_numeric=[...] to override)")

    ignored_numeric = STRUCTURAL_NUMERIC_EXCLUSIONS + id_like_numeric
    scalar_cols = [c for c in all_numeric_cols if c not in ignored_numeric]

    # --- string side: behaviorally detected timestamp-like and ID-like columns ---
    all_string_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    string_candidates = [c for c in all_string_cols if c not in STRUCTURAL_STRING_EXCLUSIONS]

    timestamp_like_strings = [
        c for c in _infer_timestamp_like_string_columns(df, string_candidates, parse_success_threshold=timestamp_parse_threshold)
        if c not in force_keep_string
    ]
    if timestamp_like_strings:
        print(f"  Auto-excluded timestamp-like string columns: {timestamp_like_strings} "
              f"(pass force_keep_string=[...] to override)")

    remaining_string_candidates = [c for c in string_candidates if c not in timestamp_like_strings]
    id_like_strings = [
        c for c in _infer_case_id_like_columns(df, remaining_string_candidates, uniqueness_threshold=id_uniqueness_threshold)
        if c not in force_keep_string
    ]
    if id_like_strings:
        print(f"  Auto-excluded case-ID-like string columns: {id_like_strings} "
              f"(pass force_keep_string=[...] to override)")

    ignored_strings = STRUCTURAL_STRING_EXCLUSIONS + timestamp_like_strings + id_like_strings
    cat_cols = [c for c in all_string_cols if c not in ignored_strings]

    RESERVED = {"activity", "resource", "concept:name"}
    scalar_cols = [c for c in scalar_cols if c.replace("case:", "") not in RESERVED]
    cat_cols = [c for c in cat_cols if c.replace("case:", "") not in RESERVED]

    discovered_vocabularies = {}
    for col in cat_cols:
        clean_name = col.replace("case:", "")
        discovered_vocabularies[clean_name] = df[col].dropna().unique().tolist()

    min_max_bounds = {}
    for col in scalar_cols:
        clean_name = col.replace("case:", "")
        min_max_bounds[clean_name] = (float(df[col].min()), float(df[col].max()))

    cases: List[event_log.Case] = []
    for case_id, g in df.groupby("case:concept:name"):
        g = g.sort_values("time:timestamp")

        events: List[event_log.Event] = []
        for idx in range(len(g)):
            row = g.iloc[idx]

            event_attrs = {
                "activity": str(row["concept:name"]),
                "resource": _resource_label(row),
            }

            for col in scalar_cols:
                event_attrs[col.replace("case:", "")] = float(row[col]) if pd.notna(row[col]) else 0.0
            for col in cat_cols:
                event_attrs[col.replace("case:", "")] = str(row[col]) if pd.notna(row[col]) else "missing"

            events.append(
                event_log.Event(
                    attributes=event_attrs,
                    remaining_time=float(row.get("remaining_time", np.nan))
                )
            )

        cases.append(event_log.Case(case_id=str(case_id), events=events))

    event_log.set_domain(activities, resources, cases_log=cases, discovered_vocabularies=discovered_vocabularies)
    event_log.NUMERIC_MIN_MAX = min_max_bounds

    print(f"  Loaded {len(cases)} cases for dataset='{dataset_label}': "
          f"{len(activities)} activities, {len(resources)} resources, "
          f"{len(discovered_vocabularies)} categorical cols, "
          f"{len(min_max_bounds)} scalar cols")

    return cases