from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from contest.data import event_log


def _parse_timestamps(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.isna().any():
        bad_count = int(parsed.isna().sum())
        raise ValueError(f"Failed to parse {bad_count} timestamp rows in CSV.")
    return parsed


def prepare_event_log_csv(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    if "time:timestamp" not in df.columns:
        raise ValueError("Missing time:timestamp column in CSV.")

    df["time:timestamp"] = _parse_timestamps(df["time:timestamp"])
    df = df.sort_values(["case:concept:name", "time:timestamp"])

    end_times = df.groupby("case:concept:name")["time:timestamp"].transform("max")
    remaining_hours = (end_times - df["time:timestamp"]).dt.total_seconds() / 3600.0
    df["remaining_time"] = remaining_hours.astype(float)

    if "reward" in df.columns:
        df = df.drop(columns=["reward"])

    df.to_csv(output_path, index=False)


def prepare_bpic2012_csv(input_path: str, output_path: str) -> None:
    prepare_event_log_csv(input_path, output_path)


def prepare_bpic2017_csv(input_path: str, output_path: str) -> None:
    prepare_event_log_csv(input_path, output_path)


def load_event_log_csv(
    path: str,
    compute_remaining_time: bool = True,
    sample_n_cases: Optional[int] = None,
    seed: int = 0,
    dataset_label: str = "event_log",
) -> List[event_log.Case]:
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

    all_numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    ignored_numeric = ["remaining_time", "time:timestamp", "case:concept:name", "case:landRegisterID", "case:IDofConceptCase"]
    scalar_cols = [c for c in all_numeric_cols if c not in ignored_numeric]

    all_string_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    ignored_strings = ["case:concept:name", "concept:name", "org:resource", "actor", "start:timestamp"]
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


def load_bpic2012_csv(
    path: str,
    compute_remaining_time: bool = True,
    sample_n_cases: Optional[int] = None,
    seed: int = 0,
) -> List[event_log.Case]:
    return load_event_log_csv(
        path, compute_remaining_time=compute_remaining_time,
        sample_n_cases=sample_n_cases, seed=seed, dataset_label="bpic2012",
    )


def load_bpic2017_csv(
    path: str,
    compute_remaining_time: bool = True,
    sample_n_cases: Optional[int] = None,
    seed: int = 0,
) -> List[event_log.Case]:
    return load_event_log_csv(
        path, compute_remaining_time=compute_remaining_time,
        sample_n_cases=sample_n_cases, seed=seed, dataset_label="bpic2017_after",
    )
