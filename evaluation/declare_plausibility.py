import argparse
import csv
import json
import os
import re
from pathlib import Path

import pandas as pd

from baseline.revised_plus.iBCM.iBCM import ConstraintMining
from baseline.revised_plus.iBCM.run_iBCM import mine_constraints

CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
RESOURCE_COL = "org:resource"
TIMESTAMP_COL = "time:timestamp"

MIN_SUP = 1
NO_WIN = 1

FNAME_RE = re.compile(
    r"case_(?P<case_id>.+)_step_(?P<slice_idx>\d+)_m(?P<milestone>\d+)pct_(?P<mode>\w+)\.json"
)


def parse_filename(path):
    m = FNAME_RE.match(Path(path).name)
    if not m:
        return {}
    d = m.groupdict()
    d["case_id"] = d["case_id"]
    d["slice_idx"] = int(d["slice_idx"])
    d["milestone_pct"] = int(d["milestone"])
    return d


def load_cf_json(path):
    with open(path) as f:
        return json.load(f)


def parse_changed_events(summary_text):
    changes = []
    block = summary_text.split("Changed Events:")[-1].split("Changed Attributes:")[0]
    pattern = re.compile(
        r"Step (\d+): \(([^,]+),\s*([^)]+)\) to \(([^,]+),\s*([^)]+)\)"
    )
    for m in pattern.finditer(block):
        step, orig_act, orig_res, cf_act, cf_res = m.groups()
        changes.append({
            "step": int(step),
            "original": (orig_act.strip(), orig_res.strip()),
            "counterfactual": (cf_act.strip(), cf_res.strip()),
        })
    return changes


def extract_cf_recommendation(summary_text):
    m = re.search(r"Counterfactual Recommendation:\s*\(([^,]+),\s*([^)]+)\)", summary_text)
    if not m:
        raise ValueError("Could not parse CF recommendation from summary_text")
    return (m.group(1).strip(), m.group(2).strip())


def extract_original_recommendation(summary_text):
    m = re.search(r"Original Black-Box Recommendation:\s*\(([^,]+),\s*([^)]+)\)", summary_text)
    if not m:
        raise ValueError("Could not parse original black-box recommendation from summary_text")
    return (m.group(1).strip(), m.group(2).strip())


def _load_raw_log(path):
    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="raise", format='mixed')
    df = df.sort_values([CASE_COL, TIMESTAMP_COL])
    return df


def _traces_from_raw_log(raw_log_df):
    traces = {}
    for case_id, group in raw_log_df.groupby(CASE_COL):
        events = list(zip(group[ACTIVITY_COL].astype(str), group[RESOURCE_COL].astype(str)))
        traces[str(case_id)] = events
    return traces


def get_unchanged_prefix(case_id, slice_idx, raw_log_df, _trace_cache={}):
    if not _trace_cache:
        _trace_cache.update(_traces_from_raw_log(raw_log_df))

    case_events = _trace_cache.get(str(case_id))
    if case_events is None:
        raise KeyError(f"case_id {case_id} not found in raw log")

    return case_events[: slice_idx - 1]


def reconstruct_full_cf_trace(case_json, cf_candidate, raw_log_df):
    case_id = case_json["case_id"]
    slice_idx = case_json["slice_idx"]

    changes = parse_changed_events(cf_candidate["summary_text"])
    change_by_step = {c["step"]: c["counterfactual"] for c in changes}

    full_case_events = get_unchanged_prefix(case_id, slice_idx + 1, raw_log_df)

    full_trace = []
    for step in range(1, slice_idx + 1):
        full_trace.append(change_by_step.get(step, full_case_events[step - 1]))

    cf_rec_activity_resource = extract_cf_recommendation(cf_candidate["summary_text"])
    full_trace.append(cf_rec_activity_resource)

    return full_trace


SENTINEL_ACTIVITIES = {"START", "END"}


def mine_constraints_from_log(raw_log_df, min_sup=MIN_SUP, no_win=NO_WIN):
    traces_by_case = _traces_from_raw_log(raw_log_df)

    all_traces = []
    all_activities = set()
    prefixes_by_label = {}

    for case_id, events in traces_by_case.items():
        activities = [a for a, _r in events if a not in SENTINEL_ACTIVITIES]
        if not activities:
            continue
        all_traces.append(activities)
        all_activities.update(activities)

        for i in range(1, len(activities)):
            prefix = activities[:i]
            label = activities[i]
            prefixes_by_label.setdefault(label, []).append(prefix)

    print(f"Mining TDC over {len(all_traces)} traces, {len(all_activities)} activities...")
    tdc_keys, _ = mine_constraints(all_traces, sorted(all_activities), "TDC", min_sup, no_win)
    trace_constraints = set(tdc_keys)
    print(f"  -> {len(trace_constraints)} TDC constraints (min_sup={min_sup})")

    label_constraints = {}
    for label, label_prefixes in prefixes_by_label.items():
        label_prefixes = [p for p in label_prefixes if len(p) >= no_win]
        if not label_prefixes:
            continue
        activity_count = {}
        for prefix in label_prefixes:
            for act in prefix:
                activity_count[act] = activity_count.get(act, 0) + 1
        non_redundant = sorted(
            act for act, cnt in activity_count.items()
            if cnt >= len(label_prefixes) * min_sup
        )
        ldc_keys, _ = mine_constraints(label_prefixes, non_redundant, label, min_sup, no_win)
        label_constraints[label] = set(ldc_keys)

    print(f"  -> LDC mined for {len(label_constraints)} labels "
          f"(from {sum(len(v) for v in prefixes_by_label.values())} prefix instances)")

    return {
        "no_win": no_win,
        "trace_constraints": trace_constraints,
        "label_constraints": label_constraints,
    }


UNARY = {"absence", "exactly", "exactly2", "existence3", "init", "last"}


def export_constraints(constraints, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    def key(c):
        return (c.name, str(c.a), str(c.b))

    def b_of(c):
        return "" if c.name in UNARY else c.b

    with open(os.path.join(out_dir, "tdc.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["template", "a", "b"])
        for c in sorted(constraints["trace_constraints"], key=key):
            w.writerow([c.name, c.a, b_of(c)])
    with open(os.path.join(out_dir, "ldc.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label_activity", "template", "a", "b"])
        for label in sorted(constraints["label_constraints"]):
            for c in sorted(constraints["label_constraints"][label], key=key):
                w.writerow([label, c.name, c.a, b_of(c)])


def _mined_constraints_for(activities, required, no_win):
    if len(activities) < no_win or not required:
        return set()
    relevant_activities = {c.a for c in required} | {c.b for c in required} | set(activities)
    miner = ConstraintMining(activities, "cf_check", sorted(relevant_activities), no_win)
    return miner.FindConstraints()


def _split_checkable(required, present_activities):
    checkable, not_yet = set(), set()
    for c in required:
        needed = {c.a} if c.a == c.b else {c.a, c.b}
        (checkable if needed <= present_activities else not_yet).add(c)
    return checkable, not_yet


def _tag_constraint_scope(constraint, changed_activities, prefix_activities_before_change):
    operands = {constraint.a, constraint.b}
    involves_changed = bool(operands & changed_activities)
    involves_prior = bool(operands & set(prefix_activities_before_change))

    if involves_changed and involves_prior:
        return "neighborhood"
    if involves_changed:
        return "local"
    return "global"


def check_declare_violations(trace, constraints, label=None,
                              changed_activities=None,
                              prefix_activities_before_change=None):
    no_win = constraints.get("no_win", NO_WIN)
    activities = [ev[0] for ev in trace if ev[0] not in SENTINEL_ACTIVITIES]
    present = set(activities)

    tdc_required_all = set(constraints["trace_constraints"])
    tdc_checkable, tdc_not_yet = _split_checkable(tdc_required_all, present)
    tdc_found = _mined_constraints_for(activities, tdc_checkable, no_win)
    tdc_violated = [c for c in tdc_checkable if c not in tdc_found]

    ldc_required = set()
    ldc_violated = []
    if label is not None and label in constraints["label_constraints"]:
        ldc_required = set(constraints["label_constraints"][label])
        prefix_activities = activities[:-1] if activities and activities[-1] == label else activities
        ldc_found = _mined_constraints_for(prefix_activities, ldc_required, no_win)
        ldc_violated = [c for c in ldc_required if c not in ldc_found]

    has_tdc = len(tdc_checkable) > 0
    has_ldc = len(ldc_required) > 0
    if not has_tdc and not has_ldc:
        return None

    plausible_ldc = (len(ldc_violated) == 0) if has_ldc else None
    plausible_tdc = (len(tdc_violated) == 0) if has_tdc else None
    plausible_strict = (len(tdc_violated) == 0 and len(ldc_violated) == 0)

    result = {
        "tdc_violations": len(tdc_violated),
        "tdc_checkable": len(tdc_checkable),
        "tdc_not_yet_applicable": len(tdc_not_yet),
        "ldc_violations": len(ldc_violated),
        "ldc_required": len(ldc_required),
        "declare_plausible_ldc_only": plausible_ldc,
        "declare_plausible_tdc_only": plausible_tdc,
        "declare_plausible_strict": plausible_strict,
        "declare_n_violations": len(ldc_violated),
        "declare_plausible": plausible_ldc,
    }

    CHAIN_TEMPLATES = {"chain_response", "chain_precedence", "chain_succession"}

    if changed_activities:
        prefix_before = prefix_activities_before_change or []
        scope_total = {"local": 0, "neighborhood": 0, "global": 0}
        scope_violated = {"local": 0, "neighborhood": 0, "global": 0}
        scope_violated_chain = {"local": 0, "neighborhood": 0, "global": 0}
        scope_violated_nonchain = {"local": 0, "neighborhood": 0, "global": 0}
        scope_total_chain = {"local": 0, "neighborhood": 0, "global": 0}
        scope_total_nonchain = {"local": 0, "neighborhood": 0, "global": 0}
        violated_detail = []

        def _record(c, scope, is_violation, source):
            is_chain = c.name in CHAIN_TEMPLATES
            scope_total[scope] += 1
            (scope_total_chain if is_chain else scope_total_nonchain)[scope] += 1
            if is_violation:
                scope_violated[scope] += 1
                (scope_violated_chain if is_chain else scope_violated_nonchain)[scope] += 1
                violated_detail.append({
                    "source": source,
                    "scope": scope,
                    "template": c.name,
                    "is_chain": is_chain,
                    "a": c.a,
                    "b": c.b,
                })

        for c in tdc_checkable:
            scope = _tag_constraint_scope(c, changed_activities, prefix_before)
            _record(c, scope, c in tdc_violated, "tdc")

        for c in ldc_required:
            scope = _tag_constraint_scope(c, changed_activities, prefix_before)
            _record(c, scope, c in ldc_violated, "ldc")

        for scope in ("local", "neighborhood", "global"):
            total = scope_total[scope]
            violated = scope_violated[scope]
            result[f"{scope}_total"] = total
            result[f"{scope}_violations"] = violated
            result[f"{scope}_plausibility"] = (
                1.0 - (violated / total) if total > 0 else None
            )
            result[f"{scope}_total_chain"] = scope_total_chain[scope]
            result[f"{scope}_total_nonchain"] = scope_total_nonchain[scope]
            result[f"{scope}_violations_chain"] = scope_violated_chain[scope]
            result[f"{scope}_violations_nonchain"] = scope_violated_nonchain[scope]

        result["violated_detail"] = violated_detail

    return result


def score(train_csv, test_csv, explanations_dir=None, output_csv=None, export_constraints_dir=None, quiet=False):
    import contextlib
    import io
    out = io.StringIO() if quiet else None
    with (contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext()):
        return _score(train_csv, test_csv, explanations_dir, output_csv, export_constraints_dir)


def _score(train_csv, test_csv, explanations_dir, output_csv, export_constraints_dir):
    train_log_df = _load_raw_log(train_csv)
    constraints = mine_constraints_from_log(train_log_df)
    if export_constraints_dir:
        export_constraints(constraints, export_constraints_dir)
        print(f"Constraints written to {export_constraints_dir}")
    if not explanations_dir:
        return None

    cf_files = sorted(Path(explanations_dir).glob("case_*.json"))
    if not cf_files:
        print(f"No case_*.json files found in {explanations_dir}.")
        return None
    print(f"Found {len(cf_files)} per-case CF files.")

    test_log_df = _load_raw_log(test_csv)

    results = []
    for fp in cf_files:
        meta = parse_filename(fp)
        case_json = load_cf_json(fp)
        for cf in case_json["counterfactuals"]:
            cf_activity = None
            orig_rec_activity = None
            try:
                full_trace = reconstruct_full_cf_trace(case_json, cf, test_log_df)
                cf_activity = full_trace[-1][0]

                changes = parse_changed_events(cf["summary_text"])
                changed_activities = set()
                earliest_changed_step = None
                for c in changes:
                    changed_activities.add(c["original"][0])
                    changed_activities.add(c["counterfactual"][0])
                    earliest_changed_step = (
                        c["step"] if earliest_changed_step is None
                        else min(earliest_changed_step, c["step"])
                    )
                orig_rec_activity, _orig_rec_resource = extract_original_recommendation(cf["summary_text"])
                changed_activities.add(cf_activity)
                if orig_rec_activity:
                    changed_activities.add(orig_rec_activity)

                prefix_cutoff = (earliest_changed_step - 1) if earliest_changed_step else case_json["slice_idx"]
                prefix_before_change = [
                    ev[0] for ev in get_unchanged_prefix(case_json["case_id"], prefix_cutoff + 1, test_log_df)
                ]

                check = check_declare_violations(
                    full_trace, constraints, label=cf_activity,
                    changed_activities=changed_activities,
                    prefix_activities_before_change=prefix_before_change,
                )
            except (KeyError, ValueError) as e:
                print(f"  [warn] skipping candidate in {fp.name}: {e}")
                check = None

            check = check or {}
            results.append({
                "case_id": case_json["case_id"],
                "slice_idx": case_json["slice_idx"],
                "milestone": case_json["milestone"],
                "kpi_rank": cf["kpi_rank"],
                "contrast_type": cf["contrast_type"],
                "cf_activity": cf_activity,
                "orig_rec_activity": orig_rec_activity,
                "sparsity": cf["cf_diagnostics"]["sparsity"],
                "n_iterations": cf["cf_diagnostics"]["n_iterations"],
                "used_expulsion": cf["cf_diagnostics"]["used_expulsion"],
                "l_man": cf["cf_diagnostics"]["l_man"],
                "proximity": cf["cf_diagnostics"]["proximity_latent_l2sq"],
                "declare_n_violations": check.get("declare_n_violations"),
                "declare_plausible": check.get("declare_plausible"),
                "declare_plausible_ldc_only": check.get("declare_plausible_ldc_only"),
                "declare_plausible_tdc_only": check.get("declare_plausible_tdc_only"),
                "declare_plausible_strict": check.get("declare_plausible_strict"),
                "tdc_violations": check.get("tdc_violations"),
                "tdc_checkable": check.get("tdc_checkable"),
                "tdc_not_yet_applicable": check.get("tdc_not_yet_applicable"),
                "ldc_violations": check.get("ldc_violations"),
                "ldc_required": check.get("ldc_required"),
                "local_total": check.get("local_total"),
                "local_violations": check.get("local_violations"),
                "local_plausibility": check.get("local_plausibility"),
                "neighborhood_total": check.get("neighborhood_total"),
                "neighborhood_violations": check.get("neighborhood_violations"),
                "neighborhood_plausibility": check.get("neighborhood_plausibility"),
                "global_total": check.get("global_total"),
                "global_violations": check.get("global_violations"),
                "global_plausibility": check.get("global_plausibility"),
                "local_total_chain": check.get("local_total_chain"),
                "local_total_nonchain": check.get("local_total_nonchain"),
                "local_violations_chain": check.get("local_violations_chain"),
                "local_violations_nonchain": check.get("local_violations_nonchain"),
                "neighborhood_total_chain": check.get("neighborhood_total_chain"),
                "neighborhood_total_nonchain": check.get("neighborhood_total_nonchain"),
                "neighborhood_violations_chain": check.get("neighborhood_violations_chain"),
                "neighborhood_violations_nonchain": check.get("neighborhood_violations_nonchain"),
                "global_total_chain": check.get("global_total_chain"),
                "global_total_nonchain": check.get("global_total_nonchain"),
                "global_violations_chain": check.get("global_violations_chain"),
                "global_violations_nonchain": check.get("global_violations_nonchain"),
                "violated_detail_json": (
                    json.dumps(check.get("violated_detail"))
                    if check.get("violated_detail") is not None else None
                ),
            })

    df = pd.DataFrame(results)
    if output_csv:
        df.to_csv(output_csv, index=False)
    if df.empty:
        print("No counterfactuals to score.")
        return df

    def _print_rate(col, label):
        if col not in df.columns or not df[col].notna().any():
            return
        rate = df[col].dropna().astype(bool).mean()
        n = df[col].notna().sum()
        print(f"\n{label}: {rate:.1%} over {n} CFs")
        if "used_expulsion" in df.columns:
            for regime, sub in df.groupby(df["used_expulsion"].fillna(False)):
                regime_label = "expulsion-regime" if regime else "clean-convergence"
                sub_vals = sub[col].dropna()
                if len(sub_vals):
                    r = sub_vals.astype(bool).mean()
                    print(f"  {regime_label:>18}: {r:.1%} plausible (n={len(sub_vals)})")

    if df["declare_plausible"].notna().any():
        _print_rate("declare_plausible_ldc_only", "LDC-only plausible rate")
        _print_rate("declare_plausible_tdc_only", "TDC-only plausible rate")
        _print_rate("declare_plausible_strict", "Plausible rate (zero TDC and zero LDC violations)")

    for col, label in [
        ("local_plausibility", "Local plausibility (constraints touching the changed activity)"),
        ("neighborhood_plausibility",
         "Neighborhood plausibility (changed activity vs. unchanged prefix)"),
        ("global_plausibility", "Global plausibility (constraints unrelated to the changed activity)"),
    ]:
        if col in df.columns and df[col].notna().any():
            mean_val = df[col].dropna().mean()
            n = df[col].notna().sum()
            print(f"\n{label}: mean={mean_val:.1%} over {n} CFs")
        else:
            print(f"\nNo {col} computed.")

    if output_csv:
        print(f"Saved: {output_csv}")
    return df


def main():
    p = argparse.ArgumentParser(description="Declare plausibility of counterfactual explanations (TDC and LDC mined with iBCM from the training log).")
    p.add_argument("--train-csv", required=True, help="Training log. The constraints are mined from it.")
    p.add_argument("--test-csv", required=True, help="Test log the explanations refer to.")
    p.add_argument("--explanations-dir", default=None, help="Folder with the case_*.json files of a run.")
    p.add_argument("--output-csv", default="declare_plausibility.csv", help="One row per counterfactual.")
    p.add_argument("--export-constraints-dir", default=None, help="If set, write the mined constraints as tdc.csv and ldc.csv.")
    args = p.parse_args()
    score(args.train_csv, args.test_csv, args.explanations_dir, args.output_csv, args.export_constraints_dir)


if __name__ == "__main__":
    main()
