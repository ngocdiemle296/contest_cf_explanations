import argparse
import contextlib
import csv
import io
import json
import os
import re
import sys
from pathlib import Path

from baseline.revised_plus.iBCM.iBCM import Constraint
from evaluation.declare_plausibility import check_declare_violations

KPI_THRESHOLD = -0.05
MAX_LISTED = 10


def load_prefix(test_csv, case_id, slice_idx):
    from contest.data.event_log_csv import load_event_log_csv
    with contextlib.redirect_stdout(io.StringIO()):
        cases = load_event_log_csv(test_csv, sample_n_cases=None)
    for c in cases:
        if str(c.case_id) == str(case_id):
            return [(str(e.activity), str(e.resource)) for e in c.events[:slice_idx]]
    raise SystemExit(f"Case {case_id} not found in {test_csv}")


def load_constraints(folder, label):
    def make(name, a, b):
        return Constraint(name, a, b if b else a, 0)

    with open(os.path.join(folder, "tdc.csv"), newline="") as f:
        tdc = {make(r["template"], r["a"], r["b"]) for r in csv.DictReader(f)}
    ldc = set()
    with open(os.path.join(folder, "ldc.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["label_activity"] == label:
                ldc.add(make(r["template"], r["a"], r["b"]))
    labels = {label: ldc} if ldc else {}
    return {"no_win": 1, "trace_constraints": tdc, "label_constraints": labels}


def original_recommendation(summary_text):
    m = re.search(r"Original Black-Box Recommendation:\s*\((.+?),\s*([^,()]+)\)\s*$", summary_text, re.M)
    if not m:
        raise SystemExit("Could not read the original recommendation from summary_text.")
    return m.group(1).strip(), m.group(2).strip()


def cf_recommendation(summary_text):
    m = re.search(r"Counterfactual Recommendation:\s*\((.+?),\s*([^,()]+)\)\s*$", summary_text, re.M)
    if not m:
        raise SystemExit("Could not read the counterfactual recommendation from summary_text.")
    return m.group(1).strip(), m.group(2).strip()


SENTINELS = {"START", "END"}
COUNT_RULES = {"exactly": "exactly once", "exactly2": "exactly twice", "existence3": "at least three times"}


def reason(v, present, counts):
    t, a, b = v["template"], v["a"], v["b"]
    if t == "absence":
        return f"{a} should not occur, but it does"
    if t in COUNT_RULES:
        n = counts.get(a, 0)
        if n == 0:
            return f"{a} should occur {COUNT_RULES[t]}, but it is missing"
        return f"{a} should occur {COUNT_RULES[t]}, but it occurs {n} time{'s' if n != 1 else ''}"
    if t == "init":
        return f"the history should start with {a}"
    if t == "last":
        return f"the history should end with {a}"
    miss = [x for x in (a, b) if x not in present]
    if len(miss) == 2:
        return f"needs both {a} and {b}, both are missing"
    if miss:
        return f"needs both {a} and {b}, {miss[0]} is missing"
    return f"{a} and {b} are both present, but not in the required order"


def steps_text(steps):
    steps = [str(s) for s in steps]
    if len(steps) == 1:
        return f"step {steps[0]}"
    return "steps " + ", ".join(steps[:-1]) + " and " + steps[-1]


def pair(p):
    return f"{p[0]}, {p[1]}"


def print_table(prefix, cf_prefix, rec_step, orig_rec, cf_rec):
    rows = []
    changed = [i for i in range(len(prefix)) if prefix[i] != cf_prefix[i]]
    first = changed[0] if changed else len(prefix)
    if first > 0:
        label = "1" if first == 1 else f"1 - {first}"
        rows.append((label, "Identical", "Identical"))
    for i in range(first, len(prefix)):
        o, c = prefix[i], cf_prefix[i]
        mark = " *" if o != c else ""
        rows.append((str(i + 1), pair(o), pair(c) + mark))
    rows.append((f"{rec_step} (Rec.)", pair(orig_rec), pair(cf_rec)))
    w0 = max(len("Step"), *(len(r[0]) for r in rows))
    w1 = max(len("Original"), *(len(r[1]) for r in rows))
    w2 = max(len("Counterfactual"), *(len(r[2]) for r in rows))
    line = f"  {'-' * w0}   {'-' * w1}   {'-' * w2}"
    print(f"  {'Step':<{w0}}   {'Original':<{w1}}   {'Counterfactual':<{w2}}")
    print(line)
    for r in rows:
        print(f"  {r[0]:<{w0}}   {r[1]:<{w1}}   {r[2]:<{w2}}")
    print(line)
    print("  * changed event")


def report_cf(unit, cf, prefix, constraints_dir):
    rec_step = unit["explains_step"]
    orig_rec = original_recommendation(cf["summary_text"])
    cf_rec = cf_recommendation(cf["summary_text"])

    cf_prefix = list(prefix)
    act_steps, res_steps = [], []
    for ev in cf.get("changed_events", []):
        t = int(ev["t"])
        cf_prefix[t - 1] = (ev["activity_to"], ev["resource_to"])
        if ev["activity_from"] != ev["activity_to"]:
            act_steps.append(t)
        if ev["resource_from"] != ev["resource_to"]:
            res_steps.append(t)
    feats = cf.get("changed_features", [])

    print_table(prefix, cf_prefix, rec_step, orig_rec, cf_rec)
    if feats:
        print("\n  Changed attributes:")
        for f in feats:
            print(f"    {f['feature']}: {f['original']} -> {f['counterfactual']}")

    parts = []
    if act_steps:
        parts.append(f"the activity at {steps_text(act_steps)}")
    if res_steps:
        parts.append(f"the resource at {steps_text(res_steps)}")
    changes = " and ".join(parts) if parts else "no event"
    attr = ""
    if feats:
        items = [f"{f['feature']} changing from {f['original']} to {f['counterfactual']}" for f in feats]
        attr = ", alongside " + (items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1])

    print("\n  Contrastive Justification:")
    print(f"    The model recommends ({pair(orig_rec)}) at step {rec_step}. Under the alternative history "
          f"in the table, F would instead recommend ({pair(cf_rec)}). This history changes {changes}{attr}.")

    k0, k1 = cf["kpi_original"], cf["kpi_counterfactual"]
    d = cf["kpi_delta_pct"]
    trend = "improving" if d < 0 else "worsening"
    print("\n  Downstream Consequences:")
    text = (f"    The original recommendation is expected to take {k0:.1f} hours, while the counterfactual is "
            f"expected to take {k1:.1f} hours, with a difference of {abs(k0 - k1):.1f} hours "
            f"({trend} {abs(d) * 100:.1f}%).")
    if cf.get("changed_events"):
        text += (" As this counterfactual also revises the history before the recommended step, this "
                 "comparison reflects the black box's forecast under a different case trajectory rather "
                 "than a controlled intervention on the recommendation alone.")
    print(text)

    n_act, n_res = len(act_steps), len(res_steps)
    summary = f"It changes {n_act} activit{'y' if n_act == 1 else 'ies'}, {n_res} resource{'' if n_res == 1 else 's'}"
    if feats:
        summary += f" and {len(feats)} attribute{'' if len(feats) == 1 else 's'}"
    print(f"\n  {summary}.")

    if constraints_dir:
        constraints = load_constraints(constraints_dir, cf_rec[0])
        trace = cf_prefix + [cf_rec]
        changed_acts = {ev["activity_from"] for ev in cf.get("changed_events", [])}
        changed_acts |= {ev["activity_to"] for ev in cf.get("changed_events", [])}
        changed_acts |= {orig_rec[0], cf_rec[0]}
        first = min((int(ev["t"]) for ev in cf.get("changed_events", [])), default=len(prefix) + 1)
        check = check_declare_violations(trace, constraints, label=cf_rec[0],
                                         changed_activities=changed_acts,
                                         prefix_activities_before_change=[a for a, _ in prefix[: first - 1]])
        print("\n  Declare plausibility:")
        if not check:
            print("    No checkable constraint for this history.")
        else:
            n_tdc, n_ldc = check["tdc_checkable"], check["ldc_required"]
            v_tdc, v_ldc = check["tdc_violations"], check["ldc_violations"]
            total = n_tdc + n_ldc
            if v_tdc + v_ldc == 0:
                print(f"    Satisfies all {total} checkable Declare constraints ({n_tdc} TDC + {n_ldc} LDC) "
                      f"mined from the training log: plausible.")
            else:
                print(f"    Violates {v_tdc + v_ldc} of {total} checkable Declare constraints "
                      f"({v_tdc} of {n_tdc} TDC, {v_ldc} of {n_ldc} LDC): not plausible.")
                detail = sorted(check.get("violated_detail", []),
                                key=lambda v: (v["source"] != "tdc", v["template"], str(v["a"]), str(v["b"])))
                acts = [a for a, _ in trace if a not in SENTINELS]
                seen = {"tdc": set(acts), "ldc": set(acts[:-1])}
                counts = {"tdc": {}, "ldc": {}}
                for a in acts:
                    counts["tdc"][a] = counts["tdc"].get(a, 0) + 1
                for a in acts[:-1]:
                    counts["ldc"][a] = counts["ldc"].get(a, 0) + 1
                missing = {}
                for v in detail:
                    for a in {v["a"], v["b"]}:
                        if a not in seen[v["source"]] and v["template"] != "absence":
                            missing[a] = missing.get(a, 0) + 1
                if missing:
                    # These activities come from the LDC of the recommended activity: in every
                    # training history followed by it, they occur earlier (TDC rules are only
                    # checked when their activities are present, so they never add to this list).
                    share = "Most" if v_ldc > v_tdc else "Some"
                    print(f"    {share} broken rules come from the recommended activity {cf_rec[0]}. In the training log, "
                          f"{cf_rec[0]} only follows histories that contain the activities below, and this history does not.")
                    for a, n in sorted(missing.items(), key=lambda x: -x[1]):
                        print(f"      {a}, in {n} of the broken rules")
                print(f"    Broken rules{' (first ' + str(MAX_LISTED) + ')' if len(detail) > MAX_LISTED else ''}:")
                for v in detail[:MAX_LISTED]:
                    b = "" if v["a"] == v["b"] else f", {v['b']}"
                    why = reason(v, seen[v["source"]], counts[v["source"]])
                    print(f"      {v['source'].upper()} {v['template']}({v['a']}{b}): {why}")
                if len(detail) > MAX_LISTED:
                    print(f"      ... and {len(detail) - MAX_LISTED} more")
                print("    TDC rules hold in every training trace. LDC rules hold in every training history "
                      f"followed by {cf_rec[0]}.")


def find_unit(output_dir, case_id, milestone):
    pattern = f"case_{case_id}_step_*_m{round(float(milestone) * 100)}pct_*.json"
    files = sorted(Path(output_dir, "explanations").glob(pattern))
    if not files:
        raise FileNotFoundError(f"No result for case {case_id} at milestone {milestone} in {output_dir}/explanations")
    return str(files[0])


def report(unit_json, test_csv, constraints=None, show_all=False):
    with open(unit_json) as f:
        unit = json.load(f)
    cfs = unit.get("counterfactuals", [])
    slice_idx = unit["slice_idx"]

    print(f"\nCase {unit['case_id']} at milestone {unit['milestone'] * 100:.0f}% "
          f"({slice_idx} of {unit['trace_len']} events)")
    if not cfs:
        print("No counterfactual was found for this unit.")
        return

    orig_rec = original_recommendation(cfs[0]["summary_text"])
    print(f"After {slice_idx} events, the black box F recommends ({pair(orig_rec)}) "
          f"and forecasts {cfs[0]['kpi_original']:.1f} hours.")

    if show_all:
        chosen = cfs
        print(f"{len(cfs)} counterfactual(s) found.")
    else:
        improving = [c for c in cfs if c.get("kpi_delta_pct") is not None and c["kpi_delta_pct"] <= KPI_THRESHOLD]
        if not improving:
            print(f"{len(cfs)} counterfactual(s) found, none improves the KPI by at least 5%. "
                  f"Show them all with show_all=True (or --all).")
            return
        chosen = [min(improving, key=lambda c: c["cf_diagnostics"]["sparsity"])]
        print(f"{len(cfs)} counterfactual(s) found. Shown: the KPI-improving one with the fewest changes.")

    prefix = load_prefix(test_csv, unit["case_id"], slice_idx)
    for i, cf in enumerate(chosen, 1):
        if len(chosen) > 1:
            print(f"\n=== Counterfactual {i} of {len(chosen)} ===")
        print()
        report_cf(unit, cf, prefix, constraints)
    print()


def main():
    p = argparse.ArgumentParser(description="Print the counterfactual explanation of one unit in the format of the paper's worked example.")
    p.add_argument("unit_json", help="A case_*.json file from the explanations/ folder of a run.")
    p.add_argument("--test-csv", required=True, help="Test log of the run, used for the original history.")
    p.add_argument("--constraints", default=None,
                   help="Folder with tdc.csv and ldc.csv (e.g. declare_constraints/<log>) to check plausibility.")
    p.add_argument("--all", action="store_true",
                   help="Show every counterfactual of the unit, not only the representative one.")
    args = p.parse_args()
    report(args.unit_json, args.test_csv, args.constraints, args.all)


if __name__ == "__main__":
    main()
