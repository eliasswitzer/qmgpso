import argparse
import glob
import json
import os
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

METRIC_DIRECTION = {
    "VD": True,
    "S": True,
    "MS": False,
    "acc": True,
    "stab": True,
    "NS": False,
}

ALL_METRICS = list(METRIC_DIRECTION.keys())

def load_records(results_dir):
    """Reads every *.jsonl file in results_dir and returns a flat list of the parsed raw metrics dicts."""
    records = []
    n_bad = 0
    files = sorted(glob.glob(os.path.join(results_dir, "*.jsonl")))
    if not files:
        raise SystemExit(f"No .jsonl files found in {results_dir}")
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                records.append(rec)

    if n_bad:
        print(f"[warn] skipped {n_bad} malformed lines across {len(files)} files")
    print(f"[info] loaded {len(records)} seed runs from {len(files)} files")
    return records

def build_groups(records, combo_fields):
    """Returns a list of per-seed arrays"""
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rec in records:
        try:
            key = (rec["problem"], rec["nt"], rec["tt"])
            combo_key = tuple(rec[f] for f in combo_fields)
        except KeyError as e:
            raise SystemExit(f"Record missing expected field {e}: {rec}")
        for m in ALL_METRICS:
            if m in rec and rec[m] is not None:
                groups[key][combo_key][m].append(rec[m])
    return groups

def analyze_job(job):
    """Returns dict with problem, nt, tt, pm and wins, losses and n_changes"""
    problem, nt, tt, pm, combo_data, alpha, min_samples = job
    minimize = METRIC_DIRECTION[pm]

    combos = list(combo_data.keys())
    if len(combos) < 2:
        return None

    nc = max((len(arr) for arrs in combo_data.values() for arr in arrs), default=0)
    if nc == 0:
        return None

    wins = defaultdict(int)
    losses = defaultdict(int)

    for t in range(nc):
        samples = {}
        for c in combos:
            vals = [arr[t] for arr in combo_data[c] if len(arr) > t]
            if len(vals) >= min_samples:
                samples[c] = vals
        valid = list(samples.keys())
        if len(valid) < 2:
            continue

        all_vals = [v for c in valid for v in samples[c]]
        if len(set(all_vals)) <= 1:
            continue

        try:
            _, p_kw = kruskal(*(samples[c] for c in valid))
        except ValueError:
            continue
        if not (p_kw < alpha):
            continue

        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                a, b = valid[i], valid[j]
                ga, gb = samples[a], samples[b]
                if len(set(ga + gb)) <= 1:
                    continue
                try:
                    _, p_mw = mannwhitneyu(ga, gb, alternative="two-sided")
                except ValueError:
                    continue
                if not (p_mw < alpha):
                    continue
                med_a, med_b = statistics.median(ga), statistics.median(gb)
                if med_a == med_b:
                    continue
                if minimize:
                    winner, loser = (a,b) if med_a < med_b else (b,a)
                else:
                    winner, loser = (a,b) if med_a > med_b else (b,a)
                wins[winner] += 1
                losses[loser] += 1

    return {
        "problem": problem,
        "nt": nt,
        "tt": tt,
        "pm": pm,
        "wins": dict(wins),
        "losses": dict(losses),
        "n_changes": nc
    }

def make_table(job_results, group_key_fn, combo_label_fn):
    """Creates pandas dataframe with wins, losses, diff and rank for each combo"""
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for jr in job_results:
        gk = group_key_fn(jr)
        nc = jr["n_changes"]
        combos = set(jr["wins"]) | set(jr["losses"])

        for c in jr.get("_all_combos", combos):
            agg[gk][c][2] += nc
        for c, w in jr["wins"].items():
            agg[gk][c][0] += w
        for c, l in jr["losses"].items():
            agg[gk][c][1] += l

    rows = []
    for gk, combo_map in agg.items():
        for combo, (w, l, nc) in combo_map.items():
            diff = w - l
            rows.append({
                **{f"g{i}": v for i, v in enumerate(gk)},
                "combo": combo_label_fn(combo),
                "Wins": w,
                "Losses": l,
                "Wins_norm": w/nc if nc else 0.0,
                "Losses_norm": l/nc if nc else 0.0,
                "Diff": diff,
                "n_changes": nc
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    group_cols = [c for c in df.columns if c.startswith("g")]
    if group_cols:
        df["Rank"] = df.groupby(group_cols)["Diff"].rank(ascending=False, method="min").astype(int)
        df = df.sort_values(group_cols + ["Rank"])
    else:
        df["Rank"] = df["Diff"].rank(ascending=False, method="min").astype(int)
        df = df.sort_values("Rank")

    return df

def print_table(df, title):
    print(f"\n=== {title} ===")
    if df.empty:
        print("(no significant results / no data)")
        return
    print(df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True, help="Directory of *.jsonl result files")
    parser.add_argument("--out-dir", default="analysis_output", help="Where to write CSV tables to")
    parser.add_argument("--combo-fields", nargs="+", default=["quantum_strategy", "quantum_guide", "archive_strategy"], help="Fields that define an combo to compare")
    parser.add_argument("--metrics", nargs="+", default=ALL_METRICS, choices=ALL_METRICS, help="Performance measures to include")
    parser.add_argument("--problems", nargs="+", default=None, help="Restrict to these problems")
    parser.add_argument("--severities", nargs="+", default=None, help="Restrict to these nn,tt combos")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--min-samples", type=int, default=2, help="Minimum seed samples a combo needs at a timepoint to be tested")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    records = load_records(args.results_dir)

    if args.problems:
        records = [r for r in records if r["problem"] in args.problems]
    if args.severities:
        wanted = {tuple(int(x) for x in s.split(",")) for s in args.severities}
        records = [r for r in records if (r["nt"], r["tt"]) in wanted]
    if not records:
        raise SystemExit("No records left after filtering.")

    groups = build_groups(records, args.combo_fields)
    print(f"[info] {len(groups)} (problem, nt, tt) severity combinations found")

    def combo_label(combo_key):
        return "_".join(str(x) for x in combo_key)

    jobs = []
    for (problem, nt, tt), combo_data_all in groups.items():
        all_combos = list(combo_data_all.keys())
        for pm in args.metrics:
            combo_data_pm = {c: combo_data_all[c].get(pm, []) for c in all_combos}
            combo_data_pm = {c: v for c, v in combo_data_pm.items() if v}
            if len(combo_data_pm) < 2:
                continue
            jobs.append((problem, nt, tt, pm, combo_data_pm, args.alpha, args.min_samples, all_combos))

    print(f"[info] {len(jobs)} (problem, nt, tt, pm) jobs to run across {len(args.combo_fields)}-field combos with {args.workers} worker(s)")

    results = []
    submit_jobs = [j[:-1] for j in jobs]
    all_combos_by_job = [j[-1] for j in jobs]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(analyze_job, j): i for i, j in enumerate(submit_jobs)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            r = fut.result()
            if r is not None:
                r["_all_combos"] = all_combos_by_job[i]
                results.append(r)
            done += 1
            if done % max(1, len(jobs) // 20) == 0:
                print(f"[info] {done}/{len(jobs)} jobs done")

    if not results:
        raise SystemExit("No jobs produced results -- check data/filters")

    os.makedirs(args.out_dir, exist_ok=True)

    overall = make_table(results, lambda r: (), combo_label)
    print_table(overall.drop(columns=[c for c in overall.columns if c.startswith("g")], errors="ignore"), "Overall wins/losses/diff/rank")
    overall.to_csv(os.path.join(args.out_dir, "overall.csv"), index=False)

    by_sev = make_table(results, lambda r: (r["nt"], r["tt"]), combo_label)
    by_sev = by_sev.rename(columns={"g0": "nt", "g1": "tt"})
    print_table(by_sev, "By severity (nt, tt) -- across all problems & metrics")
    by_sev.to_csv(os.path.join(args.out_dir, "by_severity.csv"), index=False)

    by_pm = make_table(results, lambda r: (r["pm"],), combo_label)
    by_pm = by_pm.rename(columns={"g0": "pm"})
    print_table(by_pm, "By performance measure -- across all problems & severities")
    by_pm.to_csv(os.path.join(args.out_dir, "by_metric.csv"), index=False)

    by_problem = make_table(results, lambda r: (r["problem"],), combo_label)
    by_problem = by_problem.rename(columns={"g0": "problem"})
    print_table(by_problem, "By benchmark problem -- across all severities & metrics")
    by_problem.to_csv(os.path.join(args.out_dir, "by_problem.csv"), index=False)

    print(f"\n[info] CSV tables written to {args.out_dir}/")
    best = overall.sort_values("Rank").iloc[0]
    print(f"\n[info] Best overall combo ({'+'.join(args.combo_fields)}): "
          f"{best['combo']}  (Diff={best['Diff']}, Wins={best['Wins']}, Losses={best['Losses']})")

if __name__ == "__main__":
    main()