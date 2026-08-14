import argparse
import glob
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

import experiment_config as cfg

def load_cell(results_dir, proportion, problem_name, nt, tt, combo_id, num_runs):
    prop_tag = f"q{int(proportion * 100)}"
    sub = os.path.join(results_dir, prop_tag, problem_name, f"nt{nt}_tt{tt}, combo_id")
    files = sorted(glob.glob(os.path.join(sub, "seed_*.pkl")))[:num_runs]
    if not files:
        return None

    runs = []
    for f in files:
        with open(f, "rb") as fh:
            runs.append(pickle.load(fh))

    min_len = min(len(r["VD"]) for r in runs)
    if min_len == 0:
        return None

    out = {}
    for pm in ["VD", "S", "MS", "acc", "NS"]:
        out[pm] = np.array(r[pm][:min_len] for r in runs)

    # derive stability
    acc = out["acc"]
    if min_len >= 2:
        stab = np.maximum(0.0, acc[:, :-1] - acc[:, 1:])
    else:
        stab = np.zeros((acc.shape[0], 0))
    out["stab"] = stab
    return out

def load_all(results_dir, proportions, problems, nt_tau_combos, combos, num_runs):
    data = {}
    for proportion in proportions:
        data[proportion] = {}
        for problem_name in problems:
            data[proportion][problem_name] = {}
            for (label, nt, tt) in nt_tau_combos:
                cell = {}
                for combo_id in combos:
                        loaded = load_cell(results_dir, proportion, problem_name, nt, tt, combo_id, num_runs)
                        if loaded is not None:
                            cell[combo_id] = loaded
                if cell:
                    data[proportion][problem_name][(label, nt, tt)] = cell
    return data

def compute_wins_losses(cell_data, pm, combos, alpha=cfg.ALPHA):
    present = [c for c in combos if c in cell_data and cell_data[c][pm].shape[1] > 0]
    if len(present) < 2:
        return {}, {}, 0

    num_changes = min(cell_data[c][pm].shape[1] for c in present)
    lower_is_better = cfg.BETTER_LESS_THAN[pm]

    wins = defaultdict(float)
    losses = defaultdict(float)

    for t in range(num_changes):
        samples = {c: cell_data[c][pm][:, t] for c in present}
        try:
            _, p_kw = kruskal(*samples.values())
        except ValueError:
            continue
        if not np.isfinite(p_kw) or p_kw >= alpha:
            continue

        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                xa, xb = samples[a], samples[b]
                try:
                    _, p_mw = mannwhitneyu(xa, xb, alternative="two_sided")
                except ValueError:
                    continue
                if not np.isfinite(p_mw) or p_mw >= alpha:
                    continue
                med_a, med_b = np.median(xa), np.median(xb)
                if med_a == med_b:
                    continue
                a_better = (med_a < med_b) if lower_is_better else (med_a > med_b)
                if a_better:
                    wins[a] += 1
                    losses[b] += 1
                else:
                    wins[b] += 1
                    losses[a] += 1

    wins_norm = {c: wins.get(c, 0.0) / num_changes for c in present}
    losses_norm = {c: losses.get(c, 0.0) / num_changes for c in present}
    return wins_norm, losses_norm, num_changes

def compute_all_results(data, combos):
    rows = []
    for proportion, by_problem in data.items():
        for problem_name, by_ntt in by_problem.items():
            for (label, nt, tt), cell_data in by_ntt.items():
                for pm in cfg.PERFOMANCE_MEASURES:
                    wins, losses, num_changes = compute_wins_losses(cell_data, pm, combos)
                    for combo_id in wins.keys() | losses.keys():
                        w = wins.get(combo_id, 0.0)
                        l = losses.get(combo_id, 0.0)
                        rows.append(dict(
                            proportion=proportion, problem=problem_name, nt_tau_label=label, nt=nt, tt=tt, pm=pm, combo_id=combo_id, wins=w, losses=l, diff=w-l
                        ))
    return pd.DataFrame(rows)

def rank_table(df, group_cols=()):
    """Aggregates wins/losses/diff by summing over everthing except group columns + combo id, then ranks combos within each group"""
    keys = list(group_cols) + ["combo_id"]
    agg = df.groupby(keys, as_index=False)[["wins", "losses", "diff"]].sum()

    if group_cols:
        agg["rank"] = agg.groupby(list(group_cols))["diff"].rank(ascending=False, method="min")
        agg = agg.sort_values(list(group_cols) + ["rank"])
    else:
        agg["rank"] = agg["diff"].rank(ascending=False, method="min")
        agg= agg.sort_values("rank")

    return agg.reset_index(drop=True)

def overall_table(df, proportion=None):
    """Overall wins/losses/diff/rank per combo (summed over all DMOPs, nt_tau combos, and performance measures), optionally for one proportion"""
    d = df if proportion is None else df[df.proportion == proportion]
    return rank_table(d, group_cols=())

def by_nt_tau_table(df, proportion=None):
    """Table displaying ranks per nt_tau combination"""
    d = df if proportion is None else df[df.proportion == proportion]
    return rank_table(d, group_cols=("nt_tau_label",))

def by_pm_table(df, proportion=None):
    """Table displaying ranks per performance measure"""
    d = df if proportion is None else df[df.proportion == proportion]
    return rank_table(d, group_cols=("pm",))

def by_archive_strategy_table(df, proportion=None):
    """Strips the combo_id down to archive_strategy and re-ranks, then sums diff contributions of all 6 QPSO variants sharing that archive strategy"""
    d = df if proportion is None else df[df.proportion == proportion]
    d = d.copy()
    d["archive_strategy"] = d["combo_id"].str.split("__").str[0]
    agg = d.groupby("archive_strategy", as_index=False)[["wins", "losses", "diff"]].sum()
    agg["rank"] = agg["diff"].rank(ascending=False, method="min")
    return agg.sort_values("rank").reset_index(drop=True)

def by_qpso_variant_table(df, proportion=None):
    """Strips the combo_id down to qpso variant and re-ranks, then sums diff contributions of all archive strategies sharing that QPSO variant"""
    d = df if proportion is None else df[df.proportion == proportion]
    d = d.copy()
    d["qpso_variant"] = d["combo_id"].str.split("__").str[1]
    agg = d.groupby("qpso_variant", as_index=False)[["wins", "losses", "diff"]].sum()
    agg["rank"] = agg["diff"].rank(ascending=False, method="min")
    return agg.sort_values("rank").reset_index(drop=True)

def combo_heatmap_table(df, proportion=None):
    """Full 6x6 archive_strategy x qpso_variant diff matrix (overall, summed over DMOPs/nt_tau/pm), useful for spotting interaction effects"""
    d = df if proportion is None else df[df.proportion == proportion]
    d = d.copy()
    d["archive_strategy"] = d["combo_id"].str.split("__").str[0]
    d["qpso_variant"] = d["combo_id"].str.split("__").str[1]
    pivot = d.groupby(["archive_strategy", "qpso_variant"])["diff"].sum().unstack()
    return pivot.reindex(index=cfg.ARCHIVE_STRATEGIES, columns=list(cfg.QPSO_VARIANTS.keys()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--num_runs", type=int, default=cfg.NUM_RUNS)
    parser.add_argument("--out-dir", default="analysis_out")
    parser.add_argument("--problems", default=",".join(cfg.PROBLEMS.keys()), help="Comma-separated subset of problem names")
    parser.add_argument("--nt-tau", default=",".join(l for l, _, _, in cfg.NT_TAU_COMBOS), help="Comma-separated subset of nt-tau labels")
    parser.add_argument("--combos", default=",".join(cfg.COMBOS.keys()), help="Comma-separated subset of combo ids of form (archive_qpsovariant)")
    parser.add_argument("--proportions", default=",".join(str(p) for p in cfg.QUANTUM_PROPORTIONS), help="Comma-separated quantum proportions, e.g. 0.5,0.1")
    args = parser.parse_args()

    problems = args.problems.split(",")
    nt_tau_labels = set(args.nt_tau.split(","))
    nt_tau_combos = [t for t in cfg.NT_TAU_COMBOS if t[0] in nt_tau_labels]
    combos = args.combos.split(",")
    proportions = [float(p) for p in args.proportions.split(",")]

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading raw results...")
    data = load_all(args.results_dir, proportions, problems, nt_tau_combos, combos, args.num_runs)

    print("Running significance tests...")
    df = compute_all_results(data, combos)
    df.to_csv(os.path.join(args.out_dir, "long_form_wins_losses.csv", index=False))

    for proportion in proportions:
        tag = f"q{int(proportion * 100)}"
        overall_table(df, proportion).to_csv(os.path.join(args.out_dir, f"overall_{tag}.csv"), index=False)
        by_nt_tau_table(df, proportion).to_csv(os.path.join(args.out_dir, f"overall_{tag}.csv"), index=False)
        by_pm_table(df, proportion).to_csv(os.path.join(args.out_dir, f"overall_{tag}.csv"), index=False)
        by_archive_strategy_table(df, proportion).to_csv(os.path.join(args.out_dir, f"overall_{tag}.csv"), index=False)
        by_qpso_variant_table(df, proportion).to_csv(os.path.join(args.out_dir, f"overall_{tag}.csv"), index=False)
        combo_heatmap_table(df, proportion).to_csv(os.path.join(args.out_dir, f"overall_{tag}.csv"), index=False)

        print(f"\n===Overall ranking (quantum proportion={proportion}) ===")
        print(overall_table(df, proportion).to_string(index=False))

    print(f"\nAll tables written to {args.out_dir}")

if __name__ == "__main__":
    main()