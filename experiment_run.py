import argparse
import os
import pickle
import random
import time
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from qmgpso import QMGPSO
import experiment_config as cfg
from experiment_helpers import build_true_pof_cache, record_fast

def cell_path(results_dir, proportion, problem_name, nt, tt, combo_id, seed):
    prop_tag = f"q{int(proportion * 100)}"
    sub = os.path.join(
        results_dir, prop_tag, problem_name, f"nt{nt}_tt{tt}", combo_id
    )
    os.makedirs(sub, exist_ok=True)
    return os.path.join(sub, f"seed_{seed}.pkl")

def true_pof_cache_path(results_dir, problem_name, nt, tt, num_iterations):
    sub = os.path.join(results_dir, "true_pof_cache")
    os.makedirs(sub, exist_ok=True)
    return os.path.join(sub, f"{problem_name}_nt{nt}_tt{tt}_it{num_iterations}.pkl")

def get_or_build_true_pof_cache(results_dir, problem_name, nt, tt, num_iterations):
    path = true_pof_cache_path(results_dir, problem_name, nt, tt, num_iterations)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    cache = build_true_pof_cache(problem_name, nt, tt, num_iterations)
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    return cache

def run_single(problem_name, nt, tt, combo_id, proportion, seed, true_pof_cache, num_particles=cfg.NUM_PARTICLES, num_iterations=cfg.NUM_ITERATIONS):
    """Runs one full QMGPSO optimization run and returns the raw per-environment-change performance measure lists"""
    np.random.seed(seed)
    random.seed(seed)
    combo = cfg.COMBOS[combo_id]
    problem_cls = cfg.PROBLEMS[problem_name]
    problem = problem_cls(tau_T=tt, n_T=nt)

    qmgpso = QMGPSO(
        num_particles=num_particles,
        search_bounds=problem.search_bounds,
        objective=problem.evaluate,
        num_objectives=problem.num_objectives,
        is_minimization=problem.is_minimization,
        quantum_proportion=proportion,
        archive_strategy=combo["archive_strategy"],
        quantum_strategy=combo["quantum_strategy"],
        quantum_guide=combo["quantum_guide"],
        **cfg.FIXED_PARAMS
    )

    qmgpso.initialize()

    vd_values, s_values, ms_values, acc_values, ns_values = [], [], [], [], []
    change_idx = 0

    for _ in range(num_iterations):
        qmgpso.step()
        problem.advance()
        if problem.has_changed():
            if change_idx < len(true_pof_cache):
                entry = true_pof_cache[change_idx]
                rec = record_fast(problem.is_minimization, qmgpso.archive.fitnesses, entry)
                vd_values.append(rec["VD"])
                s_values.append(rec["S"])
                ms_values.append(rec["MS"])
                acc_values.append(rec["acc"])
                ns_values.append(rec["NS"])
            change_idx += 1
            problem.handle_change()
            qmgpso.handle_change()

    return {
        "VD": vd_values,
        "S": s_values,
        "MS": ms_values,
        "acc": acc_values,
        "NS": ns_values
    }

_WORKER_CACHE_STORE = {}

def _worker(args):
    (problem_name, nt, tt, combo_id, proportion, seed, num_particles, num_iterations, results_dir) = args
    out_path = cell_path(results_dir, proportion, problem_name, nt, tt, combo_id, seed)
    if os.path.exists(out_path):
        return out_path, "skipped"

    cache_path = true_pof_cache_path(results_dir, problem_name, nt, tt, num_iterations)
    true_pof_cache = _WORKER_CACHE_STORE.get(cache_path)
    if true_pof_cache is None:
        true_pof_cache = get_or_build_true_pof_cache(results_dir, problem_name, nt, tt, num_iterations)
        _WORKER_CACHE_STORE[cache_path] = true_pof_cache

    t0 = time.time()
    result = run_single(problem_name, nt, tt, combo_id, proportion, seed, true_pof_cache, num_particles=num_particles, num_iterations=num_iterations)
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    return out_path, f"done in {time.time() - t0:.1f}s"

def build_task_list(results_dir, problems, nt_tau_combos, combos, proportions, num_runs, num_particles, num_iterations):
    tasks = []
    for proportion in proportions:
        for problem_name in problems:
            for (label, nt, tt) in nt_tau_combos:
                for combo_id in combos:
                    for seed in range(num_runs):
                        tasks.append((problem_name, nt, tt, combo_id, proportion, seed, num_particles, num_iterations, results_dir))
    return tasks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--num_runs", type=int, default=cfg.NUM_RUNS)
    parser.add_argument("--num_iterations", type=int, default=cfg.NUM_ITERATIONS)
    parser.add_argument("--num_particles", type=int, default=cfg.NUM_PARTICLES)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
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

    os.makedirs(args.results_dir, exist_ok=True)

    print("Precomputing true-POF hypervolume caches shared across all combos/seeds/proportions...")
    t0 = time.time()
    for problem_name in problems:
        for (label, nt, tt) in nt_tau_combos:
            get_or_build_true_pof_cache(args.results_dir, problem_name, nt, tt, args.num_iterations)
    print(f"Cache build done in {time.time() - t0:.1f}s")

    tasks = build_task_list(args.results_dir, problems, nt_tau_combos, combos, proportions, args.num_runs, args.num_particles, args.num_iterations)
    print(f"Total task cells: {len(proportions)} proportions x {len(problems)} problems x {len(nt_tau_combos)} nt-tau x {len(combos)} combos x {args.num_runs} runs")

    completed, skipped = 0, 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers = args.workers) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            path, status = fut.result()
            if status == "skipped":
                skipped += 1
            else:
                completed += 1
            if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                elapsed = time.time() - t_start
                print(f"[{i}/{len(tasks)}] completed={completed} skipped={skipped} elpased={elapsed:.1f}s")

    print("Done!")

if __name__ == "__main__":
    main()