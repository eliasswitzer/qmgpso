import numpy as np
import experiment_config as cfg
from metrics import _dominates_min, compute_reference_point, _to_minimization, hypervolume, normalize, variational_distance_t, spacing_t, maximum_spread_t

def build_true_pof_cache(problem_name, nt, tt, num_iterations, num_points=1000, num_points_3obj=900):
    """Precomputes, for one (problem, nn, tt), the sequence of (true_pof, ref_point_min, hv_true) at every environment change over num_iterations, independent of seed/combo/proportion"""
    problem_cls = cfg.PROBLEMS[problem_name]
    problem = problem_cls(tau_T=tt, n_T=nt)

    n_points = num_points if problem.num_objectives == 2 else num_points_3obj

    cache = []
    for iteration in range(num_iterations):
        problem.advance()
        if problem.has_changed():
            true_pof = problem.true_pareto_front(num_points=n_points)
            ref_point = compute_reference_point(true_pof, problem.is_minimization)
            pof_true_min = np.array([_to_minimization(s, problem.is_minimization) for s in true_pof])
            ref_point_min = _to_minimization(ref_point, problem.is_minimization)
            hv_true = hypervolume(pof_true_min, ref_point_min)
            cache.append(dict(true_pof=true_pof, ref_point=ref_point, hv_true=hv_true))
            problem.handle_change()
    return cache

def record_fast(is_minimization, pof_found, change_cache_entry):
    """Equivalent to a MetricsTracker.record() call but reuses a precomputed true POF cache entry (avoids recomputing the true hypervolume). Returns a dict with the same five oer-change values that MetricsTracker.record() would append."""
    pof_found = np.array(pof_found, dtype=float) if len(pof_found) else np.empty((0, len(is_minimization)))
    pof_true = change_cache_entry["true_pof"]
    ref_point = change_cache_entry["ref_point"]
    hv_true = change_cache_entry["hv_true"]

    ns = len(pof_found)
    if ns == 0 or len(pof_true) == 0:
        return dict(VD=0.0, S=0.0, MS=0.0, acc=0.0, NS=ns)

    normalized_found = normalize(pof_found, pof_true)
    normalized_true = normalize(pof_true, pof_true)

    vd = variational_distance_t(normalized_found, normalized_true)
    s = spacing_t(normalized_found)
    ms = maximum_spread_t(pof_found, pof_true)

    pof_found_min = np.array([_to_minimization(sol, is_minimization) for sol in pof_found])
    ref_point_min = _to_minimization(ref_point, is_minimization)
    hv_found = hypervolume(pof_found_min, ref_point_min)
    acc = abs(hv_true - hv_found)

    return dict(VD=vd, S=s, MS=ms, acc=acc, NS=ns)