import numpy as np

def variational_distance_t(pof_found, pof_true):
    """
    VD(t) estimates how close the found pareto front (PF*) is to the true pareto front (PF') by averaging, over each solution in PF*,
    the squared Euclidean distance to its nearest member of PF'. Prior knowledge and a normalized PF* are required.
    """
    pof_found = np.array(pof_found, dtype=float)
    pof_true = np.array(pof_true, dtype=float)
    n = len(pof_found)
    if n == 0 or len(pof_true) == 0:
        return 0.0
    d = 0.0
    for solution in pof_found:
        dists = np.sqrt(np.sum((pof_true - solution)** 2, axis=1))
        d += np.min(dists) ** 2

    return float(np.sqrt(n * d) / n)

def variational_distance(vd_t_values):
    """
    Averages VD(t) over the number of environment changes 
    """
    if not vd_t_values:
        return 0.0
    return float(np.mean(vd_t_values))

def number_of_nd_solutions(pof_found):
    """
    NS is the number of non-dominated solutions found in PF*. It is cheap to compute but says nothing about solution quality on its own 
    and is to be used alongside other measures.
    """
    return len(pof_found)

def spacing_t(pof_found):
    """
    S(t) measures how evenly the solutions in PF* are distributed, using the minimum sum of absolute per-objective differences ("Manhattan"
    nearest neighbor distance) for each solution, then taking the standard deviation of those distances. PF* should be normalized. A lower
    S value indicates a more even spread.
    """
    pof_found = np.array(pof_found, dtype=float)
    n = len(pof_found)
    if n <= 1:
        return 0.0

    d = np.zeros(n)
    for i in range(n):
        diffs = np.sum(np.abs(pof_found - pof_found[i]), axis=1)
        diffs[i] = np.inf
        d[i] = np.min(diffs)

    d_bar = np.mean(d)
    return float(np.sqrt(np.sum((d - d_bar) ** 2) / (n-1)))

def spacing(s_t_values):
    """
    Averaeges S(t) over the number of environment changes
    """
    if not s_t_values:
        return 0.0
    return float(np.mean(s_t_values))

def maximum_spread_t(pof_found, pof_true):
    """
    MS(t) measures how well PF* covers PF' by comparing, per objective, the overlap between range spanned by PF* and the range spanned by PF'.
    A high MS(t) value indicates good coverage/spread relative to the true front. Uses unnormalized values.
    """
    pof_found = np.array(pof_found, dtype=float)
    pof_true = np.array(pof_true, dtype=float)
    if len(pof_found) == 0 or len(pof_true) == 0:
        return 0.0

    num_objectives = pof_found.shape[1]
    pof_found_max, pof_found_min = pof_found.max(axis=0), pof_found.min(axis=0)
    pof_true_max, pof_true_min = pof_true.max(axis=0), pof_true.min(axis=0)

    total = 0.0
    for k in range(num_objectives):
        extent = pof_true_max[k] - pof_true_min[k]
        if extent == 0:
            continue
        overlap = min(pof_found_max[k], pof_true_max[k]) - max(pof_found_min[k], pof_true_min[k])
        total += (overlap / extent) ** 2

    return float(np.sqrt(total / num_objectives))

def maximum_spread(ms_t_values):
    """
    Averages MS(t) over the number of environment changes
    """
    if not ms_t_values:
        return 0.0
    return float(np.mean(ms_t_values))

def _dominates_min(a,b):
    """
    True if point a Pareto-dominates point b, assuming minimization in every objective
    """
    return np.all(a <= b) and np.any(a < b)

def _filter_nondominated_min(points):
    """
    Removes dominated points from a point set, assuming minimization in every objective (vectorized via NumPy broadcasting)
    """
    points = np.array(points, dtype=float)
    n = len(points)
    if n == 0:
        return points
    # dominates[j, i] True if point j dominates point i (both <=, at least one <)
    le = np.all(points[:, None, :] <= points[None, :, :], axis=2)
    lt = np.any(points[:, None, :] < points[None, :, :], axis=2)
    dominates = le & lt
    np.fill_diagonal(dominates, False)
    is_dominated = dominates.any(axis=0)
    return points[~is_dominated]

def hypervolume(points, ref_point):
    """
    Computes the hypervolume dominated by a set of points with respect to ref_point, assuming minimization in every objective and that ref_point
    is dominated by every point.
    """
    points = np.array(points, dtype=float)
    ref_point = np.array(ref_point, dtype=float)
    if points.size == 0:
        return 0.0

    points = _filter_nondominated_min(points)
    num_objectives = points.shape[1]

    if num_objectives == 1:
        return max(0.0, float(ref_point[0] - np.min(points[:, 0])))

    order = np.argsort(points[:, -1])
    points = points[order]

    hv = 0.0
    prev_bound = ref_point[-1]
    for i in range(len(points)):
        height = prev_bound - points[i, -1]
        if height > 0:
            sub_points = points[i:, :-1]
            hv += height * hypervolume(sub_points, ref_point[:-1])
        prev_bound = points[i, -1]

    return float(hv)

def _to_minimization(vector, is_minimization):
    """
    Converts any objective that is being maximized so that metrics can assume minimization
    """
    vector = np.array(vector, dtype=float)
    return np.array([v if minimize else -v for v, minimize in zip(vector, is_minimization)])

def compute_reference_point(pof_true, is_minimization, padding=0.1):
    """
    Computes a hypervolume reference point that is dominated by every solution in PF', with a small padding beyond worst observed values so that
    boundary solutions still contribute volume. Hypervolume is the area of the search space bounded by the Pareto-optimal front.
    """
    pof_true = np.array(pof_true, dtype=float)
    ranges = pof_true.max(axis=0) - pof_true.min(axis=0)
    ranges = np.where(ranges == 0, 1.0, ranges)
    ref = np.zeros(pof_true.shape[1])
    for k in range(pof_true.shape[1]):
        if is_minimization[k]:
            ref[k] = pof_true[:, k].max() + padding * ranges[k]
        else:
            ref[k] = pof_true[:, k].min() + padding * ranges[k]
    return ref

def accuracy_t(pof_found, pof_true, ref_point, is_minimization):
    """
    acc_alt(t) is the absolute hypervolume difference (HVD) between the true PF' and the found PF* at time t. Objectives being maximized are
    internally negated so that hypervolume can assume minimization throughout. ref_point must be given in the original objective space and must
    be dominated by every solution in true POF.
    """
    pof_found = np.array(pof_found, dtype=float)
    pof_true = np.array(pof_true, dtype=float)

    pof_true_min = np.array([_to_minimization(solution, is_minimization) for solution in pof_true])
    ref_point_min = _to_minimization(ref_point, is_minimization)

    if len(pof_found) == 0:
        hv_found = 0.0
    else:
        pof_found_min = np.array([_to_minimization(solution, is_minimization) for solution in pof_found])
        hv_found = hypervolume(pof_found_min, ref_point_min)

    hv_true = hypervolume(pof_true_min, ref_point_min)

    return abs(hv_true - hv_found)

def accuracy(acc_t_values):
    """
    Averages acc_alt(t) over the number of environment changes
    """
    if not acc_t_values:
        return 0.0
    return float(np.mean(acc_t_values))

def stability_t(acc_t, acc_t_minus_1):
    """
    stab(t) quantifies how well the algorithm recovers after an environment change by measuring the drop in accuracy relative to the previous environment.
    A low stab value indicates good performance (i.e. accuracy did not degrade very much after the change)
    """
    return max(0.0, acc_t_minus_1 - acc_t)

def stability(acc_t_values):
    """
    Averages stab(t) over the number of environment changes minus 1 (first environment change has no predecessor to compare against)
    """
    if len(acc_t_values) < 2:
        return 0.0
    stabs = [stability_t(acc_t_values[i], acc_t_values[i-1]) for i in range(1, len(acc_t_values))]
    return float(np.mean(stabs))

def normalize(fitnesses, pof_true):
    """
    Normalizes fitnesses into the range spanned by the true POF (PF') per-objective min/max as required prior to computing VD and S
    """
    fitnesses = np.array(fitnesses, dtype=float)
    pof_true = np.array(pof_true, dtype=float)
    if fitnesses.size == 0:
        return fitnesses
    f_min = pof_true.min(axis=0)
    f_max = pof_true.max(axis=0)
    extent = np.where((f_max - f_min) == 0, 1.0, f_max - f_min)
    return (fitnesses - f_min) / extent

class MetricsTracker:
    """
    Accumulates and records the six performance measures described in the Jocko 2022 paper
    """
    def __init__(self, is_minimization):
        self.is_minimization = is_minimization
        self.vd_values = []
        self.s_values = []
        self.ms_values = []
        self.acc_values = []
        self.ns_values = []

    def record(self, pof_found, pof_true, ref_point=None):
        """
        Records all six performance measures for the current found POF against the current true POF.
        If ref point is omitted, one is derived from the true POF
        """
        pof_found = np.array(pof_found, dtype=float) if len(pof_found) else np.empty((0, len(self.is_minimization)))
        pof_true = np.array(pof_true, dtype=float)

        self.ns_values.append(number_of_nd_solutions(pof_found))

        if len(pof_found) == 0 or len(pof_true) == 0:
            self.vd_values.append(0.0)
            self.s_values.append(0.0)
            self.ms_values.append(0.0)
            self.acc_values.append(0.0)
            return

        normalized_pof_found = normalize(pof_found, pof_true)
        normalized_pof_true = normalize(pof_true, pof_true)

        self.vd_values.append(variational_distance_t(normalized_pof_found, normalized_pof_true))
        self.s_values.append(spacing_t(normalized_pof_found))
        self.ms_values.append(maximum_spread_t(pof_found, pof_true))

        if ref_point is None:
            ref_point = compute_reference_point(pof_true, self.is_minimization)
        self.acc_values.append(accuracy_t(pof_found, pof_true, ref_point, self.is_minimization))

    def summary(self):
        """Returns the six averaged performance measures"""
        return {
            'VD': variational_distance(self.vd_values),
            'NS': float(np.mean(self.ns_values)) if self.ns_values else 0.0,
            'S': spacing(self.s_values),
            'MS': maximum_spread(self.ms_values),
            'acc': accuracy(self.acc_values),
            'stab': stability(self.acc_values)
        }

    