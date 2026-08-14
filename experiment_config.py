from problems import FDA1, ZJZ, FDA2, F5, F6, F7

ARCHIVE_STRATEGIES = ["cl", "re", "h2", "h5", "h10", "hd"]

QPSO_VARIANTS = {
    "QPSOn": dict(quantum_strategy="adaptive", quantum_guide="n"),
    "QPSOr": dict(quantum_strategy="adaptive", quantum_guide="r"),
    "QPSOt": dict(quantum_strategy="adaptive", quantum_guide="t"),
    "PCXn": dict(quantum_strategy="pcx", quantum_guide="n"),
    "PCXr": dict(quantum_strategy="pcx", quantum_guide="r"),
    "PCXt": dict(quantum_strategy="pcx", quantum_guide="t")
}

def build_combos():
    combos = {}
    for ar in ARCHIVE_STRATEGIES:
        for name, params in QPSO_VARIANTS.items():
            combo_id = f"{ar}_{name}"
            combos[combo_id] = dict(archive_strategy=ar, **params)
    return combos

COMBOS = build_combos()

QUANTUM_PROPORTIONS = [0.5, 0.1]

PROBLEMS = {
    "FDA1": FDA1,
    "ZJZ": ZJZ,
    "FDA2": FDA2,
    "F5": F5,
    "F6": F6,
    "F7": F7
}

NT_TAU_COMBOS = [
    ("medium_fast", 10, 10),
    ("medium_medium", 10, 25),
    ("medium_slow", 10, 50),
    ("big_fast", 1, 10),
    ("small_fast", 20, 10)
]

NUM_PARTICLES = 100
NUM_ITERATIONS = 1000
NUM_RUNS = 30 # independent runes per (per proportion, problem, nt_tau, combo) cell

FIXED_PARAMS = dict(
    w=0.6,
    c1=0.1,
    c2=0.02,
    c3=1.8,
    neighborhood_size=3,
    pcx_sigma1=0.1,
    pcx_sigma2=0.3,
    pcx_num_parents=3,
    archive_size=100,
    stability_guided=False
)

PERFOMANCE_MEASURES = ["VD", "S", "MS", "acc", "NS", "stab"]
BETTER_LESS_THAN = {
    "VD": True, # variational distance: closer to true POF is better
    "S": True, # spacing: more even sprtead is better (lower std)
    "MS": False, # maximum spread: higher coverage of true POF is better
    "acc": True, # hypervolume distance: smaller gap to true POF is better
    "NS": False, # number of non-dominated solutions: more is generally better
    "stab": True # stability: smaller accuracy drop after change is better
}

ALPHA = 0.05 # significance threshold for Kurskal-Wallis / Mann-Whitney U tests