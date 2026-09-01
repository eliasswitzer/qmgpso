#!/bin/bash
#SBATCH --job-name=qmgpso_combo_experiment
#SBATCH --account=def-bmombuki
#SBATCH --array=0-1979%50
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module purge
module load python/3.11 scipy-stack/2024a

VENV="$HOME/envs/qmgpso"
if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --no-index --upgrade pip
    pip install --no-index numpy tqdm matplotlib scipy
else
    source "$VENV/bin/activate"
fi

export MPLBACKEND=Agg

cd "$SLURM_SUBMIT_DIR"

RESULTS_DIR=results/qmgpso_combo_experiment
mkdir -p "$RESULTS_DIR" logs

# fallback to temp dir if script is ever run outside SLURM for local testing
: "${SLURM_TMPDIR:=$(mktemp -d)}"

PROBLEMS=(FDA1 ZJZ FDA2 F5 F6 F7 DIMP1 DF4 DF5 DF6 FDA4)

NT=(10 10 10 1 20)
TT=(10 25 50 10 10)

QSTRAT=(adaptive adaptive adaptive pcx pcx pcx)
QGUIDE=(n r t n r t)

ARCHIVE=(cl re h2 h5 h10 hd)

NUM_RUNS=5
ITERATIONS=1000
PARTICLES=100
QUANTUM_PROPORTION=0.5
ARCHIVE_SIZE=100
SEED_BASE=1000

N_PROBLEMS=${#PROBLEMS[@]}
N_SEV=${#NT[@]}
N_QPSO=${#QSTRAT[@]}
N_ARCH=${#ARCHIVE[@]}
TOTAL=$(( N_PROBLEMS * N_SEV * N_QPSO * N_ARCH ))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "Array index $SLURM_ARRAY_TASK_ID >= $TOTAL configs; nothing to do."
    exit 0
fi

# decode SLURM_ARRAY_TASK_ID to (problem, severity, qpso variant, archive)
idx=$SLURM_ARRAY_TASK_ID
arch_i=$(( idx % N_ARCH )); idx=$(( idx / N_ARCH ))
qpso_i=$(( idx % N_QPSO )); idx=$(( idx / N_QPSO ))
sev_i=$(( idx % N_SEV )); idx=$(( idx / N_SEV ))
prob_i=$(( idx % N_PROBLEMS ))

PROBLEM=${PROBLEMS[$prob_i]}
NT_VAL=${NT[$sev_i]}
TT_VAL=${TT[$sev_i]}
QS=${QSTRAT[$qpso_i]}
QG=${QGUIDE[$qpso_i]}
ARCH=${ARCHIVE[$arch_i]}

TAG="${PROBLEM}_nt${NT_VAL}_tt${TT_VAL}_qs-${QS}_qg-${QG}_arch-${ARCH}"
FINAL_OUT="${RESULTS_DIR}/${TAG}.jsonl"
echo "[task $SLURM_ARRAY_TASK_ID / $TOTAL] $TAG ($NUM_RUNS seeds, ${SLURM_CPUS_PER_TASK:-1} parallel)"

if [ -s "$FINAL_OUT" ] && [ "$(wc -l < "$FINAL_OUT")" -ge "$NUM_RUNS" ]; then
    echo "[task $SLURM_ARRAY_TASK_ID] $FINAL_OUT already has $NUM_RUNS+ lines, skipping."
    exit 0
fi

run_one() {
    local run_number=$1
    local seed=$(( SEED_BASE + SLURM_ARRAY_TASK_ID * NUM_RUNS + run_number ))
    local tmp_out="${SLURM_TMPDIR}/${TAG}_seed${seed}.jsonl"
    if [ -s "$tmp_out" ]; then
        return 0
    fi

    python main.py \
        -s "$seed" \
        -p "$PROBLEM" \
        -i "$ITERATIONS" \
        -tt "$TT_VAL" \
        -nt "$NT_VAL" \
        -np "$PARTICLES" \
        -qp "$QUANTUM_PROPORTION" \
        -qs "$QS" \
        -qg "$QG" \
        -st "$ARCH" \
        -as "$ARCHIVE_SIZE" \
        -m \
        2>> "${RESULTS_DIR}/${TAG}.err" \
        | grep '^RAW_METRICS_JSON:' | sed 's/^RAW_METRICS_JSON://' > "$tmp_out" || true
}

export -f run_one
export SLURM_TMPDIR RESULTS_DIR TAG PROBLEM ITERATIONS TT_VAL NT_VAL PARTICLES QUANTUM_PROPORTION QS QG ARCH ARCHIVE_SIZE SEED_BASE NUM_RUNS SLURM_ARRAY_TASK_ID

seq 0 $(( NUM_RUNS - 1 )) | xargs -n1 -P"${SLURM_CPUS_PER_TASK:-1}" -I{} bash -c 'run_one "$@"' _ {}

cat "${SLURM_TMPDIR}/${TAG}"_seed*.jsonl > "$FINAL_OUT" 2>/dev/null || true

n_done=$(wc -1 < "$FINAL_OUT" 2>/dev/null || echo 0)
echo "[task $SLURM_ARRAY_TASK_ID] done: $TAG ($n_done/$NUM_RUNS seeds captured -> $FINAL_OUT)"
if [ "$n_done" -lt "$NUM_RUNS" ]; then
    echo "[task $SLURM_ARRAY_TASK_ID] WARNING fewer lines than expected seeds -- check ${RESULTS_DIR}/${TAG}.err"
fi