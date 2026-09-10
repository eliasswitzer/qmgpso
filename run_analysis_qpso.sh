#!/bin/bash
#SBATCH --job-name=analyze_qpso_isolation
#SBATCH --account=def-bmombuki
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load python/3.11 scipy-stack/2024a
source "$HOME/envs/qmgpso/bin/activate"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

for QP in 50 10; do
    RESULTS_DIR="results/qpso_isolation_experiment/qp${QP}"
    OUT_DIR="analysis_output/qpso_isolation/qp${QP}"
    mkdir -p "$OUT_DIR"

    if [ -z "$(ls -A "$RESULTS_DIR"/*.jsonl 2>/dev/null)" ]; then
        echo "[warn] no .jsonl files found in $RESULTS_DIR, skipping qp${QP}"
        continue
    fi
    
    echo "=== Analyzing qp${QP} (${RESULTS_DIR}) ==="
    python analyze_results.py \
        --results-dir "$RESULTS_DIR" \
        --out-dir "$OUT_DIR" \
        --combo-fields quantum_strategy quantum_guide \
        --workers "$SLURM_CPUS_PER_TASK"
    echo "=== Done qp${QP} -> ${OUT_DIR} ==="
done