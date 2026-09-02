#!/bin/bash
#SBATCH --job-name=analyze_qmgpso
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
mkdir -p logs analysis_output

python analyze_results.py \
    --results-dir results/qmgpso_combo_experiment \
    --out-dir analysis_output \
    --workers "$SLURM_CPUS_PER_TASK"