#!/bin/bash
#SBATCH --job-name=qmgpso_experiment
#SBATCH --account=def-bmombuki
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=runlogs/%x_%j.out

module load python/3.11

source "$HOME/envs/qmgpso/bin/activate"

cd "$HOME/projects/qmgpso" || exit 1

python experiment_run.py \
    --results_dir "$SCRATCH/qmgpso_results" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --num_runs 5