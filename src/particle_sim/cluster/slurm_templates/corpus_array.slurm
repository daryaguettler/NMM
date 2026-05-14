#!/bin/bash
#SBATCH -J pcf-corpus
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p mit_normal
#SBATCH --array=0-9
#SBATCH -c 8
#SBATCH --time=04:00:00
#SBATCH -o logs/corpus_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH=src
RUNS_PER=50
START=$((SLURM_ARRAY_TASK_ID * RUNS_PER))
uv run python -m particle_sim.apps.build_corpus_cluster \
  --out corpus/v0_particle_shards \
  --run-start "${START}" \
  --run-count "${RUNS_PER}" \
  --shard-id "${SLURM_ARRAY_TASK_ID}" \
  --particles 300
