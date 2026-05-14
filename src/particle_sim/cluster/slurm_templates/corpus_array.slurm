#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 24:00:00
#SBATCH -J pcf-corpus
#SBATCH --array=0-9
#SBATCH -o slurm/logs/corpus_%A_%a.out
#SBATCH -e slurm/logs/corpus_%A_%a.err
#
# same scale as slurm/run_corpus_shard.sbatch (see comments there).

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
