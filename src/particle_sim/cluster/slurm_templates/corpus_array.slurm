#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -J pcf-corpus
#SBATCH --array=0-9
#SBATCH -o slurm/logs/corpus_%A_%a.out
#SBATCH -e slurm/logs/corpus_%A_%a.err
#
# same scale as slurm/run_corpus_shard.sbatch (see comments there).

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR}"
cd "$REPO_DIR"
# shellcheck source=/dev/null
source "${REPO_DIR}/slurm/inc_env.sh"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
RUNS_PER=50
TASK="${SLURM_ARRAY_TASK_ID:?}"
START=$((TASK * RUNS_PER))
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "stage=corpus_shard template job=${SLURM_JOB_ID:-} task=${TASK} host=$(hostname) utc=${TS} repo=${REPO_DIR}"
corpus_run_py -m particle_sim.apps.build_corpus_cluster \
  --out corpus/v0_particle_shards \
  --run-start "${START}" \
  --run-count "${RUNS_PER}" \
  --shard-id "${TASK}" \
  --particles 300
echo "stage=corpus_shard template task=${TASK} finished utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
