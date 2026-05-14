#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -J nmm-micro-shard
#SBATCH -o slurm/logs/microclimate_shard_%A_%a.out
#SBATCH -e slurm/logs/microclimate_shard_%A_%a.err
#
# required env: REPO_DIR, SHARDS_ROOT, RUNS_PER_SHARD
# optional: GLOBAL_SEED, NUM_SHARDS, TOTAL_RUNS, MICRO_NX, MICRO_NZ, MICRO_PDE_ITERS

set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR not set}"
cd "$REPO_DIR"
# shellcheck source=/dev/null
source "${REPO_DIR}/slurm/inc_env.sh"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

SHARDS_ROOT="${SHARDS_ROOT:?SHARDS_ROOT not set}"
RUNS_PER_SHARD="${RUNS_PER_SHARD:?RUNS_PER_SHARD not set}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
MICRO_NX="${MICRO_NX:-200}"
MICRO_NZ="${MICRO_NZ:-100}"
MICRO_PDE_ITERS="${MICRO_PDE_ITERS:-4000}"
TASK="${SLURM_ARRAY_TASK_ID:?not running under array?}"
RUN_START=$((TASK * RUNS_PER_SHARD))
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "stage=microclimate_shard job=${SLURM_JOB_ID:-} task=${TASK} host=$(hostname) utc=${TS}"
echo "repo=${REPO_DIR} nx=${MICRO_NX} nz=${MICRO_NZ} iters=${MICRO_PDE_ITERS} run_start=${RUN_START}"

corpus_run_py -m microclimate.apps.build_corpus_shard \
  --out "$SHARDS_ROOT" \
  --run-start "$RUN_START" \
  --run-count "$RUNS_PER_SHARD" \
  --shard-id "$TASK" \
  --global-seed "$GLOBAL_SEED" \
  --nx "$MICRO_NX" \
  --nz "$MICRO_NZ" \
  --pde-iters "$MICRO_PDE_ITERS"

echo "stage=microclimate_shard task=${TASK} finished utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
