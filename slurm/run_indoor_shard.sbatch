#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -J nmm-indoor-shard
#SBATCH -o slurm/logs/indoor_shard_%A_%a.out
#SBATCH -e slurm/logs/indoor_shard_%A_%a.err
#
# required env: REPO_DIR, SHARDS_ROOT, RUNS_PER_SHARD
# optional: GLOBAL_SEED, INDOOR_NX, INDOOR_NZ, INDOOR_PDE_ITERS

set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR not set}"
cd "$REPO_DIR"
# shellcheck source=/dev/null
source "${REPO_DIR}/slurm/inc_env.sh"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

SHARDS_ROOT="${SHARDS_ROOT:?SHARDS_ROOT not set}"
RUNS_PER_SHARD="${RUNS_PER_SHARD:?RUNS_PER_SHARD not set}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
INDOOR_NX="${INDOOR_NX:-240}"
INDOOR_NZ="${INDOOR_NZ:-100}"
INDOOR_PDE_ITERS="${INDOOR_PDE_ITERS:-15000}"
TASK="${SLURM_ARRAY_TASK_ID:?not running under array?}"
RUN_START=$((TASK * RUNS_PER_SHARD))
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "stage=indoor_shard job=${SLURM_JOB_ID:-} task=${TASK} host=$(hostname) utc=${TS}"
echo "repo=${REPO_DIR} nx=${INDOOR_NX} nz=${INDOOR_NZ} iters=${INDOOR_PDE_ITERS} run_start=${RUN_START}"

corpus_run_py -m indoor.apps.build_corpus_shard \
  --out "$SHARDS_ROOT" \
  --run-start "$RUN_START" \
  --run-count "$RUNS_PER_SHARD" \
  --shard-id "$TASK" \
  --global-seed "$GLOBAL_SEED" \
  --nx "$INDOOR_NX" \
  --nz "$INDOOR_NZ" \
  --pde-iters "$INDOOR_PDE_ITERS"

echo "stage=indoor_shard task=${TASK} finished utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
