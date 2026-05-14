#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -J nmm-corpus-shard
#SBATCH -o slurm/logs/corpus_shard_%A_%a.out
#SBATCH -e slurm/logs/corpus_shard_%A_%a.err
#
# one array task = one shard (disjoint run_id range).
# mem: default corpus uses ~6 zones * 300 particles (numpy); stays well under 8G.
# time: 12h default (mit_normal often rejects >12h). for ~50 full runs you may need
#   a longer queue or sbatch --time=... / see SLURM_TIME_SHARD in submit_particle_corpus.sh.
#   raise if you increase --runs-per-shard, PARTICLES, or DURATION_HOURS.
# required env (set by submit_particle_corpus.sh or export before sbatch):
#   REPO_DIR, SHARDS_ROOT, RUNS_PER_SHARD
# optional: GLOBAL_SEED, PARTICLES, DURATION_HOURS, NUM_SHARDS, TOTAL_RUNS (logging)
#
# override on submit if needed: sbatch --partition=... --array=0-9 ...

set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR not set}"
cd "$REPO_DIR"
# shellcheck source=/dev/null
source "${REPO_DIR}/slurm/inc_env.sh"

export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

SHARDS_ROOT="${SHARDS_ROOT:?SHARDS_ROOT not set}"
RUNS_PER_SHARD="${RUNS_PER_SHARD:?RUNS_PER_SHARD not set}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
PARTICLES="${PARTICLES:-300}"
TASK="${SLURM_ARRAY_TASK_ID:?not running under array?}"
RUN_START=$((TASK * RUNS_PER_SHARD))
_read_num_shards="${NUM_SHARDS:-}"
_read_total_runs="${TOTAL_RUNS:-}"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
_shard_ntasks="${NUM_SHARDS:-${SLURM_ARRAY_TASK_MAX:-?}}"
echo "stage=corpus_shard job=${SLURM_JOB_ID:-} array_job=${SLURM_ARRAY_JOB_ID:-} task=${TASK}/${_shard_ntasks} host=$(hostname) utc=${TS}"
echo "repo=${REPO_DIR} shards_root=${SHARDS_ROOT} num_shards=${_read_num_shards:-?} total_runs=${_read_total_runs:-?}"
echo "global_seed=${GLOBAL_SEED} particles=${PARTICLES} duration_hours=${DURATION_HOURS:-default}"

CMD=(-m particle_sim.apps.build_corpus_cluster
  --out "$SHARDS_ROOT"
  --run-start "$RUN_START"
  --run-count "$RUNS_PER_SHARD"
  --shard-id "$TASK"
  --global-seed "$GLOBAL_SEED"
  --particles "$PARTICLES")
if [[ -n "${DURATION_HOURS:-}" ]]; then
  CMD+=(--duration-hours "$DURATION_HOURS")
fi

echo "shard task=${TASK} run_start=${RUN_START} run_count=${RUNS_PER_SHARD}"
corpus_run_py "${CMD[@]}"
echo "stage=corpus_shard task=${TASK} finished utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
