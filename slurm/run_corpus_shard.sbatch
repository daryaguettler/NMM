#!/bin/bash
#SBATCH -J nmm-corpus-shard
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH -o slurm/logs/corpus_shard_%A_%a.out
#SBATCH -e slurm/logs/corpus_shard_%A_%a.err
#
# one array task = one shard (disjoint run_id range).
# required env (set by submit_particle_corpus.sh or export before sbatch):
#   REPO_DIR, SHARDS_ROOT, RUNS_PER_SHARD
# optional: GLOBAL_SEED, PARTICLES, DURATION_HOURS
#
# override account/partition on submit, e.g.:
#   sbatch -A YOUR_ACCOUNT -p YOUR_PARTITION --array=0-9 slurm/run_corpus_shard.sbatch

set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR not set}"
cd "$REPO_DIR"

export PYTHONPATH="${PYTHONPATH:-src}"

SHARDS_ROOT="${SHARDS_ROOT:?SHARDS_ROOT not set}"
RUNS_PER_SHARD="${RUNS_PER_SHARD:?RUNS_PER_SHARD not set}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
PARTICLES="${PARTICLES:-300}"
TASK="${SLURM_ARRAY_TASK_ID:?not running under array?}"
RUN_START=$((TASK * RUNS_PER_SHARD))

CMD=(uv run python -m particle_sim.apps.build_corpus_cluster
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
"${CMD[@]}"
