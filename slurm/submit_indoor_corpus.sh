#!/bin/bash
#===============================================================
# submit_indoor_corpus.sh — Engaging: shard array for indoor PDE corpus
#
# 18 total cases: 3 T_hot × 3 T_outdoor × 2 window_open
#
# usage:
#   bash slurm/submit_indoor_corpus.sh
#   bash slurm/submit_indoor_corpus.sh --num-shards 3 --runs-per-shard 6
#   bash slurm/submit_indoor_corpus.sh --test
#   bash slurm/submit_indoor_corpus.sh --dry-run
#
# env (optional):
#   REPO_DIR, SHARDS_ROOT, INDOOR_NX, INDOOR_NZ, INDOOR_PDE_ITERS
#   CORPUS_PYTHON, SLURM_TIME_SHARD, SLURM_TIME_SHARD_TEST
#===============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR

for _need in \
  "$REPO_DIR/slurm/run_indoor_shard.sbatch" \
  "$REPO_DIR/slurm/inc_env.sh" \
  "$REPO_DIR/src/indoor"
do
  [[ -e "$_need" ]] || { echo "missing required path: $_need" >&2; exit 1; }
done

SHARDS_ROOT="${SHARDS_ROOT:-$HOME/orcd/scratch/nmm_indoor_shards}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
INDOOR_NX="${INDOOR_NX:-240}"
INDOOR_NZ="${INDOOR_NZ:-100}"
INDOOR_PDE_ITERS="${INDOOR_PDE_ITERS:-15000}"

NUM_SHARDS=3
RUNS_PER_SHARD=6   # 18 total cases / 3 shards
TEST_MODE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-shards)     NUM_SHARDS="$2";     shift ;;
    --runs-per-shard) RUNS_PER_SHARD="$2"; shift ;;
    --test)           TEST_MODE=1 ;;
    --dry-run)        DRY_RUN=1 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
  shift
done

if (( TEST_MODE )); then
  NUM_SHARDS=1
  RUNS_PER_SHARD=2
  INDOOR_NX="${TEST_INDOOR_NX:-60}"
  INDOOR_NZ="${TEST_INDOOR_NZ:-25}"
  INDOOR_PDE_ITERS="${TEST_INDOOR_PDE_ITERS:-500}"
fi

export SHARDS_ROOT GLOBAL_SEED RUNS_PER_SHARD INDOOR_NX INDOOR_NZ INDOOR_PDE_ITERS

mkdir -p "$REPO_DIR/slurm/logs" "$SHARDS_ROOT"

ARRAY_MAX=$((NUM_SHARDS - 1))
TOTAL_RUNS=$((NUM_SHARDS * RUNS_PER_SHARD))

echo "shards=$NUM_SHARDS runs_per_shard=$RUNS_PER_SHARD total_runs=$TOTAL_RUNS"
echo "grid=${INDOOR_NX}x${INDOOR_NZ} iters=$INDOOR_PDE_ITERS"
echo "shards_root=$SHARDS_ROOT"

EXPORT="ALL,REPO_DIR=$REPO_DIR,SHARDS_ROOT=$SHARDS_ROOT"
EXPORT="${EXPORT},NUM_SHARDS=$NUM_SHARDS,TOTAL_RUNS=$TOTAL_RUNS,RUNS_PER_SHARD=$RUNS_PER_SHARD"
EXPORT="${EXPORT},GLOBAL_SEED=$GLOBAL_SEED"
EXPORT="${EXPORT},INDOOR_NX=$INDOOR_NX,INDOOR_NZ=$INDOOR_NZ,INDOOR_PDE_ITERS=$INDOOR_PDE_ITERS"
[[ -n "${CORPUS_PYTHON:-}" ]] && EXPORT="${EXPORT},CORPUS_PYTHON=${CORPUS_PYTHON}"

EXTRA=()
if (( TEST_MODE )); then
  EXTRA+=(--time="${SLURM_TIME_SHARD_TEST:-01:00:00}")
elif [[ -n "${SLURM_TIME_SHARD:-}" ]]; then
  EXTRA+=(--time="$SLURM_TIME_SHARD")
fi

SHARD_CMD=(
  sbatch --parsable
  --chdir "$REPO_DIR"
  --array="0-${ARRAY_MAX}"
  --export="$EXPORT"
  "${EXTRA[@]}"
  "$REPO_DIR/slurm/run_indoor_shard.sbatch"
)

if (( DRY_RUN )); then
  echo "dry-run: ${SHARD_CMD[*]}"
  exit 0
fi

JOB="$("${SHARD_CMD[@]}")"
echo "submitted indoor shard array: $JOB"
echo "shards_root=$SHARDS_ROOT"
