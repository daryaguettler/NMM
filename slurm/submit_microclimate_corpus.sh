#!/bin/bash
#===============================================================
# submit_microclimate_corpus.sh — Engaging: shard array + merge (pde corpus)
#
# usage:
#   bash slurm/submit_microclimate_corpus.sh
#   bash slurm/submit_microclimate_corpus.sh --num-shards 3 --runs-per-shard 9
#   bash slurm/submit_microclimate_corpus.sh --test
#   bash slurm/submit_microclimate_corpus.sh --dry-run
#
# env (optional):
#   REPO_DIR, SHARDS_ROOT, CORPUS_OUT
#   GLOBAL_SEED, MICRO_NX, MICRO_NZ, MICRO_PDE_ITERS
#   CORPUS_PYTHON, SLURM_TIME_SHARD, SLURM_TIME_MERGE (+ _TEST variants)
#===============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR

for _need in \
  "$REPO_DIR/slurm/run_microclimate_shard.sbatch" \
  "$REPO_DIR/slurm/run_microclimate_merge.sbatch" \
  "$REPO_DIR/slurm/inc_env.sh" \
  "$REPO_DIR/src/microclimate"
do
  [[ -e "$_need" ]] || { echo "missing required path: $_need" >&2; exit 1; }
done

SHARDS_ROOT="${SHARDS_ROOT:-$HOME/orcd/scratch/nmm_microclimate_shards}"
CORPUS_OUT="${CORPUS_OUT:-$HOME/orcd/scratch/nmm_corpus/v0_microclimate}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
MICRO_NX="${MICRO_NX:-200}"
MICRO_NZ="${MICRO_NZ:-100}"
MICRO_PDE_ITERS="${MICRO_PDE_ITERS:-4000}"

NUM_SHARDS=3
RUNS_PER_SHARD=9
TEST_MODE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-shards)     NUM_SHARDS="$2"; shift ;;
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
  MICRO_NX="${TEST_MICRO_NX:-48}"
  MICRO_NZ="${TEST_MICRO_NZ:-32}"
  MICRO_PDE_ITERS="${TEST_MICRO_PDE_ITERS:-400}"
fi

export SHARDS_ROOT CORPUS_OUT GLOBAL_SEED RUNS_PER_SHARD MICRO_NX MICRO_NZ MICRO_PDE_ITERS

mkdir -p "$REPO_DIR/slurm/logs" "$SHARDS_ROOT" "$CORPUS_OUT"

ARRAY_MAX=$((NUM_SHARDS - 1))
TOTAL_RUNS=$((NUM_SHARDS * RUNS_PER_SHARD))

echo "shards=$NUM_SHARDS runs_per_shard=$RUNS_PER_SHARD total_runs=$TOTAL_RUNS"
echo "grid=${MICRO_NX}x${MICRO_NZ} iters=$MICRO_PDE_ITERS"
echo "shards_root=$SHARDS_ROOT corpus_out=$CORPUS_OUT"

EXPORT="ALL,REPO_DIR=$REPO_DIR,SHARDS_ROOT=$SHARDS_ROOT,CORPUS_OUT=$CORPUS_OUT"
EXPORT="${EXPORT},NUM_SHARDS=$NUM_SHARDS,TOTAL_RUNS=$TOTAL_RUNS,RUNS_PER_SHARD=$RUNS_PER_SHARD"
EXPORT="${EXPORT},GLOBAL_SEED=$GLOBAL_SEED,MICRO_NX=$MICRO_NX,MICRO_NZ=$MICRO_NZ,MICRO_PDE_ITERS=$MICRO_PDE_ITERS"
[[ -n "${CORPUS_PYTHON:-}" ]] && EXPORT="${EXPORT},CORPUS_PYTHON=${CORPUS_PYTHON}"

SHARD_EXTRA=()
if (( TEST_MODE )); then
  SHARD_EXTRA+=(--time="${SLURM_TIME_SHARD_TEST:-02:00:00}")
elif [[ -n "${SLURM_TIME_SHARD:-}" ]]; then
  SHARD_EXTRA+=(--time="$SLURM_TIME_SHARD")
fi

MERGE_EXTRA=()
if (( TEST_MODE )); then
  MERGE_EXTRA+=(--time="${SLURM_TIME_MERGE_TEST:-00:15:00}")
elif [[ -n "${SLURM_TIME_MERGE:-}" ]]; then
  MERGE_EXTRA+=(--time="$SLURM_TIME_MERGE")
fi

MERGE_EXPORT="ALL,REPO_DIR=$REPO_DIR,SHARDS_ROOT=$SHARDS_ROOT,CORPUS_OUT=$CORPUS_OUT"
MERGE_EXPORT="${MERGE_EXPORT},NUM_SHARDS=$NUM_SHARDS,TOTAL_RUNS=$TOTAL_RUNS"
[[ -n "${CORPUS_PYTHON:-}" ]] && MERGE_EXPORT="${MERGE_EXPORT},CORPUS_PYTHON=${CORPUS_PYTHON}"

SHARD_SBATCH=(
  sbatch --parsable
  --chdir "$REPO_DIR"
  --array="0-${ARRAY_MAX}"
  --export="$EXPORT"
  "${SHARD_EXTRA[@]}"
  "$REPO_DIR/slurm/run_microclimate_shard.sbatch"
)

if (( DRY_RUN )); then
  echo "dry-run shard: ${SHARD_SBATCH[*]}"
  exit 0
fi

SHARD_JOB="$("${SHARD_SBATCH[@]}")"
echo "submitted shard array: $SHARD_JOB"

MERGE_SBATCH=(
  sbatch --parsable
  --chdir "$REPO_DIR"
  --dependency="afterok:${SHARD_JOB}"
  --export="$MERGE_EXPORT"
  "${MERGE_EXTRA[@]}"
  "$REPO_DIR/slurm/run_microclimate_merge.sbatch"
)

MERGE_JOB="$("${MERGE_SBATCH[@]}")"
echo "submitted merge: $MERGE_JOB -> $CORPUS_OUT"
