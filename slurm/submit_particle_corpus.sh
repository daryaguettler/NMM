#!/bin/bash
#===============================================================
# submit_particle_corpus.sh — Engaging/Slurm: shard array + merge
#
# usage (partition/resources are in the .sbatch files, default mit_normal):
#   bash slurm/submit_particle_corpus.sh
#
#   bash slurm/submit_particle_corpus.sh --num-shards 10 --runs-per-shard 50
#   bash slurm/submit_particle_corpus.sh --test
#   bash slurm/submit_particle_corpus.sh --dry-run
#
# env (optional):
#   REPO_DIR          default: repo root (parent of slurm/)
#   SHARDS_ROOT       default: ~/orcd/scratch/nmm_corpus_shards
#   CORPUS_OUT        default: ~/orcd/scratch/nmm_corpus/v0_particle
#   GLOBAL_SEED, PARTICLES, DURATION_HOURS  passed to build_corpus_cluster
#   CORPUS_PYTHON     optional absolute path to python (uv not on compute PATH)
#
#   SLURM_TIME_SHARD, SLURM_TIME_MERGE  optional sbatch --time override (production)
#   (test mode always passes short --time unless SLURM_TIME_SHARD_TEST / SLURM_TIME_MERGE_TEST set)
#
# sbatch time in slurm/run_corpus_shard.sbatch is 12h (mit_normal rejects many >24h requests).
# increase via env or edit .sbatch if your partition allows longer jobs.
#===============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR

for _need in \
  "$REPO_DIR/slurm/run_corpus_shard.sbatch" \
  "$REPO_DIR/slurm/run_corpus_merge.sbatch" \
  "$REPO_DIR/slurm/inc_env.sh" \
  "$REPO_DIR/src/particle_sim"
do
  [[ -e "$_need" ]] || { echo "missing required path: $_need" >&2; exit 1; }
done

SHARDS_ROOT="${SHARDS_ROOT:-$HOME/orcd/scratch/nmm_corpus_shards}"
CORPUS_OUT="${CORPUS_OUT:-$HOME/orcd/scratch/nmm_corpus/v0_particle}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
PARTICLES="${PARTICLES:-300}"
DURATION_HOURS="${DURATION_HOURS:-}"

NUM_SHARDS=10
RUNS_PER_SHARD=50
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
  PARTICLES="${TEST_PARTICLES:-80}"
  DURATION_HOURS="${TEST_DURATION_HOURS:-1.0}"
fi

export SHARDS_ROOT CORPUS_OUT GLOBAL_SEED PARTICLES RUNS_PER_SHARD
[[ -n "${DURATION_HOURS:-}" ]] && export DURATION_HOURS

mkdir -p "$REPO_DIR/slurm/logs" "$SHARDS_ROOT" "$CORPUS_OUT"

ARRAY_MAX=$((NUM_SHARDS - 1))
TOTAL_RUNS=$((NUM_SHARDS * RUNS_PER_SHARD))

echo "shards=$NUM_SHARDS runs_per_shard=$RUNS_PER_SHARD total_runs=$TOTAL_RUNS"
echo "shards_root=$SHARDS_ROOT  corpus_out=$CORPUS_OUT"

EXPORT="ALL,REPO_DIR=$REPO_DIR,SHARDS_ROOT=$SHARDS_ROOT,CORPUS_OUT=$CORPUS_OUT"
EXPORT="${EXPORT},NUM_SHARDS=$NUM_SHARDS,TOTAL_RUNS=$TOTAL_RUNS,RUNS_PER_SHARD=$RUNS_PER_SHARD"
EXPORT="${EXPORT},GLOBAL_SEED=$GLOBAL_SEED,PARTICLES=$PARTICLES"
[[ -n "${DURATION_HOURS:-}" ]] && EXPORT="${EXPORT},DURATION_HOURS=$DURATION_HOURS"
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
  "$REPO_DIR/slurm/run_corpus_shard.sbatch"
)

if (( DRY_RUN )); then
  echo "dry-run shard: ${SHARD_SBATCH[*]}"
  echo "dry-run merge (substitute <shard_jobid>):"
  echo "  sbatch --parsable --chdir \"$REPO_DIR\" \\"
  echo "    --dependency=afterok:<shard_jobid> --export \"$MERGE_EXPORT\" \\"
  if ((${#MERGE_EXTRA[@]} > 0)); then
    _mextra=""
    for _x in "${MERGE_EXTRA[@]}"; do _mextra+=" ${_x}"; done
    echo "    ${_mextra# } \\"
  fi
  echo "    \"$REPO_DIR/slurm/run_corpus_merge.sbatch\""
  exit 0
fi

SHARD_JOB="$("${SHARD_SBATCH[@]}")"
echo "submitted shard array: $SHARD_JOB  (tasks 0-${ARRAY_MAX})"

MERGE_SBATCH=(
  sbatch --parsable
  --chdir "$REPO_DIR"
  --dependency="afterok:${SHARD_JOB}"
  --export="$MERGE_EXPORT"
  "${MERGE_EXTRA[@]}"
  "$REPO_DIR/slurm/run_corpus_merge.sbatch"
)

MERGE_JOB="$("${MERGE_SBATCH[@]}")"
echo "submitted merge (after array): $MERGE_JOB"
echo "merged corpus → $CORPUS_OUT"
echo "shard array job_id=${SHARD_JOB} (not the merge id). per-task logs:"
echo "  $REPO_DIR/slurm/logs/corpus_shard_${SHARD_JOB}_<task>.out"
echo "  $REPO_DIR/slurm/logs/corpus_shard_${SHARD_JOB}_<task>.err"
echo "merge job_id=${MERGE_JOB} logs:"
echo "  $REPO_DIR/slurm/logs/corpus_merge_${MERGE_JOB}.out"
echo "  $REPO_DIR/slurm/logs/corpus_merge_${MERGE_JOB}.err"
