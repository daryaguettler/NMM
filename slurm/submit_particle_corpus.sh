#!/bin/bash
#===============================================================
# submit_particle_corpus.sh — Engaging/Slurm: shard array + merge
#
# usage:
#   export SLURM_ACCOUNT=... SLURM_PARTITION=...   # your ORCD values
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
#===============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR

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

if [[ -z "${SLURM_ACCOUNT:-}" || -z "${SLURM_PARTITION:-}" ]]; then
  echo "set SLURM_ACCOUNT and SLURM_PARTITION (ORCD) before submitting" >&2
  exit 1
fi

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

EXPORT="ALL,REPO_DIR=$REPO_DIR,SHARDS_ROOT=$SHARDS_ROOT,CORPUS_OUT=$CORPUS_OUT,RUNS_PER_SHARD=$RUNS_PER_SHARD,GLOBAL_SEED=$GLOBAL_SEED,PARTICLES=$PARTICLES"
[[ -n "${DURATION_HOURS:-}" ]] && EXPORT="${EXPORT},DURATION_HOURS=$DURATION_HOURS"

SHARD_SBATCH=(
  sbatch --parsable
  -A "$SLURM_ACCOUNT"
  -p "$SLURM_PARTITION"
  --chdir "$REPO_DIR"
  --array="0-${ARRAY_MAX}"
  --export="$EXPORT"
  "$REPO_DIR/slurm/run_corpus_shard.sbatch"
)

if (( DRY_RUN )); then
  echo "dry-run: ${SHARD_SBATCH[*]}"
  echo "then: sbatch ... --dependency=afterok:<shard_jobid> slurm/run_corpus_merge.sbatch"
  exit 0
fi

SHARD_JOB="$("${SHARD_SBATCH[@]}")"
echo "submitted shard array: $SHARD_JOB  (tasks 0-${ARRAY_MAX})"

MERGE_EXPORT="ALL,REPO_DIR=$REPO_DIR,SHARDS_ROOT=$SHARDS_ROOT,CORPUS_OUT=$CORPUS_OUT"
MERGE_SBATCH=(
  sbatch --parsable
  -A "$SLURM_ACCOUNT"
  -p "$SLURM_PARTITION"
  --chdir "$REPO_DIR"
  --dependency="afterok:${SHARD_JOB}"
  --export="$MERGE_EXPORT"
  "$REPO_DIR/slurm/run_corpus_merge.sbatch"
)

MERGE_JOB="$("${MERGE_SBATCH[@]}")"
echo "submitted merge (after array): $MERGE_JOB"
echo "merged corpus → $CORPUS_OUT"
echo "logs: $REPO_DIR/slurm/logs/corpus_shard_${SHARD_JOB}_*.out"
