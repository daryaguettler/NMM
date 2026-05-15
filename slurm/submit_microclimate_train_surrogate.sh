#!/bin/bash
#===============================================================
# submit_microclimate_train_surrogate.sh — train surrogate on compute node
#
# usage:
#   bash slurm/submit_microclimate_train_surrogate.sh
#   CORPUS=... SURROGATE_OUT=... TRAIN_ITERATIONS=8000 bash slurm/submit_microclimate_train_surrogate.sh
#
# env (optional):
#   REPO_DIR, CORPUS, SURROGATE_OUT
#   TRAIN_ITERATIONS, TRAIN_LR, TRAIN_PHYSICS_WEIGHT, TRAIN_VAL_FRACTION,
#   TRAIN_MAX_POINTS (0 = all points per run), TRAIN_SEED
#   CORPUS_PYTHON (see slurm/inc_env.sh)
#   SLURM_TIME_TRAIN (override batch time limit, e.g. 08:00:00)
#===============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR

for _need in \
  "$REPO_DIR/slurm/run_microclimate_train_surrogate.sbatch" \
  "$REPO_DIR/slurm/inc_env.sh" \
  "$REPO_DIR/src/microclimate/apps/train_surrogate.py"
do
  [[ -e "$_need" ]] || { echo "missing required path: $_need" >&2; exit 1; }
done

CORPUS="${CORPUS:-$HOME/orcd/scratch/nmm_corpus/v0_microclimate}"
SURROGATE_OUT="${SURROGATE_OUT:-$HOME/orcd/scratch/nmm_surrogate/micro_v0}"
TRAIN_ITERATIONS="${TRAIN_ITERATIONS:-5000}"
TRAIN_LR="${TRAIN_LR:-1e-3}"
TRAIN_PHYSICS_WEIGHT="${TRAIN_PHYSICS_WEIGHT:-0}"
TRAIN_VAL_FRACTION="${TRAIN_VAL_FRACTION:-0.2}"
TRAIN_MAX_POINTS="${TRAIN_MAX_POINTS:-8000}"
TRAIN_SEED="${TRAIN_SEED:-0}"

mkdir -p "$REPO_DIR/slurm/logs"

echo "corpus=$CORPUS"
echo "surrogate_out=$SURROGATE_OUT"
echo "iterations=$TRAIN_ITERATIONS lr=$TRAIN_LR physics_weight=$TRAIN_PHYSICS_WEIGHT max_points=$TRAIN_MAX_POINTS"

EXPORT="ALL,REPO_DIR=$REPO_DIR"
EXPORT="${EXPORT},CORPUS=$CORPUS,SURROGATE_OUT=$SURROGATE_OUT"
EXPORT="${EXPORT},TRAIN_ITERATIONS=$TRAIN_ITERATIONS,TRAIN_LR=$TRAIN_LR"
EXPORT="${EXPORT},TRAIN_PHYSICS_WEIGHT=$TRAIN_PHYSICS_WEIGHT"
EXPORT="${EXPORT},TRAIN_VAL_FRACTION=$TRAIN_VAL_FRACTION"
EXPORT="${EXPORT},TRAIN_MAX_POINTS=$TRAIN_MAX_POINTS,TRAIN_SEED=$TRAIN_SEED"
[[ -n "${CORPUS_PYTHON:-}" ]] && EXPORT="${EXPORT},CORPUS_PYTHON=${CORPUS_PYTHON}"

SBATCH_EXTRA=()
if [[ -n "${SLURM_TIME_TRAIN:-}" ]]; then
  SBATCH_EXTRA+=(--time="$SLURM_TIME_TRAIN")
fi

JOB="$(
  sbatch --parsable --chdir "$REPO_DIR" \
    --export="$EXPORT" \
    "${SBATCH_EXTRA[@]}" \
    "$REPO_DIR/slurm/run_microclimate_train_surrogate.sbatch"
)"
echo "submitted train_surrogate: $JOB -> $SURROGATE_OUT"
echo "log: $REPO_DIR/slurm/logs/microclimate_train_${JOB}.out"
