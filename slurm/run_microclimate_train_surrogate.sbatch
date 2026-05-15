#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -J nmm-micro-train
#SBATCH -o slurm/logs/microclimate_train_%j.out
#SBATCH -e slurm/logs/microclimate_train_%j.err
#
# required env: REPO_DIR, CORPUS, SURROGATE_OUT
# optional env: TRAIN_ITERATIONS, TRAIN_LR, TRAIN_PHYSICS_WEIGHT, TRAIN_VAL_FRACTION,
#               TRAIN_MAX_POINTS (0 = use all points per run), TRAIN_SEED,
#               SLURM_CPUS handled by slurm; tune threads for jax/xla on cpu:
#               OMP_NUM_THREADS, XLA_FLAGS (export before sbatch or in slurm/env.sh)

set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR not set}"
cd "$REPO_DIR"
# shellcheck source=/dev/null
source "${REPO_DIR}/slurm/inc_env.sh"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

_nt="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$_nt}"

CORPUS="${CORPUS:?CORPUS not set (merged corpus dir with runs/)}"
SURROGATE_OUT="${SURROGATE_OUT:?SURROGATE_OUT not set}"
TRAIN_ITERATIONS="${TRAIN_ITERATIONS:-5000}"
TRAIN_LR="${TRAIN_LR:-1e-3}"
TRAIN_PHYSICS_WEIGHT="${TRAIN_PHYSICS_WEIGHT:-0}"
TRAIN_VAL_FRACTION="${TRAIN_VAL_FRACTION:-0.2}"
TRAIN_MAX_POINTS="${TRAIN_MAX_POINTS:-8000}"
TRAIN_SEED="${TRAIN_SEED:-0}"

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "stage=microclimate_train job=${SLURM_JOB_ID:-} host=$(hostname) cpus=${_nt} utc=${TS}"
echo "corpus=${CORPUS} out=${SURROGATE_OUT} iterations=${TRAIN_ITERATIONS}"

corpus_run_py -m microclimate.apps.train_surrogate \
  --corpus "$CORPUS" \
  --out "$SURROGATE_OUT" \
  --iterations "$TRAIN_ITERATIONS" \
  --lr "$TRAIN_LR" \
  --physics-weight "$TRAIN_PHYSICS_WEIGHT" \
  --val-fraction "$TRAIN_VAL_FRACTION" \
  --seed "$TRAIN_SEED" \
  --max-points "$TRAIN_MAX_POINTS"

echo "stage=microclimate_train finished utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
