#!/bin/bash
#===============================================================
# submit_indoor_run_case.sh — four-method indoor run on a compute node
#
# usage:
#   bash slurm/submit_indoor_run_case.sh
#   INDOOR_T_HOT=55 INDOOR_WIN_OPEN=0 bash slurm/submit_indoor_run_case.sh
#   bash slurm/submit_indoor_run_case.sh --dry-run
#
# env (optional):
#   REPO_DIR
#   INDOOR_OUT           output directory for npz + metrics.json
#   INDOOR_NX, INDOOR_NZ, INDOOR_PDE_ITERS
#   INDOOR_T_HOT, INDOOR_T_OUTDOOR, INDOOR_WIN_OPEN
#   INDOOR_PARTICLES     0 or 1  (default 1: run particle method too)
#   INDOOR_N_PARTICLES, INDOOR_SPINUP, INDOOR_AVG
#   CORPUS_PYTHON, SLURM_TIME_RUN_CASE
#===============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR

for _need in \
  "$REPO_DIR/slurm/run_indoor_run_case.sbatch" \
  "$REPO_DIR/slurm/inc_env.sh" \
  "$REPO_DIR/src/indoor/apps/run_case.py"
do
  [[ -e "$_need" ]] || { echo "missing required path: $_need" >&2; exit 1; }
done

INDOOR_OUT="${INDOOR_OUT:-$HOME/orcd/scratch/nmm_indoor/default_run}"
INDOOR_NX="${INDOOR_NX:-240}"
INDOOR_NZ="${INDOOR_NZ:-100}"
INDOOR_PDE_ITERS="${INDOOR_PDE_ITERS:-15000}"
INDOOR_T_HOT="${INDOOR_T_HOT:-45.0}"
INDOOR_T_OUTDOOR="${INDOOR_T_OUTDOOR:-30.0}"
INDOOR_WIN_OPEN="${INDOOR_WIN_OPEN:-1.0}"
INDOOR_PARTICLES="${INDOOR_PARTICLES:-1}"
INDOOR_N_PARTICLES="${INDOOR_N_PARTICLES:-20000}"
INDOOR_SPINUP="${INDOOR_SPINUP:-5000}"
INDOOR_AVG="${INDOOR_AVG:-4000}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
  shift
done

mkdir -p "$REPO_DIR/slurm/logs"

echo "indoor_out=$INDOOR_OUT"
echo "grid=${INDOOR_NX}x${INDOOR_NZ} pde_iters=$INDOOR_PDE_ITERS"
echo "T_hot=$INDOOR_T_HOT T_outdoor=$INDOOR_T_OUTDOOR window_open=$INDOOR_WIN_OPEN"
echo "particles=$INDOOR_PARTICLES  n=$INDOOR_N_PARTICLES spin=$INDOOR_SPINUP avg=$INDOOR_AVG"

EXPORT="ALL,REPO_DIR=$REPO_DIR,INDOOR_OUT=$INDOOR_OUT"
EXPORT="${EXPORT},INDOOR_NX=$INDOOR_NX,INDOOR_NZ=$INDOOR_NZ,INDOOR_PDE_ITERS=$INDOOR_PDE_ITERS"
EXPORT="${EXPORT},INDOOR_T_HOT=$INDOOR_T_HOT,INDOOR_T_OUTDOOR=$INDOOR_T_OUTDOOR"
EXPORT="${EXPORT},INDOOR_WIN_OPEN=$INDOOR_WIN_OPEN,INDOOR_PARTICLES=$INDOOR_PARTICLES"
EXPORT="${EXPORT},INDOOR_N_PARTICLES=$INDOOR_N_PARTICLES"
EXPORT="${EXPORT},INDOOR_SPINUP=$INDOOR_SPINUP,INDOOR_AVG=$INDOOR_AVG"
[[ -n "${CORPUS_PYTHON:-}" ]] && EXPORT="${EXPORT},CORPUS_PYTHON=${CORPUS_PYTHON}"

SBATCH_CMD=(
  sbatch --parsable --chdir "$REPO_DIR"
  --export="$EXPORT"
)
[[ -n "${SLURM_TIME_RUN_CASE:-}" ]] && SBATCH_CMD+=(--time="$SLURM_TIME_RUN_CASE")
SBATCH_CMD+=("$REPO_DIR/slurm/run_indoor_run_case.sbatch")

if (( DRY_RUN )); then
  echo "dry-run: ${SBATCH_CMD[*]}"
  exit 0
fi

JOB="$("${SBATCH_CMD[@]}")"
echo "submitted indoor run_case: $JOB -> $INDOOR_OUT"
echo "log: $REPO_DIR/slurm/logs/indoor_run_${JOB}.out"
