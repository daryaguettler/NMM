#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 04:00:00
#SBATCH -J nmm-indoor-run
#SBATCH -o slurm/logs/indoor_run_%j.out
#SBATCH -e slurm/logs/indoor_run_%j.err
#
# required env: REPO_DIR, INDOOR_OUT
# optional: INDOOR_NX, INDOOR_NZ, INDOOR_PDE_ITERS, INDOOR_T_HOT,
#           INDOOR_T_OUTDOOR, INDOOR_WIN_OPEN, INDOOR_PARTICLES,
#           INDOOR_N_PARTICLES, INDOOR_SPINUP, INDOOR_AVG

set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR not set}"
cd "$REPO_DIR"
# shellcheck source=/dev/null
source "${REPO_DIR}/slurm/inc_env.sh"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

INDOOR_OUT="${INDOOR_OUT:?INDOOR_OUT not set}"
INDOOR_NX="${INDOOR_NX:-240}"
INDOOR_NZ="${INDOOR_NZ:-100}"
INDOOR_PDE_ITERS="${INDOOR_PDE_ITERS:-15000}"
INDOOR_T_HOT="${INDOOR_T_HOT:-45.0}"
INDOOR_T_OUTDOOR="${INDOOR_T_OUTDOOR:-30.0}"
INDOOR_WIN_OPEN="${INDOOR_WIN_OPEN:-1.0}"
INDOOR_N_PARTICLES="${INDOOR_N_PARTICLES:-20000}"
INDOOR_SPINUP="${INDOOR_SPINUP:-5000}"
INDOOR_AVG="${INDOOR_AVG:-4000}"

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "stage=indoor_run_case job=${SLURM_JOB_ID:-} host=$(hostname) utc=${TS}"

EXTRA_FLAGS=()
if [[ "${INDOOR_PARTICLES:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--particles --n-particles "$INDOOR_N_PARTICLES"
                --particle-spinup "$INDOOR_SPINUP" --particle-avg "$INDOOR_AVG")
fi

corpus_run_py -m indoor.apps.run_case \
  --out "$INDOOR_OUT" \
  --nx "$INDOOR_NX" \
  --nz "$INDOOR_NZ" \
  --pde-iters "$INDOOR_PDE_ITERS" \
  --T-hot "$INDOOR_T_HOT" \
  --T-outdoor "$INDOOR_T_OUTDOOR" \
  --window-open "$INDOOR_WIN_OPEN" \
  "${EXTRA_FLAGS[@]}"

echo "stage=indoor_run_case finished utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
