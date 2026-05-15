#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH -J nmm-micro-runcase
#SBATCH -o slurm/logs/microclimate_run_case_%j.out
#SBATCH -e slurm/logs/microclimate_run_case_%j.err
#
# required env: REPO_DIR, RUN_CASE_OUT
# optional: RUN_CASE_NX, RUN_CASE_NZ, RUN_CASE_PDE_ITERS
#           RUN_CASE_PARTICLES (0|1), RUN_CASE_N_PARTICLES, RUN_CASE_PARTICLE_SPINUP,
#           RUN_CASE_PARTICLE_AVG, SURROGATE_ARTIFACT (path or empty), RUN_CASE_FREEZE_WIND (0|1)
#           CORPUS_PYTHON (slurm/inc_env.sh)

set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR not set}"
cd "$REPO_DIR"
# shellcheck source=/dev/null
source "${REPO_DIR}/slurm/inc_env.sh"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

_nt="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$_nt}"

RUN_CASE_OUT="${RUN_CASE_OUT:?RUN_CASE_OUT not set}"
RUN_CASE_NX="${RUN_CASE_NX:-200}"
RUN_CASE_NZ="${RUN_CASE_NZ:-100}"
RUN_CASE_PDE_ITERS="${RUN_CASE_PDE_ITERS:-4000}"
RUN_CASE_PARTICLES="${RUN_CASE_PARTICLES:-1}"
RUN_CASE_N_PARTICLES="${RUN_CASE_N_PARTICLES:-20000}"
RUN_CASE_PARTICLE_SPINUP="${RUN_CASE_PARTICLE_SPINUP:-4000}"
RUN_CASE_PARTICLE_AVG="${RUN_CASE_PARTICLE_AVG:-2000}"
RUN_CASE_FREEZE_WIND="${RUN_CASE_FREEZE_WIND:-0}"
SURROGATE_ARTIFACT="${SURROGATE_ARTIFACT:-}"

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "stage=microclimate_run_case job=${SLURM_JOB_ID:-} host=$(hostname) cpus=${_nt} utc=${TS}"
echo "out=${RUN_CASE_OUT} grid=${RUN_CASE_NX}x${RUN_CASE_NZ} pde_iters=${RUN_CASE_PDE_ITERS}"

_extra=()
if [[ "$RUN_CASE_PARTICLES" == "1" ]]; then
  _extra+=(--particles)
fi
if [[ -n "$SURROGATE_ARTIFACT" ]]; then
  _extra+=(--surrogate-artifact "$SURROGATE_ARTIFACT")
fi
if [[ "$RUN_CASE_FREEZE_WIND" == "1" ]]; then
  _extra+=(--freeze-wind)
fi

corpus_run_py -m microclimate.apps.run_case \
  --out "$RUN_CASE_OUT" \
  --nx "$RUN_CASE_NX" \
  --nz "$RUN_CASE_NZ" \
  --pde-iters "$RUN_CASE_PDE_ITERS" \
  --n-particles "$RUN_CASE_N_PARTICLES" \
  --particle-spinup "$RUN_CASE_PARTICLE_SPINUP" \
  --particle-avg "$RUN_CASE_PARTICLE_AVG" \
  "${_extra[@]}"

echo "stage=microclimate_run_case finished utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
