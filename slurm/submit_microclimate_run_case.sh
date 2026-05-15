#!/bin/bash
#===============================================================
# submit_microclimate_run_case.sh — four-method run_case on compute node
#
# usage:
#   bash slurm/submit_microclimate_run_case.sh
#
# env (optional):
#   REPO_DIR
#   RUN_CASE_OUT          output directory (under repo or absolute)
#   RUN_CASE_NX, RUN_CASE_NZ, RUN_CASE_PDE_ITERS   defaults match corpus shards (200,100,4000)
#   RUN_CASE_PARTICLES    0 or 1 (default 1)
#   RUN_CASE_N_PARTICLES, RUN_CASE_PARTICLE_SPINUP, RUN_CASE_PARTICLE_AVG
#   SURROGATE_ARTIFACT    path to train_surrogate --out, or empty to omit surrogate
#   RUN_CASE_FREEZE_WIND  1 to pass --freeze-wind
#   CORPUS_PYTHON, SLURM_TIME_RUN_CASE
#===============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR

for _need in \
  "$REPO_DIR/slurm/run_microclimate_run_case.sbatch" \
  "$REPO_DIR/slurm/inc_env.sh" \
  "$REPO_DIR/src/microclimate/apps/run_case.py"
do
  [[ -e "$_need" ]] || { echo "missing required path: $_need" >&2; exit 1; }
done

RUN_CASE_OUT="${RUN_CASE_OUT:-$REPO_DIR/outputs/microclimate/run_case_slurm}"
RUN_CASE_NX="${RUN_CASE_NX:-200}"
RUN_CASE_NZ="${RUN_CASE_NZ:-100}"
RUN_CASE_PDE_ITERS="${RUN_CASE_PDE_ITERS:-4000}"
RUN_CASE_PARTICLES="${RUN_CASE_PARTICLES:-1}"
RUN_CASE_N_PARTICLES="${RUN_CASE_N_PARTICLES:-20000}"
RUN_CASE_PARTICLE_SPINUP="${RUN_CASE_PARTICLE_SPINUP:-4000}"
RUN_CASE_PARTICLE_AVG="${RUN_CASE_PARTICLE_AVG:-2000}"
RUN_CASE_FREEZE_WIND="${RUN_CASE_FREEZE_WIND:-0}"
# use ${var-default} so SURROGATE_ARTIFACT=  (empty) means omit --surrogate-artifact
_DEFAULT_SURR="$HOME/orcd/scratch/nmm_surrogate/micro_v0_sbatch"
SURROGATE_ARTIFACT="${SURROGATE_ARTIFACT-$_DEFAULT_SURR}"

mkdir -p "$REPO_DIR/slurm/logs"

echo "run_case_out=$RUN_CASE_OUT"
echo "grid=${RUN_CASE_NX}x${RUN_CASE_NZ} pde_iters=$RUN_CASE_PDE_ITERS particles=$RUN_CASE_PARTICLES"
echo "surrogate_artifact=${SURROGATE_ARTIFACT:-<none>}"

EXPORT="ALL,REPO_DIR=$REPO_DIR,RUN_CASE_OUT=$RUN_CASE_OUT"
EXPORT="${EXPORT},RUN_CASE_NX=$RUN_CASE_NX,RUN_CASE_NZ=$RUN_CASE_NZ"
EXPORT="${EXPORT},RUN_CASE_PDE_ITERS=$RUN_CASE_PDE_ITERS"
EXPORT="${EXPORT},RUN_CASE_PARTICLES=$RUN_CASE_PARTICLES"
EXPORT="${EXPORT},RUN_CASE_N_PARTICLES=$RUN_CASE_N_PARTICLES"
EXPORT="${EXPORT},RUN_CASE_PARTICLE_SPINUP=$RUN_CASE_PARTICLE_SPINUP"
EXPORT="${EXPORT},RUN_CASE_PARTICLE_AVG=$RUN_CASE_PARTICLE_AVG"
EXPORT="${EXPORT},RUN_CASE_FREEZE_WIND=$RUN_CASE_FREEZE_WIND"
[[ -n "${SURROGATE_ARTIFACT:-}" ]] && EXPORT="${EXPORT},SURROGATE_ARTIFACT=$SURROGATE_ARTIFACT"
[[ -n "${CORPUS_PYTHON:-}" ]] && EXPORT="${EXPORT},CORPUS_PYTHON=${CORPUS_PYTHON}"

SBATCH_EXTRA=()
if [[ -n "${SLURM_TIME_RUN_CASE:-}" ]]; then
  SBATCH_EXTRA+=(--time="$SLURM_TIME_RUN_CASE")
fi

JOB="$(
  sbatch --parsable --chdir "$REPO_DIR" \
    --export="$EXPORT" \
    "${SBATCH_EXTRA[@]}" \
    "$REPO_DIR/slurm/run_microclimate_run_case.sbatch"
)"
echo "submitted run_case: $JOB -> $RUN_CASE_OUT"
echo "log: $REPO_DIR/slurm/logs/microclimate_run_case_${JOB}.out"
