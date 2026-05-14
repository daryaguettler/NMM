#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:30:00
#SBATCH -J nmm-micro-merge
#SBATCH -o slurm/logs/microclimate_merge_%j.out
#SBATCH -e slurm/logs/microclimate_merge_%j.err
#
# required env: REPO_DIR, SHARDS_ROOT, CORPUS_OUT

set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR not set}"
cd "$REPO_DIR"
# shellcheck source=/dev/null
source "${REPO_DIR}/slurm/inc_env.sh"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

SHARDS_ROOT="${SHARDS_ROOT:?SHARDS_ROOT not set}"
CORPUS_OUT="${CORPUS_OUT:?CORPUS_OUT not set}"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "stage=microclimate_merge job=${SLURM_JOB_ID:-} host=$(hostname) utc=${TS}"

corpus_run_py -m microclimate.cluster.merge_shards \
  --shards-root "$SHARDS_ROOT" \
  --out "$CORPUS_OUT"

echo "stage=microclimate_merge finished utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
