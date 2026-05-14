#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:30:00
#SBATCH -J nmm-corpus-merge
#SBATCH -o slurm/logs/corpus_merge_%j.out
#SBATCH -e slurm/logs/corpus_merge_%j.err
#
# merge_shards: copies npz + merges manifest (I/O bound, low cpu/mem).
# 30m default; raise for huge corpora or slow FS.
# required env: REPO_DIR, SHARDS_ROOT, CORPUS_OUT
# optional: NUM_SHARDS, TOTAL_RUNS (logging; set by submit_particle_corpus.sh)
# submit after shard array: --dependency=afterok:<shard_jobid>

set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR not set}"
cd "$REPO_DIR"
# shellcheck source=/dev/null
source "${REPO_DIR}/slurm/inc_env.sh"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

SHARDS_ROOT="${SHARDS_ROOT:?SHARDS_ROOT not set}"
CORPUS_OUT="${CORPUS_OUT:?CORPUS_OUT not set}"
_read_num_shards="${NUM_SHARDS:-}"
_read_total_runs="${TOTAL_RUNS:-}"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "stage=corpus_merge job=${SLURM_JOB_ID:-} host=$(hostname) utc=${TS}"
echo "repo=${REPO_DIR} shards_root=${SHARDS_ROOT} out=${CORPUS_OUT} num_shards=${_read_num_shards:-?} total_runs=${_read_total_runs:-?}"

corpus_run_py -m particle_sim.cluster.merge_shards \
  --shards-root "$SHARDS_ROOT" \
  --out "$CORPUS_OUT"
echo "stage=corpus_merge finished utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
