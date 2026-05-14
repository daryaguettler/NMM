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
# submit after shard array: --dependency=afterok:<shard_jobid>

set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR not set}"
cd "$REPO_DIR"
export PYTHONPATH="${PYTHONPATH:-src}"

SHARDS_ROOT="${SHARDS_ROOT:?SHARDS_ROOT not set}"
CORPUS_OUT="${CORPUS_OUT:?CORPUS_OUT not set}"

echo "merge shards-root=$SHARDS_ROOT -> out=$CORPUS_OUT"
uv run python -m particle_sim.cluster.merge_shards \
  --shards-root "$SHARDS_ROOT" \
  --out "$CORPUS_OUT"
