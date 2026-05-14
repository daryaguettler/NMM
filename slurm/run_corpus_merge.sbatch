#!/bin/bash
#SBATCH -J nmm-corpus-merge
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH -o slurm/logs/corpus_merge_%j.out
#SBATCH -e slurm/logs/corpus_merge_%j.err
#
# required env: REPO_DIR, SHARDS_ROOT, CORPUS_OUT
# submit after all shard tasks:  --dependency=afterok:<shard_array_jobid>

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
