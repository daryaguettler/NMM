# engaging / slurm corpus shards

## repo slurm driver (recommended on Engaging)

from repo root, after `module load` / `uv sync` as needed:

```bash
export SLURM_ACCOUNT=... SLURM_PARTITION=...
bash slurm/submit_particle_corpus.sh --num-shards 10 --runs-per-shard 50
# test: bash slurm/submit_particle_corpus.sh --test
# dry-run: bash slurm/submit_particle_corpus.sh --dry-run
```

see `slurm/run_corpus_shard.sbatch` (array tasks) and `slurm/run_corpus_merge.sbatch` (runs after `afterok`).

## quick start (manual local shard)

from repo root (with `uv` env), generate one shard locally:

```bash
PYTHONPATH=src uv run python -m particle_sim.apps.build_corpus_cluster \
  --out corpus/v0_particle_shards --run-start 0 --run-count 10 --shard-id 0
```

merge shards after jobs finish:

```bash
PYTHONPATH=src uv run python -m particle_sim.cluster.merge_shards \
  --shards-root corpus/v0_particle_shards --out corpus/v0_particle
```

## template

see `slurm_templates/corpus_array.slurm` for an array job sketch; adjust partition, account, and conda/uv module loads per ORCD docs.

## parallelism

- each array task is independent (different `shard_id` / `run-start`).
- within a node you can run multiple processes with disjoint `--run-start` ranges instead of relying on mpi.
