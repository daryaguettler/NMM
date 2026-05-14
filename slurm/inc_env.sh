# sourced by run_corpus_*.sbatch after cd to repo root
# shellcheck shell=bash

_inc_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_inc_dir}/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_inc_dir}/env.sh"
fi

# uv installer defaults (compute nodes often lack login-node PATH)
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

corpus_run_py() {
  if [[ -n "${CORPUS_PYTHON:-}" ]]; then
    "${CORPUS_PYTHON}" "$@"
    return $?
  fi
  if command -v uv >/dev/null 2>&1; then
    uv run python "$@"
    return $?
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 "$@"
    return $?
  fi
  echo "corpus_run_py: no uv or python3 on PATH after slurm/inc_env.sh" >&2
  echo "  fix: copy slurm/env.sh.example -> slurm/env.sh and add module/conda," >&2
  echo "  or: export CORPUS_PYTHON=/path/to/python before sbatch (absolute path)." >&2
  return 127
}
