# Shared setup for the Slurm jobs in this folder, and for working on a login node:
#     source slurm/env.sh
# Follows the Stats HPC guide: Python from the /apps modules; the Python environment and the
# (large) run directories on /bitbucket, keeping the 20 GB home quota for code and small files.
# To use other locations, set the variable before sourcing, e.g.  RUNS=/scratch/bulk/$USER/runs

source /etc/profile.d/modules.sh
module purge
module use /apps/modulefiles/Core
module load python/3.12.13

export VENV="${VENV:-/bitbucket/$USER/venvs/score-sde}"            # Python environment (~8 GB)
export RUNS="${RUNS:-/bitbucket/$USER/runs}"                       # ~40 GB per 300k-step run
export UV_CACHE_DIR="${UV_CACHE_DIR:-/bitbucket/$USER/.cache/uv}"   # uv's download cache, out of $HOME
export TFHUB_CACHE_DIR="${TFHUB_CACHE_DIR:-$HOME/.cache/tfhub}"     # Inception network (prefetch.py)
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

mkdir -p "$RUNS"
if [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
else
  echo "slurm/env.sh: no Python environment at $VENV yet" >&2
fi
