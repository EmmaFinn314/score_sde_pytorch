#!/bin/bash
# Evaluate every checkpoint of a run with eval_mw.py, then draw the Fig. 1b panels with plot_mw.py.
# First argument: the run name used with slurm/train.sbatch; further arguments go to eval_mw.py.
# The run's config and overrides are read from $RUNS/<name>/train_args.txt. Submit from the
# repository root:
#     sbatch slurm/eval.sbatch mw_n2048
#     sbatch slurm/eval.sbatch mw_n2048 --weights=both
# Checkpoints already in results.csv are skipped, so you can run this while training is still going
# (it evaluates the checkpoints saved so far) and again later, or resubmit after a time-out.
#SBATCH --job-name=sde-eval
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your.name@stats.ox.ac.uk

source slurm/env.sh
set -eo pipefail
RUN="${1:?usage: sbatch slurm/eval.sbatch RUN_NAME [eval_mw.py arguments]}"
shift
WD="$RUNS/$RUN"
if [ ! -f "$WD/train_args.txt" ]; then
  echo "No $WD/train_args.txt: was $RUN trained with slurm/train.sbatch?" >&2
  exit 1
fi
mapfile -t TRAIN_ARGS < "$WD/train_args.txt"

echo "evaluating $WD | node $(hostname) | GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python eval_mw.py --config "${TRAIN_ARGS[0]}" --workdir "$WD" "${TRAIN_ARGS[@]:1}" "$@"
python plot_mw.py --workdir "$WD"
