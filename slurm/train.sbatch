#!/bin/bash
# Train one Marion & Wu-style run. First argument: a run name (the run goes in $RUNS/<name>); any
# further arguments go to main.py, e.g. config overrides. Submit from the repository root:
#     sbatch slurm/train.sbatch mw_n2048
#     sbatch slurm/train.sbatch mw_n4096 --config.data.num_train=4096
# If the job runs out of time, submit the same command again: training resumes from the last
# 10,000-step checkpoint (checkpoints-meta/). Change the GPU type from the command line, e.g.
#     sbatch --gres=gpu:a40:1 slurm/train.sbatch mw_n2048
#SBATCH --job-name=sde-train
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your.name@stats.ox.ac.uk

source slurm/env.sh
set -eo pipefail
RUN="${1:?usage: sbatch slurm/train.sbatch RUN_NAME [main.py arguments]}"
shift
CONFIG="${CONFIG:-configs/vp/ddpm/cifar10_mw_n2048.py}"
WD="$RUNS/$RUN"
mkdir -p "$WD"

# Record the config and overrides, so slurm/eval.sbatch rebuilds exactly the same model.
printf '%s\n' "$CONFIG" "$@" > "$WD/train_args.txt"

echo "run $RUN -> $WD | node $(hostname) | GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python main.py --config "$CONFIG" --workdir "$WD" --mode train "$@"
