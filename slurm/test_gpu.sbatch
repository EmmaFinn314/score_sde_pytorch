#!/bin/bash
# End-to-end test on one GPU (about 15 minutes): checks the GPU and the environment, trains for
# 300 steps with 2 checkpoints, evaluates both with small settings and draws the plot.
# Submit from the repository root:
#     sbatch slurm/test_gpu.sbatch
# Output: slurm/logs/sde-test_<jobid>.out, and the test run in $RUNS/test_<jobid>.
#SBATCH --job-name=sde-test
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/%x_%j.out

source slurm/env.sh
set -eo pipefail
CONFIG=configs/vp/ddpm/cifar10_mw_n2048.py
WD="$RUNS/test_$SLURM_JOB_ID"

echo "== node $(hostname)"
nvidia-smi

echo "== PyTorch on the GPU"
python - <<'PY'
import torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda,
      "| GPU available:", torch.cuda.is_available())
conv = torch.nn.Conv2d(128, 128, 3, padding=1).cuda()
y = conv(torch.randn(16, 128, 32, 32, device="cuda"))
torch.cuda.synchronize()
print("GPU:", torch.cuda.get_device_name(0), "| test convolution OK", tuple(y.shape))
PY

echo "== check_env.py"
python check_env.py || echo "check_env.py reported failures (see above); carrying on with the test run"

echo "== training: 300 steps, checkpoints at steps 100 and 300"
python main.py --config "$CONFIG" --workdir "$WD" --mode train \
  --config.training.n_iters=300 --config.training.n_log_checkpoints=2

echo "== evaluation (small settings)"
python eval_mw.py --config "$CONFIG" --workdir "$WD" \
  --sample_steps=10 --n_fid_train=256 --n_fid_test=256 --n_sw=256 --batch=128
python plot_mw.py --workdir "$WD"

echo "== timing: training speed of this GPU"
grep "training_loss" "$WD/stdout.txt" | sed -n '1p;$p'
echo "== done: results in $WD/eval_mw"
