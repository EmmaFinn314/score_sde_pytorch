# Lint as: python3
"""5-d GMM data (gmm_setup.py) with a ~1M-parameter MLP score network, trained by train_gmm.py.

  python train_gmm.py --config configs/vp/gmm_mlp.py --workdir runs/gmm_mlp_n512 --config.data.num_train=512

Same VP SDE, loss, optimiser, EMA and checkpoint schedule as runs/mac_n512_nf64: batch 16, 100k steps,
20 log-spaced checkpoints from step 100. Each point is stored as a d x 1 x 1 "image".
"""
import os

import torch

from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = True
  training.reduce_mean = True
  training.likelihood_weighting = False
  training.batch_size = 16
  training.n_iters = 100000
  training.n_log_checkpoints = 20      # checkpoints at steps 100, 144, 207, ..., 100000
  training.first_log_checkpoint = 100
  training.log_freq = 100
  training.eval_freq = 500
  training.snapshot_freq_for_preemption = 10000
  training.snapshot_sampling = False

  # optimiser, as in the CIFAR runs
  optim = config.optim
  optim.lr = 1e-4
  optim.warmup = 0
  optim.grad_clip = 1.

  # data: gmm_setup.make_datasets(param_seed, data_seed); train set = its first num_train points
  data = config.data
  data.dataset = 'GMM'
  data.num_channels = 5                # dimension d
  data.image_size = 1
  data.centered = False
  data.random_flip = False
  data.num_train = 512
  data.num_test = 1000
  data.param_seed = 0
  data.data_seed = 0

  # model (beta_min = 0.1, beta_max = 20, num_scales = 1000: the defaults, as for CIFAR)
  model = config.model
  model.name = 'mlp'
  model.nf = 128                       # sinusoidal time-embedding size
  model.hidden = 512                   # width
  model.num_layers = 3                 # residual blocks; 3 x 512 gives ~1.1M parameters
  model.ema_rate = 0.9999
  model.dropout = 0.                   # not used by the MLP

  # A small MLP on batches of 16 is as fast on the CPU as on Apple's GPU (~2 ms/step); SCORE_SDE_DEVICE overrides.
  config.device = torch.device(os.environ.get('SCORE_SDE_DEVICE', 'cpu'))

  return config
