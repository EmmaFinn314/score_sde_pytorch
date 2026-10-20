# Lint as: python3
"""Marion & Wu (2026)-style default run: first 2,048 CIFAR-10 images, no flips, ~80M-parameter U-Net.

  python main.py --config configs/vp/ddpm/cifar10_mw_n2048.py --workdir runs/mw_n2048 --mode train

Change the subset size from the command line, e.g. --config.data.num_train=4096.
Same network as cifar10_mw_unet.py (see that file for what does and doesn't match M&W).
Checkpoints: 30 log-spaced ones from step 100 to 300,000, as checkpoints/checkpoint_1.pth ...
checkpoint_30.pth, with checkpoints/steps.json giving the step of each.
"""

from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = True
  training.reduce_mean = True
  training.batch_size = 16             # M&W default
  training.n_iters = 300000            # M&W's Fig. 1b runs to ~3e5 steps
  training.n_log_checkpoints = 30      # log-spaced checkpoints, to resolve the memorization transition
  training.first_log_checkpoint = 100
  training.snapshot_sampling = False   # sample in a separate evaluation pass instead
  training.log_freq = 100
  training.eval_freq = 500

  # optimiser: M&W say "Adam with standard hyperparameters"
  optim = config.optim
  optim.lr = 1e-4                      # M&W default
  optim.warmup = 0                     # this repo defaults to 5,000 steps, which would cover the
                                       # 1e3-1e4 window where the transition happens; M&W don't say
  optim.grad_clip = 1.                 # repo default; M&W don't say

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'

  # data
  data = config.data
  data.centered = True
  data.num_train = 2048                # first N training images (in TF Datasets' order)
  data.num_test = 5120                 # M&W's fixed evaluation set size
  data.random_flip = False             # no augmentation, so memorization is of the N images themselves

  # model
  model = config.model
  model.name = 'ddpm'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999              # M&W
  model.normalization = 'GroupNorm'    # 32 groups, so nf must be a multiple of 32
  model.nonlinearity = 'swish'
  model.nf = 128                       # base channels (M&W); noise embedding is 4 * nf = 512
  model.ch_mult = (1, 2, 4)            # M&W
  model.num_res_blocks = 3             # M&W
  model.attn_resolutions = (16, 8)     # M&W don't say; single-head attention
  model.dropout = 0.1                  # M&W don't say; repo default
  model.resamp_with_conv = True
  model.conditional = True             # conditioning on the noise level (not on class labels)
  model.num_scales = 250               # number of sampler steps

  return config
