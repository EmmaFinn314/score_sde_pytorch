# Lint as: python3
"""DDPM U-Net shaped like Marion & Wu (2026), Appendix A, trained with the continuous VP SDE.

Save as configs/vp/ddpm/cifar10_mw_unet.py and run, e.g.
  python main.py --config configs/vp/ddpm/cifar10_mw_unet.py --workdir runs/mw_unet --mode train

What matches M&W:  128 base channels, channel multipliers (1, 2, 4), 3 residual blocks per level,
                   512-dim noise embedding (the DDPM U-Net always uses 4 * nf), EMA 0.9999,
                   batch 16, Adam lr 1e-4, 250 sampling steps.
What doesn't (needs code changes): class conditioning (M&W's 512-dim condition embedding),
                   64-dim attention heads (attention here is single-head over all channels),
                   rectified flow (this repo trains a VP SDE).
~79.9M parameters with attention at 16x16 and 8x8 (M&W report 92M); ~75.7M with attention at 16x16 only.
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

  # optimiser: M&W say "Adam with standard hyperparameters"; warmup and clipping are this repo's defaults
  optim = config.optim
  optim.lr = 1e-4                      # M&W default
  optim.warmup = 5000
  optim.grad_clip = 1.

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'

  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.name = 'ddpm'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999              # M&W
  model.normalization = 'GroupNorm'    # 32 groups, so nf must be a multiple of 32
  model.nonlinearity = 'swish'
  model.nf = 128                       # base channels (M&W); noise embedding is 4 * nf = 512
  model.ch_mult = (1, 2, 4)            # M&W; levels at 32x32, 16x16, 8x8 (two downsamplings)
  model.num_res_blocks = 3             # M&W
  model.attn_resolutions = (16, 8)     # M&W don't say where; a middle attention block is always added
  model.dropout = 0.1                  # M&W don't say; repo default
  model.resamp_with_conv = True
  model.conditional = True             # conditioning on the noise level (not on class labels)
  model.num_scales = 250               # number of sampler steps; continuous training doesn't use it

  return config
