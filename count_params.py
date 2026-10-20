"""Print how many parameters the score model in a config has (runs on the CPU in a few seconds).

    python count_params.py --config configs/vp/ddpm/cifar10_mw_n2048.py
    python count_params.py --config configs/vp/ddpm/cifar10_mw_n2048.py --config.model.nf=64

Training runs also log this count at the start (in the terminal and in <workdir>/stdout.txt).
"""
from collections import Counter

import torch
from absl import app, flags
from ml_collections.config_flags import config_flags

from models import ddpm, ncsnpp, ncsnv2  # noqa: F401  (importing registers the architectures)
from models import utils as mutils

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Config file describing the model.", lock_config=True)
flags.mark_flags_as_required(["config"])


def main(argv):
  config = FLAGS.config
  config.device = torch.device("cpu")          # counting needs no GPU
  model = mutils.get_model(config.model.name)(config)
  total = sum(p.numel() for p in model.parameters())
  m = config.model
  print(f"model {m.name}: nf={m.nf} ch_mult={tuple(m.ch_mult)} num_res_blocks={m.num_res_blocks} "
        f"attn_resolutions={tuple(m.attn_resolutions)}")
  print(f"parameters: {total:,} ({total / 1e6:.2f}M)")
  by_type = Counter()
  for module in model.modules():
    own = sum(p.numel() for p in module.parameters(recurse=False))
    if own:
      by_type[type(module).__name__] += own
  for name, n in by_type.most_common():
    print(f"  {name:<20} {n:>12,}  ({100 * n / total:.1f}%)")


if __name__ == "__main__":
  app.run(main)
