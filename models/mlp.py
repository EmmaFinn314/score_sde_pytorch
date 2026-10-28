"""MLP score network for low-dimensional data stored as d x 1 x 1 "images".

Batches have shape (B, d, 1, 1), so sde_lib, losses, sampling and get_score_fn work unchanged; the
MLP flattens its input and reshapes its output. Like DDPM it predicts the noise, which get_score_fn
turns into a score (-output / sigma_t). For continuous VP training it receives labels = 999 t.
"""
import torch.nn as nn

from . import layers, utils


@utils.register_model(name='mlp')
class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    d, nf, hidden = config.data.num_channels, config.model.nf, config.model.hidden
    self.nf = nf
    self.act = nn.SiLU()
    # Sinusoidal time embedding (as in DDPM) followed by a two-layer MLP
    self.temb = nn.Sequential(nn.Linear(nf, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
    self.inp = nn.Linear(d, hidden)
    self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(config.model.num_layers)])
    self.out = nn.Linear(hidden, d)

  def forward(self, x, labels):
    shape = x.shape
    temb = self.temb(layers.get_timestep_embedding(labels, self.nf))
    h = self.inp(x.reshape(shape[0], -1))
    for block in self.blocks:
      h = h + block(self.act(h + temb))  # residual block, time added at every layer
    return self.out(self.act(h)).reshape(shape)
