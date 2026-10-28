"""Train the MLP score network on GMM data (gmm_setup.py) with the repo's VP-SDE training code.

  python train_gmm.py --config configs/vp/gmm_mlp.py --workdir runs/gmm_mlp_n512 --config.data.num_train=512

Mirrors run_lib.train: the same model creation, EMA, optimiser, DSM step, log lines and checkpoint
layout (log-spaced checkpoints/checkpoint_<i>.pth with checkpoints/steps.json; checkpoints-meta for
resuming). Points are (B, d, 1, 1) "images", so none of that code changes. The data trained on are
saved to <workdir>/data.npz. When training ends, eval_gmm.py computes rho, rho_KL and log p_theta for
every checkpoint and plots them (skip with --noeval_after).
"""
import functools
import json
import logging
import os

import numpy as np
import torch
from absl import app, flags
from ml_collections.config_flags import config_flags

import eval_gmm
import losses
import sde_lib
from gmm_setup import make_datasets
from models import mlp  # noqa: F401  (registers the architecture)
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint, save_checkpoint

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_boolean("eval_after", True, "Evaluate every checkpoint with eval_gmm.py and plot when done.")
flags.DEFINE_enum("eval_weights", "ema", ["ema", "raw", "both"], "Weights to evaluate when done.")
flags.mark_flags_as_required(["config", "workdir"])


def main(argv):
  config, workdir = FLAGS.config, FLAGS.workdir
  os.makedirs(workdir, exist_ok=True)
  # Log to the terminal and to <workdir>/stdout.txt, as main.py does
  handler = logging.StreamHandler(open(os.path.join(workdir, "stdout.txt"), "a"))
  handler.setFormatter(logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s"))
  logging.getLogger().addHandler(handler)
  logging.getLogger().setLevel("INFO")
  torch.manual_seed(config.seed)

  # Data: the first num_train points of the GMM training pool, and the test set
  n = config.data.num_train
  data = make_datasets(config.data.param_seed, config.data.data_seed, n_test=config.data.num_test,
                       train_sizes=(n,), d=config.data.num_channels)
  np.savez(os.path.join(workdir, "data.npz"), **data)
  as_images = lambda a: torch.from_numpy(a).float().reshape(len(a), -1, 1, 1).to(config.device)
  x_train, x_test = as_images(data[f"x_train_{n}"]), as_images(data["x_test"])

  # Model, EMA, optimiser and resumable state, as in run_lib.train
  score_model = mutils.create_model(config)
  n_params = sum(p.numel() for p in score_model.parameters())
  logging.info("Score model %s: %d parameters (%.2fM)" % (config.model.name, n_params, n_params / 1e6))
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  meta_path = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  os.makedirs(checkpoint_dir, exist_ok=True)
  os.makedirs(os.path.dirname(meta_path), exist_ok=True)
  state = restore_checkpoint(meta_path, state, config.device)
  initial_step = int(state["step"])

  sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                      N=config.model.num_scales)
  optimize_fn = losses.optimization_manager(config)
  kw = dict(reduce_mean=config.training.reduce_mean, continuous=config.training.continuous,
            likelihood_weighting=config.training.likelihood_weighting)
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn, **kw)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn, **kw)

  # Log-spaced checkpoints numbered 1, 2, ...: the same formula and steps.json as run_lib.train
  num_train_steps = config.training.n_iters
  log_steps = np.unique(np.round(np.logspace(np.log10(config.training.first_log_checkpoint),
                                             np.log10(num_train_steps),
                                             config.training.n_log_checkpoints)).astype(int))
  ckpt_index = {int(s): i + 1 for i, s in enumerate(log_steps)}
  with open(os.path.join(checkpoint_dir, "steps.json"), "w") as f:
    json.dump({str(i): s for s, i in ckpt_index.items()}, f, indent=1)
  logging.info("Saving %d log-spaced checkpoints at steps %s" % (len(log_steps), [int(s) for s in log_steps]))

  # Batches: reshuffle every epoch, each epoch's order seeded by (seed, epoch), so the batch of a
  # given step is the same on a resumed run
  batch_size = config.training.batch_size
  epoch_order = functools.lru_cache(maxsize=2)(
    lambda epoch: np.random.default_rng([config.seed, epoch]).permutation(n))

  def get_batch(step):
    pos = np.arange(step * batch_size, (step + 1) * batch_size)
    idx = [epoch_order(int(e))[i] for e, i in zip(*np.divmod(pos, n))]
    return x_train[torch.as_tensor(idx, device=x_train.device)]

  logging.info("Starting training loop at step %d." % (initial_step,))
  for step in range(initial_step, num_train_steps + 1):
    loss = train_step_fn(state, get_batch(step))
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(meta_path, state)
    if step % config.training.eval_freq == 0:
      logging.info("step: %d, eval_loss: %.5e" % (step, eval_step_fn(state, x_test).item()))
    if step in ckpt_index:
      save_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_{ckpt_index[step]}.pth"), state)

  if FLAGS.eval_after:
    eval_gmm.evaluate(config, workdir, weights=FLAGS.eval_weights)
    eval_gmm.plot(workdir)


if __name__ == "__main__":
  app.run(main)
