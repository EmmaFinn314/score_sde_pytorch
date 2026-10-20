"""Evaluate a run's checkpoints the way Marion & Wu (2026) do: the four panels of their Fig. 1b.

For every checkpoint in <workdir>/checkpoints (EMA weights by default) this records:

  loss_train, loss_test   the training objective (denoising score matching) on all N training images
                          and on the test reference set, with the same noise and times every time
  mem_sscd                M&W's memorization score: fraction of generated samples whose largest
                          cosine similarity to a training image exceeds 0.6 in SSCD space
  sscd_maxsim_mean        mean of that largest cosine similarity (a continuous version of the above)
  mem_knn                 pixel-space cross-check: fraction with d1/d2 < 1/3 to the training images
  sw_train, sw_test       sliced Wasserstein distance (POT, p = 2, fixed unit directions) between
                          2,000 generated samples and each reference set, on Inception pool_3 features
  fid_train, fid_test     FID with as many generated samples as the reference set has images

Reference sets, as in M&W: the N training images the run was trained on (data.num_train) and the
first data.num_test test images (5,120 if unset). One set of samples per checkpoint is reused for all
metrics; the starting noise is the same for every checkpoint, so samples are comparable across training.

Results are appended to <workdir>/eval_mw/results.csv, and rerunning skips checkpoints already done,
so an interrupted evaluation can simply be restarted. Plot with plot_mw.py.

  python eval_mw.py --config configs/vp/ddpm/cifar10_mw_n2048.py --workdir runs/mw_n2048

Pass exactly the --config and --config.* overrides the run was trained with.
Quick test (a few minutes on a laptop):
  python eval_mw.py --config configs/vp/ddpm/cifar10_mw_n2048.py --workdir runs/n2048_smoke \\
      --ckpts=1 --sample_steps=10 --n_fid_train=256 --n_fid_test=256 --n_sw=256 --batch=128

Where M&W's protocol isn't specified (or can't be matched here), the choice made is noted in
README_eval_mw.md; the main ones are the sampler (probability-flow ODE, Euler steps, standing in for
their rectified-flow sampler) and the Inception network (TF-GAN's, as used for FID in this repo).
"""
import csv
import glob
import json
import os
import re
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import torch.nn.functional as F
from absl import app, flags, logging
from ml_collections.config_flags import config_flags
from torchvision.utils import save_image

import datasets as repo_datasets
import evaluation
import losses
import mw_metrics
import sde_lib
import tfgan_metrics
import utils as repo_utils
from models import ddpm, ncsnpp, ncsnv2  # noqa: F401  (registers the architectures)
from models import utils as mutils
from models.ema import ExponentialMovingAverage

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training config of the run.", lock_config=True)
flags.DEFINE_string("workdir", None, "The run's working directory (contains checkpoints/).")
flags.DEFINE_enum("weights", "ema", ["ema", "raw", "both"], "Which weights to evaluate.")
flags.DEFINE_enum("sampler", "ode", ["ode", "sde"],
                  "ode: probability-flow ODE with Euler steps (deterministic, like rectified-flow "
                  "sampling); sde: reverse-SDE Euler-Maruyama, the repo's default sampler.")
flags.DEFINE_integer("sample_steps", 0, "Sampling steps (0: config.model.num_scales; M&W use 250).")
flags.DEFINE_integer("n_fid_train", 0, "Samples for FID vs train (0: N, the training-set size, as M&W).")
flags.DEFINE_integer("n_fid_test", 0, "Samples for FID vs test (0: the test reference size, as M&W).")
flags.DEFINE_integer("n_sw", 2000, "Samples for sliced Wasserstein (M&W: 2,000).")
flags.DEFINE_integer("n_proj", 1000, "Random directions for sliced Wasserstein (M&W don't say).")
flags.DEFINE_integer("batch", 512, "Batch size for sampling and feature extraction.")
flags.DEFINE_string("sscd_path", "", "Path to sscd_disc_mixup.torchscript.pt (default: download "
                    "to ~/.cache/sscd the first time).")
flags.DEFINE_boolean("no_sscd", False, "Skip the SSCD memorization score.")
flags.DEFINE_list("ckpts", [], "Only these checkpoint numbers, e.g. --ckpts=1,10,30 (default: all).")
flags.DEFINE_integer("every", 1, "Evaluate every k-th checkpoint.")
flags.DEFINE_integer("seed", 0, "Seed for the starting noise, loss noise and SW directions.")
flags.DEFINE_string("results", "results.csv", "Output file name inside <workdir>/eval_mw.")
flags.mark_flags_as_required(["config", "workdir"])

SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
FIELDS = ["ckpt", "step", "weights", "sampler", "sample_steps", "n_train_ref", "n_test_ref", "n_gen",
          "loss_train", "loss_test", "mem_sscd", "sscd_maxsim_mean", "mem_knn",
          "sw_train", "sw_test", "fid_train", "fid_test", "seconds"]


# --------------------------------------------------------------------------- data and features
def load_split(split):
  """uint8 images [n, 32, 32, 3] of a TF Datasets CIFAR-10 split such as 'train[:2048]'."""
  data = tfds.as_numpy(tfds.load("cifar10", split=split, batch_size=-1))
  return np.asarray(data["image"], dtype=np.uint8)


class InceptionFeatures:
  """pool_3 features (2048-d) from TF-GAN's Inception network, the one this repo uses for FID."""

  def __init__(self, batch):
    self.model = evaluation.get_inception_model(inceptionv3=False)
    self.batch = batch

  def __call__(self, images_uint8):
    feats = []
    for i in range(0, len(images_uint8), self.batch):
      out = evaluation.run_inception_distributed(images_uint8[i:i + self.batch], self.model,
                                                 inceptionv3=False)
      feats.append(out["pool_3"].numpy().astype(np.float64))
    return np.concatenate(feats)


class SSCDFeatures:
  """SSCD copy-detection descriptors (Pizzi et al. 2022), as in Somepalli et al. 2023 and M&W."""

  def __init__(self, path, device, batch=128):
    if not path:
      path = os.path.join(os.path.expanduser("~"), ".cache", "sscd", "sscd_disc_mixup.torchscript.pt")
      if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logging.info("Downloading SSCD model to %s", path)
        torch.hub.download_url_to_file(SSCD_URL, path)
    self.model = torch.jit.load(path, map_location=device).eval()
    self.device, self.batch = device, batch
    self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

  @torch.no_grad()
  def __call__(self, images_uint8):
    out = []
    for i in range(0, len(images_uint8), self.batch):
      x = torch.from_numpy(images_uint8[i:i + self.batch]).to(self.device)
      x = x.permute(0, 3, 1, 2).float() / 255.0
      # SSCD's recommended preprocessing: resize to 288 and ImageNet normalisation
      x = F.interpolate(x, size=(288, 288), mode="bicubic", align_corners=False).clamp(0, 1)
      out.append(self.model((x - self.mean) / self.std).cpu().numpy().astype(np.float64))
    return np.concatenate(out)


# --------------------------------------------------------------------------- model
def load_weights(model, ckpt, which, config, device):
  """Put the raw or EMA weights of a checkpoint into `model` (checking it matches the config)."""
  expected, found = set(model.state_dict()), set(ckpt["model"])
  if expected != found:
    raise ValueError(
      "This checkpoint doesn't match the model described by --config "
      f"({len(expected ^ found)} mismatched weight names). Use the same --config and "
      "--config.* overrides as for training.")
  model.load_state_dict(repo_utils._to_device(ckpt["model"], device))
  if which == "ema":
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    ema.load_state_dict(repo_utils._to_device(ckpt["ema"], device))
    ema.copy_to(model.parameters())
  model.eval()


@torch.no_grad()
def objective(model, sde, images_uint8, scaler, config, device, batch, seed):
  """Mean training loss over a set of images, with noise and times fixed by `seed`."""
  loss_fn = losses.get_sde_loss_fn(sde, train=False, reduce_mean=config.training.reduce_mean,
                                   continuous=True,
                                   likelihood_weighting=config.training.likelihood_weighting)
  torch.manual_seed(seed)
  total = 0.0
  for i in range(0, len(images_uint8), batch):
    x = torch.from_numpy(images_uint8[i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
    total += loss_fn(model, scaler(x)).item() * len(x)
  return total / len(images_uint8)


@torch.no_grad()
def generate(model, sde, z, config, device, steps, sampler, batch, seed, eps=1e-3):
  """Samples as uint8 [n, 32, 32, 3] from fixed starting noise z (same for every checkpoint)."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)
  inverse_scaler = repo_datasets.get_data_inverse_scaler(config)
  out = []
  for b, i in enumerate(range(0, len(z), batch)):
    x = z[i:i + batch].to(device)
    n = x.shape[0]
    if sampler == "ode":
      # Probability-flow ODE, dx = [f - g^2 s / 2] dt, Euler steps from t = T to eps,
      # then the denoising step E[x_0 | x_eps] = (x + sigma^2 s) / alpha.
      ts = torch.linspace(sde.T, eps, steps + 1).tolist()
      for k in range(steps):
        t = torch.full((n,), ts[k], device=device)
        f, g = sde.sde(x, t)
        x = x + (f - 0.5 * (g ** 2)[:, None, None, None] * score_fn(x, t)) * (ts[k + 1] - ts[k])
      t = torch.full((n,), eps, device=device)
      alpha, sigma = sde.marginal_prob(torch.ones_like(x), t)
      x = (x + (sigma ** 2)[:, None, None, None] * score_fn(x, t)) / alpha
    else:
      # The repo's Euler-Maruyama predictor (sampling.EulerMaruyamaPredictor), returning the
      # noise-free mean of the last step.
      torch.manual_seed(seed + 1000 + b)
      rsde = sde.reverse(score_fn, probability_flow=False)
      dt = -1.0 / steps
      for t_val in torch.linspace(sde.T, eps, steps).tolist():
        t = torch.full((n,), t_val, device=device)
        drift, diffusion = rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * torch.randn_like(x)
      x = x_mean
    x = inverse_scaler(x).clamp(0, 1)
    out.append((x * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
  return np.concatenate(out)


# --------------------------------------------------------------------------- bookkeeping
def list_checkpoints(workdir, snapshot_freq):
  """[(number, step, path)] for every checkpoint_<number>.pth, with steps from steps.json if present."""
  ckpt_dir = os.path.join(workdir, "checkpoints")
  steps_path = os.path.join(ckpt_dir, "steps.json")
  steps = {}
  if os.path.exists(steps_path):
    with open(steps_path) as f:
      steps = {int(k): int(v) for k, v in json.load(f).items()}
  out = []
  for path in glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pth")):
    m = re.search(r"checkpoint_(\d+)\.pth$", path)
    if m:
      i = int(m.group(1))
      out.append((i, steps.get(i, i * snapshot_freq), path))
  return sorted(out)


def done_rows(path):
  if not os.path.exists(path):
    return set()
  with open(path) as f:
    return {(int(r["ckpt"]), r["weights"]) for r in csv.DictReader(f)}


def main(argv):
  config = FLAGS.config
  device = config.device
  if config.training.sde.lower() != "vpsde" or not config.training.continuous:
    raise ValueError("eval_mw.py supports continuous-time VP-SDE runs (training.sde='vpsde').")
  # TensorFlow runs Inception: let it use a GPU if it can see one, but only take memory as needed.
  for gpu in tf.config.list_physical_devices("GPU"):
    try:
      tf.config.experimental.set_memory_growth(gpu, True)
    except (RuntimeError, ValueError):
      pass
  feat_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  out_dir = os.path.join(FLAGS.workdir, "eval_mw")
  os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
  results_path = os.path.join(out_dir, FLAGS.results)
  with open(os.path.join(out_dir, "config_used.txt"), "w") as f:
    f.write(str(config))

  # ---- sliced Wasserstein via POT, with one fixed set of unit directions for everything
  import ot
  mw_metrics.check_sliced_wasserstein(lambda a, b: ot.sliced_wasserstein_distance(
    a, b, projections=ot.sliced.get_random_projections(a.shape[1], 1000, seed=FLAGS.seed), p=2))
  directions = ot.sliced.get_random_projections(2048, FLAGS.n_proj, seed=FLAGS.seed)

  def sliced_wasserstein(a, b):
    return float(ot.sliced_wasserstein_distance(a, b, projections=directions, p=2))

  # ---- reference sets: the N training images of the run, and the test reference set
  n_train = config.data.num_train or 50000
  n_test = config.data.num_test or 5120
  train_imgs = load_split(f"train[:{n_train}]")
  test_imgs = load_split(f"test[:{n_test}]")
  n_fid_train = FLAGS.n_fid_train or n_train
  n_fid_test = FLAGS.n_fid_test or n_test
  n_gen = max(n_fid_train, n_fid_test, FLAGS.n_sw)
  steps = FLAGS.sample_steps or config.model.num_scales
  logging.info("Reference sets: %d train, %d test images; %d samples per checkpoint, %s sampler, "
               "%d steps", n_train, n_test, n_gen, FLAGS.sampler, steps)

  inception = InceptionFeatures(FLAGS.batch)
  sscd = None if FLAGS.no_sscd else SSCDFeatures(FLAGS.sscd_path, feat_device)
  ref_path = os.path.join(out_dir, f"reference_n{n_train}_t{n_test}.npz")
  if os.path.exists(ref_path):
    ref = dict(np.load(ref_path))
  else:
    logging.info("Computing reference features (once per run)...")
    ref = {"inc_train": inception(train_imgs), "inc_test": inception(test_imgs)}
    np.savez(ref_path, **ref)
  if sscd is not None and "sscd_train" not in ref:
    ref["sscd_train"] = sscd(train_imgs)
    np.savez(ref_path, **ref)
  train_pixels = train_imgs.astype(np.float32) / 255.0

  # ---- model, SDE, fixed starting noise
  sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=steps)
  model = mutils.create_model(config)
  scaler = repo_datasets.get_data_scaler(config)
  z = torch.randn((n_gen, config.data.num_channels, config.data.image_size, config.data.image_size),
                  generator=torch.Generator().manual_seed(FLAGS.seed))

  ckpts = list_checkpoints(FLAGS.workdir, config.training.snapshot_freq)
  if FLAGS.ckpts:
    wanted = {int(c) for c in FLAGS.ckpts}
    ckpts = [c for c in ckpts if c[0] in wanted]
  ckpts = ckpts[::FLAGS.every]
  weight_sets = ["ema", "raw"] if FLAGS.weights == "both" else [FLAGS.weights]
  finished = done_rows(results_path)
  todo = [c for c in ckpts if any((c[0], w) not in finished for w in weight_sets)]
  logging.info("%d checkpoints found, %d to evaluate", len(ckpts), len(todo))

  new_file = not os.path.exists(results_path)
  with open(results_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
      writer.writeheader()
    for idx, step, path in todo:
      ckpt = torch.load(path, map_location="cpu", weights_only=False)
      for which in weight_sets:
        if (idx, which) in finished:
          continue
        t0 = time.time()
        load_weights(model, ckpt, which, config, device)
        loss_train = objective(model, sde, train_imgs, scaler, config, device, FLAGS.batch, FLAGS.seed + 1)
        loss_test = objective(model, sde, test_imgs, scaler, config, device, FLAGS.batch, FLAGS.seed + 2)
        samples = generate(model, sde, z, config, device, steps, FLAGS.sampler, FLAGS.batch, FLAGS.seed)
        if torch.cuda.is_available():
          torch.cuda.empty_cache()   # hand cached GPU memory back so TensorFlow's Inception can use it
        save_image(torch.from_numpy(samples[:64]).permute(0, 3, 1, 2).float() / 255.0,
                   os.path.join(out_dir, "samples", f"{which}_ckpt{idx:03d}_step{step}.png"), nrow=8)

        inc = inception(samples)
        mem_sscd = maxsim_mean = float("nan")
        if sscd is not None:
          mem_sscd, sims = mw_metrics.memorization_fraction_cosine(sscd(samples), ref["sscd_train"])
          maxsim_mean = float(sims.mean())
        mem_knn, _ = mw_metrics.memorization_fraction_knn(samples.astype(np.float32) / 255.0,
                                                          train_pixels)
        row = {
          "ckpt": idx, "step": step, "weights": which, "sampler": FLAGS.sampler,
          "sample_steps": steps, "n_train_ref": n_train, "n_test_ref": n_test, "n_gen": n_gen,
          "loss_train": loss_train, "loss_test": loss_test,
          "mem_sscd": mem_sscd, "sscd_maxsim_mean": maxsim_mean, "mem_knn": mem_knn,
          "sw_train": sliced_wasserstein(inc[:FLAGS.n_sw], ref["inc_train"]),
          "sw_test": sliced_wasserstein(inc[:FLAGS.n_sw], ref["inc_test"]),
          "fid_train": tfgan_metrics.frechet_classifier_distance_from_activations(
            inc[:n_fid_train], ref["inc_train"]),
          "fid_test": tfgan_metrics.frechet_classifier_distance_from_activations(
            inc[:n_fid_test], ref["inc_test"]),
          "seconds": round(time.time() - t0, 1),
        }
        writer.writerow(row)
        f.flush()
        logging.info("ckpt %d (step %d, %s): loss %.4f/%.4f  mem %.3f (knn %.3f)  SW %.3f/%.3f  "
                     "FID %.1f/%.1f  [%ss]", idx, step, which, loss_train, loss_test, mem_sscd,
                     mem_knn, row["sw_train"], row["sw_test"], row["fid_train"], row["fid_test"],
                     row["seconds"])
  logging.info("Done. Results in %s", results_path)


if __name__ == "__main__":
  app.run(main)
