"""rho, rho_KL, log p_theta^SDE(O_1:R) and sliced Wasserstein for every checkpoint of a GMM run, and plots.

  python eval_gmm.py --config configs/vp/gmm_mlp.py --workdir runs/gmm_mlp_n512 --config.data.num_train=512
  python eval_gmm.py --workdir runs/gmm_mlp_n512 --plot_only

train_gmm.py runs this when training ends. Pass the same --config overrides the run was trained with
(as for eval_mw.py); the data come from <workdir>/data.npz.

  rho, rho_KL, log p_theta   on the test points noised along noisy_paths.log_time_grid(num_steps, t_min);
                             KL(P_mem || P_true) is computed once and cached in eval_paths/kl_mem.npz
  sw_test, sw_train          sliced Wasserstein (as in eval_mw.py: POT, p = 2, fixed unit directions)
                             between n_sw generated samples and the test set / the training set, on the
                             raw d-dim points. Samples come from eval_mw.py's samplers (probability-flow
                             ODE by default) with the same starting noise for every checkpoint.

Results are appended to eval_paths/results.csv (rerunning skips checkpoints already there) and drawn as
rho.png, rho_kl.png, loglik.png and sliced_wasserstein.png in eval_paths/.
"""
import csv
import json
import logging
import math
import os
import time

import numpy as np
import torch

import mw_metrics
import noisy_paths as npaths
import sde_lib
import utils as repo_utils
from gmm_setup import sample_gmm
from models import mlp  # noqa: F401  (registers the architecture)
from models import utils as mutils
from models.ema import ExponentialMovingAverage

FIELDS = ["ckpt", "step", "weights", "rho", "rho_kl", "rho_kl_se", "kl_theta", "kl_theta_se", "kl_mem",
          "loglik_theta", "loglik_theta_se", "loglik_true", "loglik_mem",
          "sw_test", "sw_train", "sw_true_test", "sw_true_train", "sw_mem_test", "sw_mem_train", "seconds"]
COLOR, MEM_COLOR, TEST_COLOR = "#4C72B0", "#C44E52", "#DD8452"


def load_weights(model, ckpt, which, config, device):
  """Put the raw or EMA weights of a checkpoint into `model` (as eval_mw.load_weights)."""
  model.load_state_dict(repo_utils._to_device(ckpt["model"], device))
  if which == "ema":
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    ema.load_state_dict(repo_utils._to_device(ckpt["ema"], device))
    ema.copy_to(model.parameters())
  model.eval()


def numpy_score_fn(model, sde, device):
  """The network's score as score_fn(x [n, d], t) -> [n, d] float64, the form noisy_paths takes."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)

  def f(x, t):
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).reshape(len(x), -1, 1, 1).to(device)
    with torch.no_grad():
      s = score_fn(xt, torch.full((len(x),), float(t), device=device))
    return s.reshape(len(x), -1).cpu().numpy().astype(np.float64)

  return f


@torch.no_grad()
def generate(model, sde, z, steps, sampler, seed, eps=1e-3):
  """Samples [n, d] (float64) from fixed starting noise z [n, d, 1, 1].

  The samplers of eval_mw.generate, without its conversion to 8-bit images: 'ode' is the
  probability-flow ODE with Euler steps from T to eps and a final denoising step
  E[x_0 | x_eps] = (x + sigma^2 s) / alpha; 'sde' is the repo's reverse-SDE Euler-Maruyama,
  returning the noise-free mean of the last step.
  """
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)
  x, n = z.clone(), z.shape[0]
  if sampler == "ode":
    ts = torch.linspace(sde.T, eps, steps + 1).tolist()
    for k in range(steps):
      t = torch.full((n,), ts[k], device=x.device)
      f, g = sde.sde(x, t)
      x = x + (f - 0.5 * (g ** 2)[:, None, None, None] * score_fn(x, t)) * (ts[k + 1] - ts[k])
    t = torch.full((n,), eps, device=x.device)
    alpha, sigma = sde.marginal_prob(torch.ones_like(x), t)
    x = (x + (sigma ** 2)[:, None, None, None] * score_fn(x, t)) / alpha
  else:
    torch.manual_seed(seed + 1000)
    rsde = sde.reverse(score_fn, probability_flow=False)
    dt = -1.0 / steps
    for t_val in torch.linspace(sde.T, eps, steps).tolist():
      t = torch.full((n,), t_val, device=x.device)
      drift, diffusion = rsde.sde(x, t)
      x_mean = x + drift * dt
      x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * torch.randn_like(x)
    x = x_mean
  return x.reshape(n, -1).cpu().numpy().astype(np.float64)


def evaluate(config, workdir, weights="ema", num_steps=50, t_min=1e-3, n_paths=10_000, seed=0,
             sampler="ode", sample_steps=0, n_sw=2000, n_proj=1000):
  """Append the metrics of each checkpoint to <workdir>/eval_paths/results.csv."""
  out_dir = os.path.join(workdir, "eval_paths")
  os.makedirs(out_dir, exist_ok=True)
  steps = sample_steps or config.model.num_scales
  settings = dict(num_steps=num_steps, t_min=t_min, n_paths=n_paths, seed=seed,
                  num_train=config.data.num_train, sampler=sampler, sample_steps=steps,
                  n_sw=n_sw, n_proj=n_proj)
  settings_path = os.path.join(out_dir, "settings.json")
  if os.path.exists(settings_path):
    with open(settings_path) as f:
      if json.load(f) != settings:
        raise ValueError(f"{out_dir} holds results for other settings ({settings_path}); "
                         "move it away to evaluate with these.")
  with open(settings_path, "w") as f:
    json.dump(settings, f, indent=1)

  device, bm = config.device, dict(beta_min=config.model.beta_min, beta_max=config.model.beta_max)
  data = dict(np.load(os.path.join(workdir, "data.npz")))
  n = config.data.num_train
  if f"x_train_{n}" not in data:
    raise ValueError(f"data.npz has no x_train_{n}: pass the --config overrides used for training.")
  pi, mu, Sigma, x_train, x_test = data["pi"], data["mu"], data["Sigma"], data[f"x_train_{n}"], data["x_test"]
  d = x_train.shape[1]

  # Validation paths and the quantities that don't depend on the model, fixed for the whole run
  ts = npaths.log_time_grid(num_steps, t_min)
  xs = npaths.noise_paths(x_test, ts, seed=seed, **bm)
  ll_true = npaths.path_loglik_gmm(xs, ts, pi, mu, Sigma, **bm)
  ll_mem = npaths.path_loglik_memorizing(xs, ts, x_train, **bm)
  kl_path = os.path.join(out_dir, "kl_mem.npz")
  if os.path.exists(kl_path):
    kl_mem = dict(np.load(kl_path))
  else:
    logging.info("Computing KL(P_mem || P_true) with %d paths (once per run)...", n_paths)
    kl_mem = npaths.memorizing_kl(ts, pi, mu, Sigma, x_train, n_paths, seed, **bm)
    np.savez(kl_path, **kl_mem)

  # Sliced Wasserstein exactly as eval_mw.py: POT, p = 2, one fixed set of unit directions
  import ot
  mw_metrics.check_sliced_wasserstein(lambda a, b: ot.sliced_wasserstein_distance(
    a, b, projections=ot.sliced.get_random_projections(a.shape[1], 1000, seed=seed), p=2))
  directions = ot.sliced.get_random_projections(d, n_proj, seed=seed)

  def sliced_wasserstein(a, b):
    return float(ot.sliced_wasserstein_distance(a, b, projections=directions, p=2))

  # References: n_sw fresh draws from the true GMM, and n_sw draws from the memorizing distribution
  true_samples = sample_gmm(pi, mu, Sigma, n_sw, seed=[seed, 2])[0]
  mem_samples = x_train[np.random.default_rng([seed, 3]).integers(len(x_train), size=n_sw)]
  sw_ref = dict(sw_true_test=sliced_wasserstein(true_samples, x_test),
                sw_true_train=sliced_wasserstein(true_samples, x_train),
                sw_mem_test=sliced_wasserstein(mem_samples, x_test),
                sw_mem_train=sliced_wasserstein(mem_samples, x_train))
  z = torch.randn((n_sw, d, 1, 1), generator=torch.Generator().manual_seed(seed)).to(device)

  with open(os.path.join(workdir, "checkpoints", "steps.json")) as f:
    ckpt_steps = {int(k): int(v) for k, v in json.load(f).items()}
  ckpt_path = lambda i: os.path.join(workdir, "checkpoints", f"checkpoint_{i}.pth")
  ckpts = [(i, s) for i, s in sorted(ckpt_steps.items()) if os.path.exists(ckpt_path(i))]
  results_path = os.path.join(out_dir, "results.csv")
  done = set()
  if os.path.exists(results_path):
    with open(results_path) as f:
      done = {(int(r["ckpt"]), r["weights"]) for r in csv.DictReader(f)}
  weight_sets = ["ema", "raw"] if weights == "both" else [weights]

  sde = sde_lib.VPSDE(N=config.model.num_scales, **bm)
  model = mutils.create_model(config)
  new_file = not os.path.exists(results_path)
  with open(results_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
      writer.writeheader()
    for idx, step in ckpts:
      todo = [w for w in weight_sets if (idx, w) not in done]
      if not todo:
        continue
      ckpt = torch.load(ckpt_path(idx), map_location="cpu", weights_only=False)
      for which in todo:
        t0 = time.time()
        load_weights(model, ckpt, which, config, device)
        score_fn = numpy_score_fn(model, sde, device)
        ll_theta = npaths.path_loglik_model(xs, ts, score_fn, **bm)
        try:
          rho = npaths.rho(ll_true, ll_mem, ll_theta)
        except ValueError as e:  # e.g. a non-finite log-likelihood from an early checkpoint
          logging.warning("ckpt %d (%s): rho not computed: %s", idx, which, e)
          rho = math.nan
        r = npaths.rho_kl(score_fn, ts, pi, mu, Sigma, kl_mem=kl_mem, **bm)
        samples = generate(model, sde, z, steps, sampler, seed)
        row = dict(ckpt=idx, step=step, weights=which, rho=rho, rho_kl=r["rho_kl"], rho_kl_se=r["se"],
                   kl_theta=r["kl_theta"], kl_theta_se=r["kl_theta_se"], kl_mem=r["kl_mem"],
                   loglik_theta=ll_theta.mean(), loglik_theta_se=ll_theta.std(ddof=1) / math.sqrt(len(xs)),
                   loglik_true=ll_true.mean(), loglik_mem=ll_mem.mean(),
                   sw_test=sliced_wasserstein(samples, x_test),
                   sw_train=sliced_wasserstein(samples, x_train), **sw_ref,
                   seconds=round(time.time() - t0, 1))
        writer.writerow(row)
        f.flush()
        logging.info("ckpt %d (step %d, %s): rho %.4f  rho_KL %.4f +- %.4f  log p_theta %.2f "
                     "(true %.2f, mem %.2f)  SW test %.4f train %.4f  [%ss]", idx, step, which, rho,
                     r["rho_kl"], r["se"], row["loglik_theta"], row["loglik_true"], row["loglik_mem"],
                     row["sw_test"], row["sw_train"], row["seconds"])
  logging.info("Done. Results in %s", results_path)


def plot(workdir):
  """One figure each for rho, rho_KL, mean log p_theta and sliced Wasserstein against training step."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  out_dir = os.path.join(workdir, "eval_paths")
  with open(os.path.join(out_dir, "results.csv")) as f:
    rows = sorted(csv.DictReader(f), key=lambda r: int(r["step"]))
  if not rows:
    raise SystemExit("No results yet.")
  with open(os.path.join(out_dir, "settings.json")) as f:
    s = json.load(f)
  n = s["num_train"]
  model_label = f"MLP score model, n_train = {n}"
  paths_label = f"{len(np.load(os.path.join(workdir, 'data.npz'))['x_test'])} validation paths, " \
                f"log grid t = {s['t_min']:g} to 1 ({s['num_steps'] + 1} times)"
  col = lambda rs, k: np.array([float(r[k]) for r in rs])
  weight_sets = sorted({r["weights"] for r in rows})
  by_weights = {w: [r for r in rows if r["weights"] == w] for w in weight_sets}
  style = lambda w: dict(marker="o", ms=3, lw=1.5, ls="-" if w == "ema" else "--")

  def finish(ax, ylabel):
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False)

  ref_rho = [(1.0, "true distribution (= 1)", "k"), (0.0, "memorizing distribution (= 0)", MEM_COLOR)]
  ref_ll = [(float(rows[0]["loglik_true"]), "true distribution", "k"),
            (float(rows[0]["loglik_mem"]), "memorizing distribution", MEM_COLOR)]
  specs = [("rho.png", "rho", None, r"$\rho$", "log-likelihood ratio", ref_rho),
           ("rho_kl.png", "rho_kl", "rho_kl_se", r"$\rho_{\mathrm{KL}}$",
            r"$1 - \mathrm{KL}(P_\theta \| P_{\mathrm{true}}) / \mathrm{KL}(P_{\mathrm{mem}} \| P_{\mathrm{true}})$, "
            r"$\pm$ 1 s.e.", ref_rho),
           ("loglik.png", "loglik_theta", "loglik_theta_se", r"mean $\log p_\theta^{\mathrm{SDE}}(O_{1:R})$",
            r"mean over validation paths, $\pm$ 1 s.e.", ref_ll)]
  for fname, key, se_key, label, subtitle, refs in specs:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for w, rs in by_weights.items():
      x, y = [max(int(r["step"]), 1) for r in rs], col(rs, key)
      ax.plot(x, y, color=COLOR, label=f"{'EMA' if w == 'ema' else 'raw'} weights", **style(w))
      if se_key:
        se = col(rs, se_key)
        ax.fill_between(x, y - se, y + se, color=COLOR, alpha=0.2, lw=0)
    for value, ref_label, color in refs:
      ax.axhline(value, color=color, ls=":", lw=1, label=ref_label)
    ax.set_title(f"{label}: {model_label}\n{subtitle}; {paths_label}", fontsize=9)
    finish(ax, label)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"wrote {os.path.join(out_dir, fname)}")

  # loglik_zoom.png: the same curves with the y-range set by the model and the true line only
  ll_true, ll_mem = ref_ll[0][0], ref_ll[1][0]
  fig, ax = plt.subplots(figsize=(7, 4.5))
  lo, hi = [ll_true], [ll_true]
  for w, rs in by_weights.items():
    x, y, se = [max(int(r["step"]), 1) for r in rs], col(rs, "loglik_theta"), col(rs, "loglik_theta_se")
    ax.plot(x, y, color=COLOR, label=f"{'EMA' if w == 'ema' else 'raw'} weights", **style(w))
    ax.fill_between(x, y - se, y + se, color=COLOR, alpha=0.2, lw=0)
    lo.append(np.nanmin(y - se))
    hi.append(np.nanmax(y + se))
  ax.axhline(ll_true, color="k", ls=":", lw=1, label="true distribution")
  pad = 0.05 * (max(hi) - min(lo))
  ax.set_ylim(min(lo) - pad, max(hi) + pad)
  if min(lo) - pad <= ll_mem <= max(hi) + pad:
    ax.axhline(ll_mem, color=MEM_COLOR, ls=":", lw=1, label="memorizing distribution")
  else:
    ax.text(0.02, 0.03, f"memorizing distribution: {ll_mem:.1f} (off scale)", transform=ax.transAxes,
            color=MEM_COLOR, fontsize=8)
  ax.set_title(f"mean $\\log p_\\theta^{{\\mathrm{{SDE}}}}(O_{{1:R}})$, zoomed: {model_label}\n"
               f"mean over validation paths, $\\pm$ 1 s.e.; {paths_label}", fontsize=9)
  finish(ax, r"mean $\log p_\theta^{\mathrm{SDE}}(O_{1:R})$")
  fig.tight_layout()
  fig.savefig(os.path.join(out_dir, "loglik_zoom.png"), dpi=150)
  plt.close(fig)
  print(f"wrote {os.path.join(out_dir, 'loglik_zoom.png')}")

  # loglik_gap.png: mean (log p_true - log p_theta) on a log scale. The validation paths are true data
  # noised forward, so this estimates KL(P_true || P_theta) on the grid (and rho = 1 - gap / gap_mem).
  fig, ax = plt.subplots(figsize=(7, 4.5))
  for w, rs in by_weights.items():
    x = [max(int(r["step"]), 1) for r in rs]
    gap, se = col(rs, "loglik_true") - col(rs, "loglik_theta"), col(rs, "loglik_theta_se")
    gap = np.where(gap > 0, gap, np.nan)  # a log axis can't show gaps <= 0
    floor = 0.5 * np.nanmin(gap) if np.any(np.isfinite(gap)) else 1e-3
    ax.plot(x, gap, color=COLOR, label=f"{'EMA' if w == 'ema' else 'raw'} weights", **style(w))
    ax.fill_between(x, np.maximum(gap - se, floor), gap + se, color=COLOR, alpha=0.2, lw=0)
  ax.axhline(ll_true - ll_mem, color=MEM_COLOR, ls=":", lw=1,
             label=f"memorizing distribution ({ll_true - ll_mem:.0f})")
  ax.set_yscale("log")
  ax.set_title(f"$\\log p_{{\\mathrm{{true}}}} - \\log p_\\theta^{{\\mathrm{{SDE}}}}$ "
               f"$\\approx \\mathrm{{KL}}(P_{{\\mathrm{{true}}}} \\| P_\\theta)$: {model_label}\n"
               f"mean over validation paths, $\\pm$ 1 s.e.; {paths_label}", fontsize=9)
  finish(ax, r"mean $\log p_{\mathrm{true}}(O_{1:R}) - \log p_\theta^{\mathrm{SDE}}(O_{1:R})$ (nats)")
  fig.tight_layout()
  fig.savefig(os.path.join(out_dir, "loglik_gap.png"), dpi=150)
  plt.close(fig)
  print(f"wrote {os.path.join(out_dir, 'loglik_gap.png')}")

  if "sw_test" in rows[0]:
    # One axes: colour = reference set (test / train), line style = weights (EMA solid, raw dashed);
    # thin reference lines in the same colour: true-GMM samples dotted, memorizing samples dash-dot
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for ref_set, color, ref_name in [("test", TEST_COLOR, "validation (test) set"),
                                     ("train", COLOR, f"training set (n = {n})")]:
      for w, rs in by_weights.items():
        ax.plot([max(int(r["step"]), 1) for r in rs], col(rs, f"sw_{ref_set}"), color=color,
                label=f"generated vs {ref_name}, {'EMA' if w == 'ema' else 'raw'} weights", **style(w))
      ax.axhline(float(rows[0][f"sw_true_{ref_set}"]), color=color, ls=":", lw=1,
                 label=f"true GMM samples vs {ref_name}")
      ax.axhline(float(rows[0][f"sw_mem_{ref_set}"]), color=color, ls="-.", lw=1,
                 label=f"memorizing samples vs {ref_name}")
    ax.set_title(f"Sliced Wasserstein: {model_label}\n{s['n_sw']} samples, {s['sampler']} sampler, "
                 f"{s['sample_steps']} steps; {s['n_proj']} directions", fontsize=9)
    finish(ax, "sliced Wasserstein-2")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sliced_wasserstein.png"), dpi=150)
    plt.close(fig)
    print(f"wrote {os.path.join(out_dir, 'sliced_wasserstein.png')}")


if __name__ == "__main__":
  from absl import app, flags
  from ml_collections.config_flags import config_flags

  FLAGS = flags.FLAGS
  config_flags.DEFINE_config_file("config", None, "Training config of the run.", lock_config=True)
  flags.DEFINE_string("workdir", None, "The run's working directory.")
  flags.DEFINE_enum("weights", "ema", ["ema", "raw", "both"], "Which weights to evaluate.")
  flags.DEFINE_integer("num_steps", 50, "Steps of the log-spaced time grid (num_steps + 1 times).")
  flags.DEFINE_float("t_min", 1e-3, "First time of the grid.")
  flags.DEFINE_integer("n_paths", 10_000, "Monte Carlo paths for the KLs in rho_KL.")
  flags.DEFINE_integer("seed", 0, "Seed for validation-path noise, KL paths, sampler noise, SW directions.")
  flags.DEFINE_enum("sampler", "ode", ["ode", "sde"], "Sampler for sliced Wasserstein (as eval_mw.py).")
  flags.DEFINE_integer("sample_steps", 0, "Sampling steps (0: config.model.num_scales, as eval_mw.py).")
  flags.DEFINE_integer("n_sw", 2000, "Generated samples for sliced Wasserstein (M&W: 2,000).")
  flags.DEFINE_integer("n_proj", 1000, "Random directions for sliced Wasserstein.")
  flags.DEFINE_boolean("plot_only", False, "Only redraw the figures from results.csv.")
  flags.mark_flags_as_required(["workdir"])

  def main(argv):
    if not FLAGS.plot_only:
      if FLAGS.config is None:
        raise SystemExit("--config is required unless --plot_only.")
      logging.getLogger().setLevel("INFO")
      evaluate(FLAGS.config, FLAGS.workdir, FLAGS.weights, FLAGS.num_steps, FLAGS.t_min,
               FLAGS.n_paths, FLAGS.seed, FLAGS.sampler, FLAGS.sample_steps, FLAGS.n_sw, FLAGS.n_proj)
    plot(FLAGS.workdir)

  app.run(main)
