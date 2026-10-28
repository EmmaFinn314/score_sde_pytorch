"""Overlay one metric's training curves for every GMM run on one plot, coloured by training-set size.

    python plot_gmm_sweep.py                              # rho_KL, EMA weights, runs/gmm_mlp_n*
    python plot_gmm_sweep.py --metric loglik              # mean log p_theta, zoomed (as loglik_zoom.png)
    python plot_gmm_sweep.py --metric loglik_gap          # log p_true - log p_theta, log scale
    python plot_gmm_sweep.py --metric rho --weights raw

Reads <run>/eval_paths/results.csv and settings.json from eval_gmm.py for every run matching --runs,
and writes <out>/<metric>_<weights>.png. Training-set size is ordinal, so the colours are one blue ramp,
light (small n) to dark (large n). For rho and rho_KL, points below --ymin are drawn as triangles on
the bottom edge.
"""
import argparse
import csv
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Ordinal blue ramp (dataviz reference palette, steps 250 -> 700): monotone in lightness, and the
# lightest step still has >= 2:1 contrast on white
BLUES = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
         "#1c5cab", "#184f95", "#104281", "#0d366b"]
MUTED, MEM_COLOR = "#6b6b6b", "#C44E52"

col = lambda k: (lambda r: float(r[k]))  # noqa: E731
LOGLIK = r"mean $\log p_\theta^{\mathrm{SDE}}(O_{1:R})$"
METRICS = {
  "rho_kl": dict(value=col("rho_kl"), se=col("rho_kl_se"), label=r"$\rho_{\mathrm{KL}}$", ymin=-0.5,
                 refs=[(1.0, "true score (= 1)", "k"), (0.0, "memorizing score (= 0)", MEM_COLOR)]),
  "rho": dict(value=col("rho"), se=None, label=r"$\rho$", ymin=-0.5,
              refs=[(1.0, "true distribution (= 1)", "k"), (0.0, "memorizing distribution (= 0)", MEM_COLOR)]),
  "loglik": dict(value=col("loglik_theta"), se=col("loglik_theta_se"), label=LOGLIK,
                 ymin=math.nan, refs=[]),
  "loglik_gap": dict(value=lambda r: float(r["loglik_true"]) - float(r["loglik_theta"]),
                     se=col("loglik_theta_se"), ymin=math.nan, refs=[], log=True,
                     label=r"mean $\log p_{\mathrm{true}} - \log p_\theta^{\mathrm{SDE}}$ (nats)"),
}


def ramp(k):
    """k colours from the blue ramp, evenly spaced from light to dark."""
    return [BLUES[i] for i in np.round(np.linspace(0, len(BLUES) - 1, k)).astype(int)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs/gmm_mlp_n*", help="glob for the run directories")
    p.add_argument("--metric", default="rho_kl", choices=sorted(METRICS))
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--ymin", type=float, default=None,
                   help="lower y limit for rho / rho_kl (default -0.5; nan: automatic)")
    p.add_argument("--out", default="runs/gmm_mlp_sweep")
    args = p.parse_args()
    m = METRICS[args.metric]
    ymin = m["ymin"] if args.ymin is None else args.ymin
    log = m.get("log", False)

    runs = []
    for d in glob.glob(args.runs):
        results, settings = (os.path.join(d, "eval_paths", f) for f in ("results.csv", "settings.json"))
        if not os.path.exists(results):
            continue
        with open(settings) as f:
            s = json.load(f)
        with open(results) as f:
            rows = sorted((r for r in csv.DictReader(f) if r["weights"] == args.weights),
                          key=lambda r: int(r["step"]))
        if rows:
            runs.append((s["num_train"], s, rows, d))
    if not runs:
        raise SystemExit(f"No results for {args.weights} weights in {args.runs}.")
    runs.sort(key=lambda r: r[0])
    s0 = {k: v for k, v in runs[0][1].items() if k != "num_train"}
    if any({k: v for k, v in s.items() if k != "num_train"} != s0 for _, s, _, _ in runs):
        print("warning: the runs were evaluated with different settings; curves may not be comparable")
    ll_true = {float(rows[0]["loglik_true"]) for _, _, rows, _ in runs}
    if args.metric.startswith("loglik") and len(ll_true) > 1:
        print("warning: log p_true differs between runs (different test paths?)")

    fig, ax = plt.subplots(figsize=(9, 5))
    clipped, lo, hi = False, [], [v for v, _, _ in m["refs"]]
    for (n, _, rows, _), color in zip(runs, ramp(len(runs))):
        x = np.array([max(int(r["step"]), 1) for r in rows])
        y = np.array([m["value"](r) for r in rows])
        if log:
            y = np.where(y > 0, y, np.nan)  # a log axis can't show values <= 0
        ax.plot(x, y, color=color, lw=1.5, marker="o", ms=4, label=f"n_train = {n}", zorder=3)
        se = np.array([m["se"](r) for r in rows]) if m["se"] else np.zeros_like(y)
        lower = np.maximum(y - se, 0.5 * np.nanmin(y)) if log else y - se
        if m["se"]:
            ax.fill_between(x, lower, y + se, color=color, alpha=0.15, lw=0, zorder=2)
        lo.append(np.nanmin(lower))
        hi.append(np.nanmax(y + se))
        if not math.isnan(ymin) and np.any(y < ymin):
            low = y < ymin
            ax.plot(x[low], np.full(low.sum(), ymin), ls="none", marker="v", ms=6, color=color,
                    clip_on=False, zorder=4)
            clipped = True
        if args.metric == "loglik_gap":  # this run's memorizing distribution, in the run's colour
            ax.axhline(float(rows[0]["loglik_true"]) - float(rows[0]["loglik_mem"]), color=color,
                       ls="-.", lw=1, zorder=1)

    for value, ref_label, color in m["refs"]:
        ax.axhline(value, color=color, ls=":", lw=1, label=ref_label, zorder=1)
    if args.metric == "loglik":
        true = ll_true.pop()
        ax.axhline(true, color="k", ls=":", lw=1, label="true distribution", zorder=1)
        top, bottom = max(max(hi), true), min(lo)
        pad = 0.04 * (top - bottom)
        ax.set_ylim(bottom - pad, top + pad)
        mems = [(n, float(rows[0]["loglik_mem"])) for n, _, rows, _ in runs]
        ax.plot([], [], ls="none", label="memorizing distributions (off scale):\n"
                + ", ".join(f"{v:.0f} (n = {n})" for n, v in (mems[0], mems[-1])).replace(", ", " to "))
    elif args.metric == "loglik_gap":
        ax.plot([], [], color=MUTED, ls="-.", lw=1, label="memorizing distribution\n(one per n_train, same colour)")
        ax.set_yscale("log")
    elif not math.isnan(ymin):
        ax.set_ylim(ymin, max(hi) + 0.04 * (max(hi) - ymin))
    if clipped:  # legend entry for the off-scale markers
        ax.plot([], [], ls="none", marker="v", ms=6, color=MUTED, label=f"below {ymin:g} (off scale)")

    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(m["label"])
    ax.grid(True, alpha=0.3, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    if args.metric == "rho_kl":
        detail = f"log grid with {s0['num_steps'] + 1} times; {s0['n_paths']:,} model paths per KL"
    else:
        n_val = len(np.load(os.path.join(runs[0][3], "data.npz"))["x_test"])
        detail = (f"{n_val:,} validation paths, log grid t = {s0['t_min']:g} to 1 "
                  f"({s0['num_steps'] + 1} times)")
    shaded = "; shaded: $\\pm$ 1 s.e." if m["se"] else ""
    weights = "EMA" if args.weights == "ema" else "raw"
    ax.set_title(f"{m['label']} through training{' (zoomed)' if args.metric == 'loglik' else ''}, all training-set sizes (MLP score model, {weights} weights)\n"
                 f"{detail}{shaded}", fontsize=9, color="#333333")
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.metric}_{args.weights}.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
