"""Plot eval_mw.py's results as the 2x2 panel of Marion & Wu's Fig. 1b.

    python plot_mw.py --workdir runs/mw_n2048

Writes <workdir>/eval_mw/fig1b_<weights>.png (one figure per weight set): Loss, Memorization Score,
Sliced Wasserstein and FID against training step on a log axis; train solid, test dashed. The
minibatch training loss from <workdir>/stdout.txt is drawn faintly on the loss panel when present.
"""
import argparse
import csv
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOR = "#4C72B0"


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def training_log(workdir):
    """(steps, losses) from the 'step: N, training_loss: x' lines main.py writes to stdout.txt."""
    path = os.path.join(workdir, "stdout.txt")
    steps, losses = [], []
    if os.path.exists(path):
        pattern = re.compile(r"step: (\d+), training_loss: ([0-9.eE+-]+)")
        with open(path) as f:
            for line in f:
                m = pattern.search(line)
                if m and int(m.group(1)) > 0:
                    steps.append(int(m.group(1)))
                    losses.append(float(m.group(2)))
    return steps, losses


def plot(rows, log_steps, log_losses, title, out_path):
    rows = sorted(rows, key=lambda r: int(r["step"]))
    x = [max(int(r["step"]), 1) for r in rows]
    col = {k: [as_float(r[k]) for r in rows] for k in rows[0]}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    (ax_loss, ax_mem), (ax_sw, ax_fid) = axes

    if log_steps:
        ax_loss.plot(log_steps, log_losses, color=COLOR, alpha=0.2, lw=0.8, label="train (minibatch)")
    ax_loss.plot(x, col["loss_train"], color=COLOR, marker="o", ms=3, lw=1.5, label="train")
    ax_loss.plot(x, col["loss_test"], color=COLOR, marker="o", ms=3, lw=1.5, ls="--", label="test")
    ax_loss.set_title("Loss")

    if not all(math.isnan(v) for v in col["mem_sscd"]):
        ax_mem.plot(x, col["mem_sscd"], color=COLOR, marker="o", ms=3, lw=1.5, label="SSCD cos > 0.6")
    ax_mem.plot(x, col["mem_knn"], color="#C44E52", marker="o", ms=3, lw=1.2, ls=":",
                label="pixel k-NN d1/d2 < 1/3")
    ax_mem.set_ylim(-0.02, 1.02)
    ax_mem.set_title("Memorization Score")

    for ax, key, name in [(ax_sw, "sw", "Sliced Wasserstein"), (ax_fid, "fid", "FID")]:
        ax.plot(x, col[f"{key}_train"], color=COLOR, marker="o", ms=3, lw=1.5, label="train")
        ax.plot(x, col[f"{key}_test"], color=COLOR, marker="o", ms=3, lw=1.5, ls="--", label="test")
        ax.set_title(name)

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", required=True)
    p.add_argument("--results", default="results.csv", help="file name inside <workdir>/eval_mw")
    args = p.parse_args()

    with open(os.path.join(args.workdir, "eval_mw", args.results)) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("No results yet.")
    log_steps, log_losses = training_log(args.workdir)
    name = os.path.basename(os.path.normpath(args.workdir))
    for which in sorted({r["weights"] for r in rows}):
        subset = [r for r in rows if r["weights"] == which]
        r0 = subset[0]
        title = (f"{name}: {which} weights, N = {r0['n_train_ref']}, {r0['sampler']} sampler, "
                 f"{r0['sample_steps']} steps")
        plot(subset, log_steps, log_losses, title,
             os.path.join(args.workdir, "eval_mw", f"fig1b_{which}.png"))


if __name__ == "__main__":
    main()
