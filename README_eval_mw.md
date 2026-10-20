# Marion & Wu (2026)-style evaluation

`eval_mw.py` evaluates every saved checkpoint of a run and `plot_mw.py` draws the four panels of
M&W's Fig. 1b (loss, memorization score, sliced Wasserstein, FID; train and test).

```bash
# after (or during) training with configs/vp/ddpm/cifar10_mw_n2048.py
python eval_mw.py --config configs/vp/ddpm/cifar10_mw_n2048.py --workdir runs/mw_n2048
python plot_mw.py --workdir runs/mw_n2048
```

Use the same `--config` and `--config.*` overrides as for training: the script rebuilds the model
from the config and refuses checkpoints that don't match it. Output goes to `<workdir>/eval_mw/`:
`results.csv`, `fig1b_<weights>.png`, sample grids in `samples/`, cached reference features, and
`config_used.txt`. Rerunning skips checkpoints already in `results.csv`.

## How each quantity follows M&W

| Quantity | M&W (Sec. 3.1) | Here |
|---|---|---|
| Training set | first N CIFAR-10 training images | `train[:N]`, N = `data.num_train` (TF Datasets order) |
| Test set | fixed 5,120 test images | `test[:data.num_test]` (5,120 in the M&W config) |
| Loss | score-matching loss; one batch for train, full set for test | the training objective on **all** N train images and all test images, with the same noise and times at every checkpoint (less noisy than one batch) |
| Memorization score | fraction of samples with max cosine similarity to the training data > 0.6 in SSCD space | same; SSCD model `sscd_disc_mixup`, images resized to 288 x 288 (M&W don't name the SSCD model or preprocessing). Also reports a pixel-space k-NN score as a cross-check |
| Sliced Wasserstein | 2,000 samples vs each reference set, Inception features | same; POT's `ot.sliced_wasserstein_distance`, p = 2, 1,000 fixed unit directions on Inception `pool_3` (M&W don't give p, the number of directions or the layer, so absolute values won't match theirs) |
| FID | samples vs each reference set, as many samples as reference images | same; TF-GAN Inception (as used for FID elsewhere in this repo), formula in `tfgan_metrics.py` |
| Sampler | rectified flow, 250 steps, guidance weight 0 | probability-flow ODE of the VP SDE, 250 Euler steps, plus a final denoising step (`--sampler=sde` for the repo's Euler-Maruyama sampler) |
| Weights | not stated (EMA 0.9999 in their table) | EMA by default; `--weights=raw` or `both` |

Not matched: M&W's models are class-conditional and trained with rectified flow; this repo trains an
unconditional VP SDE. One set of samples per checkpoint is used for every metric, and the starting
noise is the same for all checkpoints.

## Cost

Generating samples dominates: per checkpoint, max(N, 5,120) samples x 250 steps. For the ~80M-parameter
M&W network that is roughly 29 PFLOP per checkpoint and weight set: minutes on a datacentre GPU,
hours on a laptop. On a laptop, test with small settings, e.g.
`--ckpts=1 --sample_steps=10 --n_fid_train=256 --n_fid_test=256 --n_sw=256 --batch=128`.

First run downloads: the Inception network (TF-Hub; set `TFHUB_CACHE_DIR` to keep it) and the SSCD
model (to `~/.cache/sscd`). On a cluster without internet on compute nodes, run once on a login node.
