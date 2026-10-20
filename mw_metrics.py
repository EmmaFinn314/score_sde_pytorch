"""NumPy helpers for the Marion & Wu-style evaluation (eval_mw.py). No PyTorch / TensorFlow needed.

  memorization_fraction_cosine  M&W's memorization score: fraction of generated samples whose
                                maximum cosine similarity to the training set exceeds 0.6
                                (computed on SSCD embeddings in eval_mw.py).
  memorization_fraction_knn     pixel-space cross-check used by Yoon et al., Gu et al. and Bonnaire
                                et al.: fraction of samples whose nearest training image is much
                                closer than the second nearest (d1 / d2 < 1/3).
"""
import numpy as np


def l2_normalize(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def max_cosine_similarity(gen_emb, train_emb, chunk=2048):
    """For each generated embedding, the largest cosine similarity to any training embedding."""
    g, t = l2_normalize(gen_emb), l2_normalize(train_emb)
    out = np.empty(len(g))
    for i in range(0, len(g), chunk):
        out[i:i + chunk] = (g[i:i + chunk] @ t.T).max(axis=1)
    return out


def memorization_fraction_cosine(gen_emb, train_emb, threshold=0.6):
    sims = max_cosine_similarity(gen_emb, train_emb)
    return float(np.mean(sims > threshold)), sims


def nearest_two_distances(gen, train, chunk=1024):
    """Euclidean distances from each generated point to its two nearest training points."""
    gen = np.asarray(gen, dtype=np.float64).reshape(len(gen), -1)
    train = np.asarray(train, dtype=np.float64).reshape(len(train), -1)
    t2 = (train ** 2).sum(axis=1)
    out = np.empty((len(gen), 2))
    for i in range(0, len(gen), chunk):
        g = gen[i:i + chunk]
        d2 = (g ** 2).sum(axis=1)[:, None] + t2[None, :] - 2 * g @ train.T
        two = np.partition(np.maximum(d2, 0.0), 1, axis=1)[:, :2]
        out[i:i + chunk] = np.sqrt(np.sort(two, axis=1))
    return out


def memorization_fraction_knn(gen, train, threshold=1 / 3):
    d = nearest_two_distances(gen, train)
    ratio = d[:, 0] / np.maximum(d[:, 1], 1e-12)
    return float(np.mean(ratio < threshold)), ratio


def check_sliced_wasserstein(sw_fn, seed=0):
    """Sanity-check a sliced Wasserstein function sw_fn(a, b) using unit directions.

    For a pure shift by m, SW_2 = sqrt(E[(theta . m)^2]) = |m| / sqrt(d) for theta uniform on the sphere.
    Raises if the function doesn't behave like that (e.g. a different POT version or API).
    """
    rng = np.random.default_rng(seed)
    d = 64
    a = rng.standard_normal((500, d))
    m = np.full(d, 0.5)
    same = sw_fn(a, a)
    shifted = sw_fn(a, a + m)
    expected = np.linalg.norm(m) / np.sqrt(d)
    if abs(same) > 1e-8 or abs(shifted - expected) > 0.1 * expected:
        raise RuntimeError(f"sliced Wasserstein sanity check failed: SW(a, a) = {same}, "
                           f"SW(a, a + shift) = {shifted} (expected about {expected:.3f})")
    return same, shifted, expected
