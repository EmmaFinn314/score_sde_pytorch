"""NumPy versions of the TF-GAN metrics used by run_lib.evaluate(): Inception Score, FID and KID.

Why this file exists: TF-GAN (tensorflow_gan) cannot be imported on TensorFlow >= 2.16, because
importing it loads its Estimator module and TensorFlow 2.16 removed tf.estimator. TensorFlow 2.16
is also the first release with Python 3.12 builds, so on Python 3.12 there is no TensorFlow that
TF-GAN works with.

These functions reproduce TF-GAN's definitions (tensorflow_gan/python/eval/classifier_metrics.py,
Apache License 2.0), so numbers stay comparable with results computed by the original code:

  classifier_score_from_logits                  Inception Score: exp(E_x KL(p(y|x) || p(y))),
                                                over all samples in one split.
  frechet_classifier_distance_from_activations  FID with unbiased (N-1) covariances and TF-GAN's
                                                SVD-based symmetric matrix square root.
  kernel_classifier_distance_from_activations   KID: unbiased MMD^2 with kernel (x.y/d + 1)^3,
                                                averaged over blocks of at most 1024 samples.

Everything is computed in float64 (TF-GAN does the same for IS and FID; for KID the original call
in run_lib used float32, so KID may differ from old numbers in the last few significant digits).
"""
import numpy as np


def _symmetric_matrix_square_root(mat, eps=1e-10):
    """Square root of a symmetric PSD matrix via SVD; singular values below eps are not rooted."""
    u, s, vt = np.linalg.svd(mat)
    si = np.where(s < eps, s, np.sqrt(s))
    return (u * si) @ vt


def trace_sqrt_product(sigma, sigma_v):
    """trace(sqrt(sigma @ sigma_v)) computed as trace(sqrt(A sigma_v A)) with A = sqrt(sigma)."""
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = sqrt_sigma @ (sigma_v @ sqrt_sigma)
    return np.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def classifier_score_from_logits(logits):
    """Inception Score from classifier logits of shape [N, num_classes]."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D, got shape {logits.shape}")
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))   # log_softmax
    p = np.exp(log_p)
    q = p.mean(axis=0)
    kl = np.sum(p * (log_p - np.log(q)), axis=1)
    return float(np.exp(kl.mean()))


def frechet_classifier_distance_from_activations(activations1, activations2):
    """FID between two sets of activations of shape [N, d] (the sets may differ in size)."""
    a1 = np.asarray(activations1, dtype=np.float64)
    a2 = np.asarray(activations2, dtype=np.float64)
    if a1.ndim != 2 or a2.ndim != 2 or a1.shape[1] != a2.shape[1]:
        raise ValueError(f"activations must be [N, d] with equal d, got {a1.shape} and {a2.shape}")
    m, m_w = a1.mean(axis=0), a2.mean(axis=0)
    sigma = np.cov(a1, rowvar=False)          # unbiased, i.e. N/(N-1) x tfp.stats.covariance
    sigma_w = np.cov(a2, rowvar=False)
    trace = np.trace(sigma + sigma_w) - 2.0 * trace_sqrt_product(sigma, sigma_w)
    mean = np.sum((m - m_w) ** 2)
    return float(trace + mean)


def _block_bounds(n, n_blocks):
    """Split n items into n_blocks contiguous blocks whose sizes differ by at most one (TF-GAN's rule)."""
    v = n // n_blocks
    n_plusone = n - v * n_blocks
    sizes = [v] * (n_blocks - n_plusone) + [v + 1] * n_plusone
    return np.concatenate([[0], np.cumsum(sizes)]).astype(int)


def kernel_classifier_distance_and_std_from_activations(activations1, activations2,
                                                        max_block_size=1024):
    """KID and an estimate of its standard error, as in TF-GAN.

    Assumes both sets are in random order (blocks are taken contiguously).
    """
    r = np.asarray(activations1, dtype=np.float64)
    g = np.asarray(activations2, dtype=np.float64)
    if r.ndim != 2 or g.ndim != 2 or r.shape[1] != g.shape[1]:
        raise ValueError(f"activations must be [N, d] with equal d, got {r.shape} and {g.shape}")
    n_blocks = int(np.ceil(max(len(r), len(g)) / max_block_size))
    inds_r, inds_g = _block_bounds(len(r), n_blocks), _block_bounds(len(g), n_blocks)
    dim = r.shape[1]
    ests = np.empty(n_blocks)
    for i in range(n_blocks):
        rb = r[inds_r[i]:inds_r[i + 1]]
        gb = g[inds_g[i]:inds_g[i + 1]]
        m, n = len(rb), len(gb)
        k_rr = (rb @ rb.T / dim + 1) ** 3
        k_rg = (rb @ gb.T / dim + 1) ** 3
        k_gg = (gb @ gb.T / dim + 1) ** 3
        ests[i] = (-2 * k_rg.mean()
                   + (k_rr.sum() - np.trace(k_rr)) / (m * (m - 1))
                   + (k_gg.sum() - np.trace(k_gg)) / (n * (n - 1)))
    mn = ests.mean()
    var = np.nan if n_blocks <= 1 else np.sum((ests - mn) ** 2) / (n_blocks - 1)
    return float(mn), float(np.sqrt(var / n_blocks))


def kernel_classifier_distance_from_activations(activations1, activations2, max_block_size=1024):
    """KID between two sets of activations of shape [N, d]."""
    return kernel_classifier_distance_and_std_from_activations(
        activations1, activations2, max_block_size)[0]
