"""Random GMM ground truth for the synthetic fidelity experiments. Parameters are drawn once, standardized analytically (not from samples), and then
held fixed across every n, seed and model in a sweep.

    data = make_datasets(param_seed=0, data_seed=0)   # 1,000 test points, n_train = 16 ... 8192
    x_train, x_test = data["x_train_512"], data["x_test"]
    np.savez("gmm_data.npz", **data)                   # to use exactly these arrays elsewhere
"""
import numpy as np


def haar_orthogonal(d, rng):
    """Uniformly random (Haar) orthogonal matrix."""
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))  # sign fix makes Q exactly Haar-distributed


def sample_gmm_params(d=5, K=10, radius=2.0, scale=0.25, cond=4.0, kappa=2.0,
                      alpha=None, seed=0, max_tries=10_000):
    """Draw (pi, mu, Sigma) for a K-component GMM in R^d.

    radius : means are uniform on a sphere of this radius (before standardization)
    scale  : typical component std (geometric mean of the eigenvalue range)
    cond   : max condition number of each Sigma_k (eigenvalues log-uniform in range)
    kappa  : require ||mu_j - mu_k|| >= kappa * (sigma_max_j + sigma_max_k)
    alpha  : None -> equal weights; else Dirichlet(alpha * 1_K)
    """
    rng = np.random.default_rng(seed)
    pi = np.full(K, 1.0 / K) if alpha is None else rng.dirichlet(np.full(K, alpha))
    iu = np.triu_indices(K, 1)

    for _ in range(max_tries):
        # Covariances: random rotation, eigenvalues log-uniform in
        # [scale^2 / sqrt(cond), scale^2 * sqrt(cond)].
        lam = scale**2 * np.exp(rng.uniform(-0.5, 0.5, (K, d)) * np.log(cond))
        Q = np.stack([haar_orthogonal(d, rng) for _ in range(K)])
        Sigma = Q @ (lam[:, :, None] * Q.transpose(0, 2, 1))  # Q diag(lam) Q^T

        # Means: uniform on the sphere, rejected unless all pairs are separated.
        z = rng.standard_normal((K, d))
        mu = radius * z / np.linalg.norm(z, axis=1, keepdims=True)
        smax = np.sqrt(lam.max(axis=1))
        D = np.linalg.norm(mu[:, None] - mu[None], axis=-1)
        if np.all(D[iu] >= kappa * (smax[:, None] + smax[None])[iu]):
            break
    else:
        raise RuntimeError("No draw met the separation constraint: lower kappa or raise radius.")

    # Exact standardization of the distribution: centre, and scale so the
    # average per-coordinate variance is 1. Separation ratios are unchanged.
    m = pi @ mu
    C = (np.einsum("k,kij->ij", pi, Sigma)
         + np.einsum("k,ki,kj->ij", pi, mu, mu) - np.outer(m, m))
    c = np.sqrt(np.trace(C) / d)
    return pi, (mu - m) / c, Sigma / c**2


def sample_gmm(pi, mu, Sigma, n, seed):
    """Draw n points; returns (x, component labels).

    Labels and noise come from separate streams, so for a fixed seed the first m
    points are the same whatever n >= m is: smaller draws are prefixes of larger ones.
    """
    label_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
    k = np.random.default_rng(label_seed).choice(len(pi), size=n, p=pi)
    L = np.linalg.cholesky(Sigma)
    eps = np.random.default_rng(noise_seed).standard_normal((n, mu.shape[1]))
    x = mu[k] + np.einsum("nij,nj->ni", L[k], eps)
    return x, k


TRAIN_SIZES = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def make_datasets(param_seed=0, data_seed=0, n_test=1000, train_sizes=TRAIN_SIZES,
                  **param_kwargs):
    """GMM parameters, one test set and nested training sets of every size.

    param_seed fixes (pi, mu, Sigma); data_seed fixes the data drawn from them. The
    test set and the training pool use independent streams of data_seed, so the test
    set doesn't depend on train_sizes. The training set of size n is the first n
    points of one pool of max(train_sizes), so each set contains every smaller one.
    Returns a dict of arrays (x_train_<n>, k_train_<n> for each n), ready for np.savez.
    """
    pi, mu, Sigma = sample_gmm_params(seed=param_seed, **param_kwargs)
    x_test, k_test = sample_gmm(pi, mu, Sigma, n_test, seed=[data_seed, 0])
    x_pool, k_pool = sample_gmm(pi, mu, Sigma, max(train_sizes), seed=[data_seed, 1])
    data = dict(pi=pi, mu=mu, Sigma=Sigma, x_test=x_test, k_test=k_test,
                param_seed=param_seed, data_seed=data_seed,
                train_sizes=np.array(sorted(train_sizes)))
    for n in sorted(train_sizes):
        data[f"x_train_{n}"], data[f"k_train_{n}"] = x_pool[:n], k_pool[:n]
    return data


if __name__ == "__main__":
    data = make_datasets(param_seed=0, data_seed=0)
    print("test set            :", data["x_test"].shape)
    print("training sets       :", [data[f"x_train_{n}"].shape for n in data["train_sizes"]])

    pi, mu, Sigma = data["pi"], data["mu"], data["Sigma"]
    x, _ = sample_gmm(pi, mu, Sigma, 200_000, seed=1)
    ev = np.linalg.eigvalsh(Sigma)
    print("empirical mean      :", np.round(x.mean(0), 3))
    print("avg coord variance  :", round(x.var(0).mean(), 3))
    print("component std range :", np.round(np.sqrt([ev.min(), ev.max()]), 3))
    D = np.linalg.norm(mu[:, None] - mu[None], axis=-1)
    print("min mean separation :", round(D[np.triu_indices(len(pi), 1)].min(), 3))
