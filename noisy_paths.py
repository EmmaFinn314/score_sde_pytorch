"""Forward VP-SDE noising along a time grid, for the path-likelihood experiments.

The SDE is sde_lib.VPSDE: dx = -beta(t) x / 2 dt + sqrt(beta(t)) dW, with
beta(t) = beta_min + t (beta_max - beta_min) on [0, 1]. Its transitions are exact Gaussians,

    x_t | x_s ~ N(a x_s, (1 - a^2) I),   a = alpha(t) / alpha(s),
    log alpha(t) = -t^2 (beta_max - beta_min) / 4 - t beta_min / 2      (VPSDE.marginal_prob),

so a path can be sampled at the grid times directly, with no discretization error.

Path log-likelihoods. For observations O_1:R at times t_1 < ... < t_R the forward process is
Markov with transitions that don't depend on the data distribution, so the exact joint density is

    log p(O_1:R) = log p_{t_1}(O_1) + sum_{r=2..R} log N(O_r; a_r O_{r-1}, (1 - a_r^2) I),

where p_{t_1} is the data distribution pushed to time t_1. For the true GMM that is
sum_k pi_k N(alpha mu_k, alpha^2 Sigma_k + (1 - alpha^2) I); for the memorizing distribution
(the empirical distribution of the n training points) it is (1/n) sum_i N(alpha x_i, (1 - alpha^2) I).

    from gmm_setup import make_datasets
    data = make_datasets(param_seed=0, data_seed=0)
    ts = time_grid(50)                             # t_1 = 1e-3, ..., t_R = T = 1
    # or log-spaced: ts = log_time_grid(50)        # 51 times, 1e-3 to T, constant ratio
    xs = noise_paths(data["x_test"], ts, seed=0)   # [1000, 50, 5]: xs[i, j] is test point i at ts[j]
    ll_true = path_loglik_gmm(xs, ts, data["pi"], data["mu"], data["Sigma"])
    ll_mem = path_loglik_memorizing(xs, ts, data["x_train_512"])
"""
import numpy as np
from scipy.special import logsumexp

BETA_MIN, BETA_MAX = 0.1, 20.0  # sde_lib.VPSDE defaults, as used by the trained configs
T = 1.0                         # sde_lib.VPSDE.T


def log_alpha(t, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """log alpha(t), the VP mean coefficient: x_t | x_0 ~ N(alpha(t) x_0, (1 - alpha(t)^2) I)."""
    t = np.asarray(t, dtype=float)
    return -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min


def time_grid(num_times=50, t_min=1e-3, t_max=T):
    """num_times evenly spaced times from t_min to t_max = T, both included.

    t_min = 1e-3 is the repo's sampling_eps for VP; it must be > 0, since the memorizing
    distribution has no density at t = 0.
    """
    ts = np.linspace(t_min, t_max, num_times)
    ts[-1] = t_max  # exactly T, so the last observation is at the end of the forward process
    return ts


def log_time_grid(num_steps=50, t_min=1e-3, t_max=T):
    """Log-spaced times t_m = t_min (t_max / t_min)^(m / M) for m = 0, ..., M, with M = num_steps.

    M + 1 points (M steps), from t_min to t_max = T, both included; the ratio between neighbouring
    times is constant, so small times, where the memorizing density changes fastest, get more points.
    """
    m = np.arange(num_steps + 1)
    ts = t_min * (t_max / t_min) ** (m / num_steps)
    ts[0], ts[-1] = t_min, t_max  # exact endpoints despite rounding in the power
    return ts


def _steps(ts, beta_min, beta_max):
    """(a, 1 - a^2) for each step t_{j-1} -> t_j of ts; 1 - a^2 via expm1, accurate for small steps."""
    dla = np.diff(log_alpha(ts, beta_min, beta_max))
    return np.exp(dla), -np.expm1(2 * dla)


def noise_paths(x0, ts, seed, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Run the forward SDE from x0 [n, d] through the increasing times ts [M]; returns [n, M, d].

    Each path is one Markov chain, x_0 -> x_{t_1} -> ... -> x_{t_M}, drawn with the exact
    transitions, so the paths have the SDE's joint law and not just its marginal at each time.
    The noise for point i depends only on (seed, i), so a larger x0 extends the paths of a smaller one.
    """
    x0 = np.asarray(x0, dtype=float)
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 1 or ts[0] < 0 or np.any(np.diff(ts) <= 0):
        raise ValueError("ts must be strictly increasing and non-negative.")
    a, var = _steps(np.concatenate([[0.0], ts]), beta_min, beta_max)  # first step from t_0 = 0
    s = np.sqrt(var)
    eps = np.random.default_rng(seed).standard_normal((len(x0), len(ts), x0.shape[1]))
    xs = np.empty_like(eps)
    x = x0
    for j in range(len(ts)):
        x = a[j] * x + s[j] * eps[:, j]
        xs[:, j] = x
    return xs


# --------------------------------------------------------------------------- path log-likelihoods
def transition_loglik(xs, ts, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """sum_{r=2..R} log N(O_r; a_r O_{r-1}, (1 - a_r^2) I) for paths xs [n, R, d]; returns [n].

    The same for every data distribution, so it is shared by the true and memorizing likelihoods.
    """
    a, var = _steps(np.asarray(ts, dtype=float), beta_min, beta_max)
    resid = xs[:, 1:] - a[None, :, None] * xs[:, :-1]
    d = xs.shape[2]
    return (-0.5 * np.sum(resid**2 / var[None, :, None], axis=(1, 2))
            - 0.5 * d * np.sum(np.log(2 * np.pi * var)))


def gmm_marginal_logpdf(x, t, pi, mu, Sigma, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """log p_t(x) for x [n, d] when x_0 ~ GMM(pi, mu, Sigma): each component becomes
    N(alpha mu_k, alpha^2 Sigma_k + (1 - alpha^2) I)."""
    la = log_alpha(t, beta_min, beta_max)
    alpha, d = np.exp(la), mu.shape[1]
    cov = alpha**2 * Sigma + (-np.expm1(2 * la)) * np.eye(d)
    L = np.linalg.cholesky(cov)                                   # [K, d, d]
    diff = x[:, None, :] - alpha * mu[None]                       # [n, K, d]
    z = np.linalg.solve(L[None], diff[..., None])[..., 0]         # L_k^{-1} (x - alpha mu_k)
    logdet = 2 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)  # [K]
    comp = -0.5 * (np.sum(z**2, -1) + logdet + d * np.log(2 * np.pi))
    return logsumexp(comp + np.log(pi), axis=1)


def memorizing_marginal_logpdf(x, t, x_train, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """log p_t(x) for x [n, d] when x_0 is uniform on the training points x_train [m, d]:
    (1/m) sum_i N(x; alpha x_i, (1 - alpha^2) I)."""
    la = log_alpha(t, beta_min, beta_max)
    alpha, var, d = np.exp(la), -np.expm1(2 * la), x.shape[1]
    y = alpha * x_train
    d2 = np.sum(x**2, 1)[:, None] - 2 * x @ y.T + np.sum(y**2, 1)[None]
    comp = -0.5 * np.maximum(d2, 0) / var - 0.5 * d * np.log(2 * np.pi * var)
    return logsumexp(comp, axis=1) - np.log(len(x_train))


def path_loglik_gmm(xs, ts, pi, mu, Sigma, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Exact log p(O_1:R) of paths xs [n, R, d] at times ts under the true GMM; returns [n]."""
    return (gmm_marginal_logpdf(xs[:, 0], ts[0], pi, mu, Sigma, beta_min, beta_max)
            + transition_loglik(xs, ts, beta_min, beta_max))


def path_loglik_memorizing(xs, ts, x_train, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Exact log p(O_1:R) of paths xs [n, R, d] at times ts under the memorizing distribution
    (uniform on x_train); returns [n]."""
    if ts[0] <= 0:
        raise ValueError("The memorizing distribution has no density at t = 0: start the grid at t > 0.")
    return (memorizing_marginal_logpdf(xs[:, 0], ts[0], x_train, beta_min, beta_max)
            + transition_loglik(xs, ts, beta_min, beta_max))


# --------------------------------------------------------------------------- score-based model on the grid
def gmm_score(x, t, pi, mu, Sigma, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """True score grad_x log p_t(x) for x [n, d] when x_0 ~ GMM(pi, mu, Sigma); returns [n, d].

    sum_k r_k(x) C_k^{-1} (alpha mu_k - x), with C_k = alpha^2 Sigma_k + (1 - alpha^2) I and r_k
    the posterior component probabilities at time t.
    """
    la = log_alpha(t, beta_min, beta_max)
    alpha, d = np.exp(la), mu.shape[1]
    cov = alpha**2 * Sigma + (-np.expm1(2 * la)) * np.eye(d)
    L = np.linalg.cholesky(cov)
    diff = x[:, None, :] - alpha * mu[None]                                  # [n, K, d]
    z = np.linalg.solve(L[None], diff[..., None])[..., 0]
    logdet = 2 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)
    logw = np.log(pi) - 0.5 * (np.sum(z**2, -1) + logdet)                    # [n, K]
    r = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))
    grad = -np.linalg.solve(np.swapaxes(L, 1, 2)[None], z[..., None])[..., 0]  # -C_k^{-1} (x - alpha mu_k)
    return np.einsum("nk,nkd->nd", r, grad)


def memorizing_score(x, t, x_train, beta_min=BETA_MIN, beta_max=BETA_MAX, chunk=1024):
    """Score grad_x log p_t(x) for x [n, d] when x_0 is uniform on x_train [m, d]; returns [n, d].

    (alpha sum_i r_i(x) x_i - x) / (1 - alpha^2), with r_i the softmax weights of the training
    points. Done in chunks of x so memory stays at chunk x m.
    """
    la = log_alpha(t, beta_min, beta_max)
    alpha, var = np.exp(la), -np.expm1(2 * la)
    y = alpha * x_train
    y2 = np.sum(y**2, 1)
    out = np.empty_like(x, dtype=float)
    for i in range(0, len(x), chunk):
        xc = x[i:i + chunk]
        logw = (xc @ y.T - 0.5 * y2[None]) / var  # -||x - y||^2 / (2 var) up to a per-row constant
        r = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))
        out[i:i + chunk] = (r @ y - xc) / var
    return out


def _reverse_steps(ts, beta_min, beta_max):
    """(1/a, (1 - a^2)/a, variance) of the reverse Gaussian step t_j -> t_{j-1} for j = 1..R-1.

    The step is the DDPM posterior q(x_s | x_t, x0_hat) with x0_hat the Tweedie estimate
    (x_t + sigma_t^2 score) / alpha_t, which simplifies to
        x_s ~ N(x_t / a + (1 - a^2) / a * score(x_t, t),  sigma_s^2 (1 - a^2) / sigma_t^2 I),
    a = alpha_t / alpha_s. Unlike an Euler-Maruyama step it stays well behaved for the large
    steps near T on a log grid, and as the steps shrink both give the same reverse SDE.
    """
    ts = np.asarray(ts, dtype=float)
    la = log_alpha(ts, beta_min, beta_max)
    a, one_m_a2 = _steps(ts, beta_min, beta_max)
    var_s, var_t = -np.expm1(2 * la[:-1]), -np.expm1(2 * la[1:])
    return 1 / a, one_m_a2 / a, var_s * one_m_a2 / var_t


def path_loglik_model(xs, ts, score_fn, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """log p_theta(O_1:R) of paths xs [n, R, d] under the reverse model on the grid; returns [n].

    p_theta starts from the prior N(0, I) at t_R and takes the reverse Gaussian steps of
    _reverse_steps with score_fn(x [n, d], t) -> [n, d] (e.g. a wrapped score network).
    """
    inv_a, coef, var = _reverse_steps(ts, beta_min, beta_max)
    d = xs.shape[2]
    ll = -0.5 * np.sum(xs[:, -1]**2, 1) - 0.5 * d * np.log(2 * np.pi)
    for j in range(len(ts) - 1, 0, -1):
        mean = inv_a[j - 1] * xs[:, j] + coef[j - 1] * score_fn(xs[:, j], ts[j])
        ll += (-0.5 * np.sum((xs[:, j - 1] - mean)**2, 1) / var[j - 1]
               - 0.5 * d * np.log(2 * np.pi * var[j - 1]))
    return ll


def reverse_kl(score_fn, ts, pi, mu, Sigma, n_paths=10_000, seed=0, d=None,
               beta_min=BETA_MIN, beta_max=BETA_MAX):
    """KL(P_theta || P_true) between reverse path laws on the grid ts, as (estimate, standard error).

    Both start from N(0, I) at t_R and take the same reverse Gaussian step, P_theta with score_fn
    and P_true with the true GMM score. By the chain rule the KL is the expectation under P_theta
    of a sum of per-step KLs between two Gaussians with equal covariance, each in closed form:
        ((1 - a^2) / a)^2 ||score_fn - gmm_score||^2 / (2 var).
    The expectation over P_theta's paths is by Monte Carlo, drawing n_paths paths.
    """
    kl = _reverse_kl_per_path(score_fn, ts, pi, mu, Sigma, n_paths, seed, d, beta_min, beta_max)
    return float(kl.mean()), float(kl.std(ddof=1) / np.sqrt(n_paths))


def memorizing_kl(ts, pi, mu, Sigma, x_train, n_paths=10_000, seed=0,
                  beta_min=BETA_MIN, beta_max=BETA_MAX):
    """KL(P_mem || P_true) per path, computed once per training set and passed to rho_kl as kl_mem.

    It doesn't depend on the model, so one computation serves every checkpoint trained on x_train.
    Returns a dict of arrays (save with np.savez(path, **out), reload with dict(np.load(path))):
    kl [n_paths] and the settings it must be paired with (ts, seed, n_paths, schedule, GMM).
    """
    kl = _reverse_kl_per_path(lambda x, t: memorizing_score(x, t, x_train, beta_min, beta_max),
                              ts, pi, mu, Sigma, n_paths, seed, None, beta_min, beta_max)
    return dict(kl=kl, ts=np.asarray(ts, dtype=float), n_paths=n_paths, seed=seed,
                n_train=len(x_train), beta_min=beta_min, beta_max=beta_max, pi=pi, mu=mu, Sigma=Sigma)


def rho_kl(score_fn, ts, pi, mu, Sigma, x_train=None, n_paths=10_000, seed=0, kl_mem=None,
           beta_min=BETA_MIN, beta_max=BETA_MAX):
    """rho_KL = 1 - KL(P_theta || P_true) / KL(P_mem || P_true), with KLs as in reverse_kl.

    P_mem is the reverse chain with the memorizing score of x_train (the training set the model
    was fit to). rho_KL = 1 when P_theta has the true score, 0 when it is as far from P_true as
    P_mem, negative when further. Both KLs use the same noise draws (common random numbers), and the
    standard error is the delta-method one for that paired ratio.

    kl_mem: the output of memorizing_kl (or a reloaded .npz of it) to skip computing KL(P_mem || P_true).
    x_train is then not needed, and n_paths and seed are taken from kl_mem so the draws stay paired.
    Returns dict(rho_kl, se, kl_theta, kl_theta_se, kl_mem, kl_mem_se).
    """
    if kl_mem is None:
        if x_train is None:
            raise ValueError("Pass x_train, or a precomputed kl_mem from memorizing_kl.")
        kl_mem = memorizing_kl(ts, pi, mu, Sigma, x_train, n_paths, seed, beta_min, beta_max)
    else:
        same = (np.array_equal(np.asarray(kl_mem["ts"]), np.asarray(ts, dtype=float))
                and float(kl_mem["beta_min"]) == beta_min and float(kl_mem["beta_max"]) == beta_max
                and all(np.array_equal(np.asarray(kl_mem[k]), v)
                        for k, v in [("pi", pi), ("mu", mu), ("Sigma", Sigma)]))
        if not same:
            raise ValueError("kl_mem was computed with a different time grid, schedule or GMM.")
        n_paths, seed = int(kl_mem["n_paths"]), int(kl_mem["seed"])
    k_theta = _reverse_kl_per_path(score_fn, ts, pi, mu, Sigma, n_paths, seed, None,
                                   beta_min, beta_max)
    k_mem = np.asarray(kl_mem["kl"])
    m_theta, m_mem = k_theta.mean(), k_mem.mean()
    sem = lambda v: float(v.std(ddof=1) / np.sqrt(n_paths))
    if m_mem == 0:
        return dict(rho_kl=np.nan, se=np.nan, kl_theta=float(m_theta), kl_theta_se=sem(k_theta),
                    kl_mem=0.0, kl_mem_se=sem(k_mem))
    ratio = m_theta / m_mem
    return dict(rho_kl=float(1 - ratio), se=float(sem(k_theta - ratio * k_mem) / abs(m_mem)),
                kl_theta=float(m_theta), kl_theta_se=sem(k_theta),
                kl_mem=float(m_mem), kl_mem_se=sem(k_mem))


def _reverse_kl_per_path(score_fn, ts, pi, mu, Sigma, n_paths, seed, d, beta_min, beta_max):
    """Per-path sums of the closed-form step KLs for reverse_kl; returns [n_paths]."""
    inv_a, coef, var = _reverse_steps(ts, beta_min, beta_max)
    d = mu.shape[1] if d is None else d
    eps = np.random.default_rng(seed).standard_normal((n_paths, len(ts), d))
    x = eps[:, -1]                                  # O_R ~ N(0, I)
    kl = np.zeros(n_paths)
    for j in range(len(ts) - 1, 0, -1):
        s_theta = score_fn(x, ts[j])
        diff = s_theta - gmm_score(x, ts[j], pi, mu, Sigma, beta_min, beta_max)
        kl += coef[j - 1]**2 * np.sum(diff**2, 1) / (2 * var[j - 1])
        x = inv_a[j - 1] * x + coef[j - 1] * s_theta + np.sqrt(var[j - 1]) * eps[:, j - 1]
    return kl


def rho(ll_true, ll_mem, ll_theta):
    """rho = (mean log p_theta - mean log p_mem) / (mean log p_true - mean log p_mem).

    Arguments are the path log-likelihoods [n] of the same n validation paths under the true,
    memorizing and trained distributions. rho = 1 when p_theta scores the paths like the true
    distribution, 0 when it scores them like the memorizing one.

    Path log-likelihoods are large (hundreds to thousands of nats) and share big terms, so the
    differences are taken per path before averaging rather than as differences of large means.
    """
    ll_true, ll_mem, ll_theta = (np.asarray(v, dtype=np.float64) for v in (ll_true, ll_mem, ll_theta))
    if ll_true.ndim != 1 or not ll_true.shape == ll_mem.shape == ll_theta.shape:
        raise ValueError("Pass three 1-d arrays of log-likelihoods of the same paths.")
    for name, v in [("ll_true", ll_true), ("ll_mem", ll_mem), ("ll_theta", ll_theta)]:
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} has {np.sum(~np.isfinite(v))} non-finite values.")
    num = np.mean(ll_theta - ll_mem)
    den = np.mean(ll_true - ll_mem)
    if den == 0:
        return np.nan  # true and memorizing distributions score the paths identically on average
    return float(num / den)


if __name__ == "__main__":
    # Compute KL(P_mem || P_true) once per training-set size and save it for rho_kl(..., kl_mem=...):
    #   python noisy_paths.py --out runs/gmm_kl_mem
    #   kl_mem = dict(np.load("runs/gmm_kl_mem/kl_mem_n512.npz"))
    import argparse
    import os
    import time

    from gmm_setup import make_datasets

    p = argparse.ArgumentParser(description="Precompute KL(P_mem || P_true) for each training-set size.")
    p.add_argument("--param_seed", type=int, default=0)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=50, help="log_time_grid(num_steps, t_min)")
    p.add_argument("--t_min", type=float, default=1e-3)
    p.add_argument("--n_paths", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0, help="noise seed; rho_kl reuses it for P_theta")
    p.add_argument("--out", default="runs/gmm_kl_mem")
    args = p.parse_args()

    data = make_datasets(param_seed=args.param_seed, data_seed=args.data_seed)
    ts = log_time_grid(args.num_steps, args.t_min)
    os.makedirs(args.out, exist_ok=True)
    for n in data["train_sizes"]:
        t0 = time.time()
        out = memorizing_kl(ts, data["pi"], data["mu"], data["Sigma"], data[f"x_train_{n}"],
                            args.n_paths, args.seed)
        out.update(param_seed=args.param_seed, data_seed=args.data_seed)
        path = os.path.join(args.out, f"kl_mem_n{n}.npz")
        np.savez(path, **out)
        print(f"n = {n:5d}: KL(P_mem || P_true) = {out['kl'].mean():.3f} "
              f"+- {out['kl'].std(ddof=1) / np.sqrt(args.n_paths):.3f}  [{time.time() - t0:.0f}s] -> {path}")
