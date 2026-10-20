"""Check that this machine can run score_sde_pytorch (with the mac-compat changes applied).

Run from the repository root, inside your Python environment:

    python check_env.py                   # all checks (first run downloads CIFAR-10 and Inception)
    python check_env.py --skip-data       # skip the CIFAR-10 check
    python check_env.py --skip-eval       # skip the FID / KID / IS checks (stats file, Inception)
    python check_env.py --benchmark       # also time training steps of the full-size DDPM model
    python check_env.py --verbose         # print full tracebacks for failures

Each check prints [ OK ] or [FAIL] (with a hint). Exit code is 0 only if every check passed.

Downloads: CIFAR-10 goes to ~/tensorflow_datasets; the Inception network goes to $TFHUB_CACHE_DIR
(default: a tfhub_modules folder in the system temp directory, which may be wiped, so consider
`export TFHUB_CACHE_DIR=$HOME/.cache/tfhub_modules`). On a cluster whose compute nodes have no
internet, run this once on a login node to fill both caches.

If the script dies with "OMP: Error #15 ... libomp already initialized", run
`export KMP_DUPLICATE_LIB_OK=TRUE` and try again.
"""
import argparse
import importlib
import importlib.util
import os
import platform
import sys
import tempfile
import time
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # quieten TensorFlow's start-up logging

RESULTS = []
VERBOSE = False


def check(name, fn, hint="", optional=False):
    t0 = time.time()
    try:
        info = fn()
    except Exception as e:  # noqa: BLE001 - we want to report anything that goes wrong
        msg = f"{type(e).__name__}: {e}".splitlines()[0][:300]
        RESULTS.append((name, False, optional))
        tag = "[ -- ]" if optional else "[FAIL]"
        print(f"{tag} {name} - {msg}")
        if hint:
            print(f"       hint: {hint}")
        if VERBOSE:
            traceback.print_exc()
        return None
    RESULTS.append((name, True, optional))
    extra = f" - {info}" if info else ""
    print(f"[ OK ] {name}{extra} ({time.time() - t0:.1f}s)")
    return info if info is not None else True


def section(title):
    print(f"\n== {title}")


def version_of(module_name):
    def _f():
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", "installed")
    return _f


def load_config(path):
    spec = importlib.util.spec_from_file_location("cfg_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    global VERBOSE
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    VERBOSE = args.verbose

    if not os.path.exists("main.py") or not os.path.isdir("configs"):
        sys.exit("Run this from the root of your score_sde_pytorch checkout (the folder with main.py).")
    sys.path.insert(0, os.getcwd())

    # ------------------------------------------------------------------ 1
    section("1. Python and platform")
    print(f"       python {sys.version.split()[0]} at {sys.executable}")
    print(f"       {platform.system()} {platform.release()} on {platform.machine()}")

    # ------------------------------------------------------------------ 2
    section("2. Packages (training, sampling, likelihoods, FID / KID / IS)")
    # TensorFlow first, mirroring the import order of main.py / run_lib.py
    required = [
        ("tensorflow", "pip install tensorflow  (if pip finds no version for your Python, "
                       "make a fresh env with python=3.11)"),
        ("tensorflow_datasets", "pip install tensorflow-datasets"),
        ("importlib_resources", "pip install importlib-resources  (tensorflow-datasets needs it but "
                                "doesn't declare it for Python >= 3.9)"),
        ("tensorboard", "pip install tensorboard"),
        ("tf_keras", "pip install tf-keras  (its version must match tensorflow's)"),
        ("tensorflow_hub", "pip install tensorflow-hub tf-keras"),
        ("torch", "pip install torch torchvision"),
        ("torchvision", "pip install torchvision"),
        ("numpy", "pip install numpy"),
        ("scipy", "pip install scipy"),
        ("ml_collections", "pip install ml-collections"),
        ("absl", "pip install absl-py"),
    ]
    for mod, hint in required:
        check(f"import {mod}", version_of(mod), hint=hint)

    # As in eval mode of main.py: let TensorFlow use a GPU only as needed, so it doesn't reserve
    # memory that PyTorch needs later in this script.
    try:
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    try:
        import torch
    except Exception:
        print("\nPyTorch is missing, so the remaining checks can't run.")
        return summarise()

    # ------------------------------------------------------------------ 3
    section("3. Devices")
    print(f"       CUDA available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    print(f"       Apple MPS available: {mps_ok}")

    def chosen_device():
        from configs.default_cifar10_configs import pick_device
        return str(pick_device())
    dev_name = check("device picked by the configs", chosen_device,
                     hint="pick_device() is missing: apply the mac-compat patch (git apply ...).")
    device = torch.device(dev_name) if dev_name else torch.device("cpu")
    if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
        print("       note: consider `export PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported ops fall back to the CPU")

    # ------------------------------------------------------------------ 4
    section("4. Repository code")
    repo_ok = check("import run_lib (pulls in every model, SDE, sampler and loss module)",
                    lambda: importlib.import_module("run_lib") and None,
                    hint="If this tries to compile CUDA code on a Mac, the mac-compat patch isn't applied. "
                         "Rerun with --verbose for the full traceback.")
    if not repo_ok:
        print("\nThe repository code doesn't import, so the remaining checks can't run.")
        return summarise()

    def native_upfirdn():
        from op import upfirdn2d
        x = torch.randn(1, 3, 8, 8, device=device)
        k = torch.ones(2, 2, device=device) / 4
        y = upfirdn2d(x, k, up=2, pad=(0, 1))
        assert y.shape == (1, 3, 16, 16), y.shape
        return f"output {tuple(y.shape)} on {y.device.type}"
    check("upfirdn2d op runs on this device (CUDA kernel or PyTorch fallback)", native_upfirdn)

    # ------------------------------------------------------------------ 5
    section("5. Config and a tiny DDPM model")
    cfg_path = os.path.join("configs", "vp", "ddpm", "cifar10_continuous.py")
    def config_summary():
        c = load_config(cfg_path)
        return (f"nf={c.model.nf} ch_mult={tuple(c.model.ch_mult)} res_blocks={c.model.num_res_blocks} "
                f"attn={tuple(c.model.attn_resolutions)} batch={c.training.batch_size} lr={c.optim.lr} "
                f"device={c.device}")
    if check(f"load {cfg_path}", config_summary) is None:
        return summarise()

    import numpy as np
    import losses
    import sampling
    import sde_lib
    from models import utils as mutils
    from models.ema import ExponentialMovingAverage

    small = load_config(cfg_path)
    small.device = device
    small.model.nf = 32                      # smallest width that works with 32-group GroupNorm
    small.model.num_res_blocks = 1
    small.training.batch_size = 8
    small.optim.warmup = 0                   # default 5,000-step warmup would make lr ~0 for 3 steps
    sde = sde_lib.VPSDE(beta_min=small.model.beta_min, beta_max=small.model.beta_max,
                        N=small.model.num_scales)
    state = {}

    def build_and_train():
        model = mutils.create_model(small)
        ema = ExponentialMovingAverage(model.parameters(), decay=small.model.ema_rate)
        opt = losses.get_optimizer(small, model.parameters())
        state.update(model=model, ema=ema, optimizer=opt, step=0)
        step_fn = losses.get_step_fn(sde, train=True, optimize_fn=losses.optimization_manager(small),
                                     reduce_mean=small.training.reduce_mean, continuous=True)
        x = torch.rand(8, 3, 32, 32, device=device) * 2 - 1
        before = next(model.parameters()).detach().clone()
        lossvals = [float(step_fn(state, x)) for _ in range(3)]
        assert all(np.isfinite(lossvals)), lossvals
        assert not torch.equal(before, next(model.parameters()).detach()), "parameters did not change"
        n = sum(p.numel() for p in model.parameters())
        return f"{n / 1e6:.2f}M params, 3 steps, losses {[round(v, 3) for v in lossvals]}"
    trained = check("build model and take 3 training steps (with EMA)", build_and_train)

    if trained:
        def score_and_grad():
            score_fn = mutils.get_score_fn(sde, state["model"], train=False, continuous=True)
            x = torch.randn(4, 3, 32, 32, device=device, requires_grad=True)
            t = torch.tensor([1e-3, 0.1, 0.5, 1.0], device=device)
            s = score_fn(x, t)
            assert s.shape == x.shape and torch.isfinite(s).all()
            (s * torch.randn_like(s)).sum().backward()
            assert x.grad is not None and torch.isfinite(x.grad).all()
            return "score finite at t in {1e-3, 0.1, 0.5, 1}; gradient w.r.t. x works"
        check("score function s(x, t) and autograd through it", score_and_grad)

        def short_sampler():
            short_sde = sde_lib.VPSDE(beta_min=small.model.beta_min, beta_max=small.model.beta_max, N=10)
            fn = sampling.get_pc_sampler(short_sde, (4, 3, 32, 32),
                                         sampling.get_predictor("euler_maruyama"),
                                         sampling.get_corrector("none"),
                                         lambda x: (x + 1) / 2, snr=0.16, continuous=True,
                                         denoise=True, eps=1e-3, device=device)
            x, nfe = fn(state["model"])
            assert torch.isfinite(x).all()
            return f"10-step Euler-Maruyama sample {tuple(x.shape)}"
        check("reverse-SDE sampler", short_sampler)

        def ode_dtype():
            z = mutils.from_flattened_numpy(np.zeros(12, dtype=np.float64), (3, 4)).to(device)
            assert z.dtype == torch.float32
            return "float64 -> float32 before moving to the device"
        check("numpy/scipy hand-off used by the ODE sampler and likelihood", ode_dtype,
              hint="On a Mac this needs the float32 fix in models/utils.py (mac-compat patch).")

        def checkpoint_roundtrip():
            from utils import restore_checkpoint, save_checkpoint
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "ckpt.pth")
                save_checkpoint(path, state)
                # Checkpoints trained on NVIDIA GPUs (e.g. the pretrained ones) hold `sigmas` in float64;
                # make this one look like that too, to check it still loads here.
                raw = torch.load(path, map_location="cpu", weights_only=False)
                for k in list(raw["model"].keys()):
                    if k.endswith("sigmas"):
                        raw["model"][k] = raw["model"][k].double()
                torch.save(raw, path)
                fresh = mutils.create_model(small)
                new_state = dict(model=fresh, ema=ExponentialMovingAverage(fresh.parameters(), decay=0.9999),
                                 optimizer=losses.get_optimizer(small, fresh.parameters()), step=0)
                new_state = restore_checkpoint(path, new_state, device)
                a = next(state["model"].parameters()).detach()
                b = next(new_state["model"].parameters()).detach()
                assert new_state["step"] == state["step"] and torch.equal(a, b)
                assert all(s.device.type == device.type for s in new_state["ema"].shadow_params)
                x = torch.rand(2, 3, 32, 32, device=device) * 2 - 1
                step_fn = losses.get_step_fn(sde, train=True, optimize_fn=losses.optimization_manager(small),
                                             reduce_mean=small.training.reduce_mean, continuous=True)
                assert np.isfinite(float(step_fn(new_state, x)))   # training resumes from the restored state
            return f"saved, restored (incl. float64 buffers) and resumed training at step {state['step']}"
        check("checkpoint save/restore", checkpoint_roundtrip)

    # ------------------------------------------------------------------ 6
    section("6. CIFAR-10 through the repo's TensorFlow data pipeline")
    if args.skip_data:
        print("[ -- ] skipped (--skip-data)")
    else:
        def data_batch():
            import datasets
            data_cfg = load_config(cfg_path)
            data_cfg.training.batch_size = 8
            data_cfg.eval.batch_size = 8
            train_ds, eval_ds, _ = datasets.get_dataset(data_cfg)
            img = next(iter(train_ds))["image"]._numpy()
            assert img.shape == (8, 32, 32, 3), img.shape
            assert -1e-3 <= img.min() and img.max() <= 1 + 1e-3, (img.min(), img.max())
            return f"train batch {img.shape}, values in [{img.min():.2f}, {img.max():.2f}]"
        check("load a training batch (downloads CIFAR-10 to ~/tensorflow_datasets the first time)",
              data_batch, hint="Needs internet the first time. Rerun with --verbose for details.")

    # ------------------------------------------------------------------ 7
    section("7. FID / KID / IS evaluation")
    if args.skip_eval:
        print("[ -- ] skipped (--skip-eval)")
    else:
        run_eval_checks()

    # ------------------------------------------------------------------ 8
    if args.benchmark:
        section("8. Speed of the full-size DDPM (vp/ddpm/cifar10_continuous)")
        for bs in (16, 128):
            def bench(bs=bs):
                cfg = load_config(cfg_path)
                cfg.device = device
                cfg.training.batch_size = bs
                model = mutils.create_model(cfg)
                st = dict(model=model, ema=ExponentialMovingAverage(model.parameters(), decay=0.9999),
                          optimizer=losses.get_optimizer(cfg, model.parameters()), step=0)
                step_fn = losses.get_step_fn(sde, train=True, optimize_fn=losses.optimization_manager(cfg),
                                             reduce_mean=cfg.training.reduce_mean, continuous=True)
                x = torch.rand(bs, 3, 32, 32, device=device) * 2 - 1
                for _ in range(3):
                    step_fn(st, x)
                sync(device)
                t0, n = time.time(), 10
                for _ in range(n):
                    step_fn(st, x)
                sync(device)
                rate = n / (time.time() - t0)
                return f"{rate:.2f} it/s -> ~{100_000 / rate / 3600:.1f} h per 100k steps"
            check(f"batch {bs}", bench)

    return summarise()


def run_eval_checks():
    """FID / KID / IS: metric functions, CIFAR-10 reference statistics, Inception network."""
    import numpy as np
    import tfgan_metrics

    def metric_sanity():
        rng = np.random.default_rng(0)
        a = rng.standard_normal((2000, 64))
        fid_same = tfgan_metrics.frechet_classifier_distance_from_activations(a, a)
        fid_shift = tfgan_metrics.frechet_classifier_distance_from_activations(a, a + 0.5)
        kid = tfgan_metrics.kernel_classifier_distance_from_activations(a, rng.standard_normal((2000, 64)))
        is_uniform = tfgan_metrics.classifier_score_from_logits(np.zeros((100, 10)))
        assert abs(fid_same) < 1e-6 and abs(fid_shift - 64 * 0.25) < 1e-6, (fid_same, fid_shift)
        assert abs(kid) < 0.01 and abs(is_uniform - 1) < 1e-9, (kid, is_uniform)
        return "FID, KID and Inception Score give the expected values on synthetic data"
    check("metric functions (tfgan_metrics.py)", metric_sanity)

    stats_path = os.path.join("assets", "stats", "cifar10_stats.npz")

    def reference_stats():
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"{stats_path} not found")
        pools = np.load(stats_path)["pool_3"]
        assert pools.ndim == 2 and pools.shape[1] == 2048, pools.shape
        half = len(pools) // 2
        fid = tfgan_metrics.frechet_classifier_distance_from_activations(pools[:half], pools[half:])
        kid = tfgan_metrics.kernel_classifier_distance_from_activations(pools[:half], pools[half:])
        return (f"pool_3 {pools.shape}; between its two halves FID {fid:.2f}, KID {kid:.1e} "
                f"(expect FID of a few units from finite-sample bias, and KID near 0)")
    check("CIFAR-10 reference statistics for FID / KID", reference_stats,
          hint="Download cifar10_stats.npz from "
               "https://drive.google.com/file/d/14UB27-Spi8VjZYKST3ZcT8YVhAluiFWI/view?usp=sharing "
               "and save it as assets/stats/cifar10_stats.npz in the repo (or use --skip-eval for now).")

    def inception_features():
        import evaluation
        model = evaluation.get_inception_model(inceptionv3=False)
        imgs = np.random.default_rng(0).integers(0, 256, (8, 32, 32, 3), dtype=np.uint8)
        out = evaluation.run_inception_distributed(imgs, model, inceptionv3=False)
        pool, logits = out["pool_3"].numpy(), out["logits"].numpy()
        assert pool.shape == (8, 2048) and logits.shape[0] == 8, (pool.shape, logits.shape)
        assert np.isfinite(pool).all() and np.isfinite(logits).all()
        cache = os.environ.get("TFHUB_CACHE_DIR",
                               os.path.join(tempfile.gettempdir(), "tfhub_modules"))
        return f"pool_3 {pool.shape}, logits {logits.shape}; cached in {cache}"
    check("Inception network from TF-Hub (downloads it the first time)", inception_features,
          hint="Needs internet the first time. Rerun with --verbose for the full traceback.")


def summarise():
    failed = [n for n, ok, opt in RESULTS if not ok and not opt]
    missing_opt = [n for n, ok, opt in RESULTS if not ok and opt]
    print("\n== Summary")
    if failed:
        print(f"{len(failed)} required check(s) failed:")
        for n in failed:
            print(f"  - {n}")
    else:
        print("All required checks passed.")
    if missing_opt:
        print("Optional checks that didn't pass: " + ", ".join(missing_opt))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
