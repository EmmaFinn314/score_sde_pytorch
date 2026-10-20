"""Check that this machine can run score_sde_pytorch (with the mac-compat changes applied).

Run from the repository root, inside your Python environment:

    python check_env.py               # all checks; downloads CIFAR-10 (~160 MB) the first time
    python check_env.py --skip-data   # skip the dataset check
    python check_env.py --benchmark   # also time training steps of the full-size DDPM model
    python check_env.py --verbose     # print full tracebacks for failures

Each check prints [ OK ], [FAIL] (with a hint) or [ -- ] for optional pieces that are only
needed by the original FID/IS evaluation pipeline. Exit code is 0 only if every required check passed.
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
    section("2. Packages needed for training and sampling")
    # TensorFlow first, mirroring the import order of main.py / run_lib.py
    required = [
        ("tensorflow", "pip install tensorflow  (if pip finds no version for your Python, "
                       "make a fresh env with python=3.11)"),
        ("tensorflow_datasets", "pip install tensorflow-datasets"),
        ("tensorboard", "pip install tensorboard"),
        ("torch", "pip install torch torchvision"),
        ("torchvision", "pip install torchvision"),
        ("numpy", "pip install numpy"),
        ("scipy", "pip install scipy"),
        ("ml_collections", "pip install ml-collections"),
        ("absl", "pip install absl-py"),
    ]
    for mod, hint in required:
        check(f"import {mod}", version_of(mod), hint=hint)

    section("2b. Packages only needed by the original FID/IS evaluation (optional for now)")
    for mod, hint in [("tensorflow_gan", "pip install tensorflow-gan"),
                      ("tensorflow_hub", "pip install tensorflow-hub"),
                      ("jax", "pip install jax")]:
        check(f"import {mod}", version_of(mod), hint=hint, optional=True)

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
    config = check(f"load {cfg_path}", lambda: load_config(cfg_path))
    if config is None:
        return summarise()
    print(f"       default model: nf={config.model.nf} ch_mult={tuple(config.model.ch_mult)} "
          f"res_blocks={config.model.num_res_blocks} attn={tuple(config.model.attn_resolutions)} "
          f"batch={config.training.batch_size} lr={config.optim.lr}")

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
                fresh = mutils.create_model(small)
                new_state = dict(model=fresh, ema=ExponentialMovingAverage(fresh.parameters(), decay=0.9999),
                                 optimizer=losses.get_optimizer(small, fresh.parameters()), step=0)
                new_state = restore_checkpoint(path, new_state, device)
                a = next(state["model"].parameters()).detach()
                b = next(new_state["model"].parameters()).detach()
                assert new_state["step"] == state["step"] and torch.equal(a, b)
            return f"saved and restored at step {state['step']}"
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
    if args.benchmark:
        section("7. Speed of the full-size DDPM (vp/ddpm/cifar10_continuous)")
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
        print("Optional (original FID/IS evaluation only): " + ", ".join(missing_opt))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
