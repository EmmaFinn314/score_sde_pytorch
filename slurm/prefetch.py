"""Download everything the Slurm jobs need, so they don't rely on internet access on compute nodes.

Run once on a login node, from the repository root:
    source slurm/env.sh
    python slurm/prefetch.py

Downloads CIFAR-10 (to ~/tensorflow_datasets), the Inception network used for FID and sliced
Wasserstein (to $TFHUB_CACHE_DIR) and the SSCD model used for the memorization score (to
~/.cache/sscd), then checks that each one loads and produces features of the right shape.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402

tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds  # noqa: E402
import torch  # noqa: E402

import eval_mw  # noqa: E402


def main():
  if not os.environ.get("TFHUB_CACHE_DIR"):
    sys.exit("TFHUB_CACHE_DIR is not set, so the Inception network would be cached in /tmp, which "
             "compute nodes can't see. Run `source slurm/env.sh` first.")

  print("CIFAR-10 ...", flush=True)
  builder = tfds.builder("cifar10")
  builder.download_and_prepare()
  splits = builder.info.splits
  print(f"  {splits['train'].num_examples} train / {splits['test'].num_examples} test images "
        f"in {builder.data_dir}")
  images = eval_mw.load_split("train[:8]")

  print("Inception (TF-Hub) ...", flush=True)
  feats = eval_mw.InceptionFeatures(batch=8)(images)
  assert feats.shape == (8, 2048), feats.shape
  print(f"  pool_3 features {feats.shape}, cached in {os.environ['TFHUB_CACHE_DIR']}")

  print("SSCD ...", flush=True)
  emb = eval_mw.SSCDFeatures("", torch.device("cpu"))(images)
  assert emb.shape[0] == 8, emb.shape
  print(f"  embeddings {tuple(emb.shape)}, cached in ~/.cache/sscd")
  print("All downloaded.")


if __name__ == "__main__":
  main()
