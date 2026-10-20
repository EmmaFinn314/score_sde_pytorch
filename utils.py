import torch
import tensorflow as tf
import os
import logging


def _to_device(obj, device):
  """Move the tensors in a (nested) checkpoint entry to `device`.

  Apple MPS has no float64, so float64 tensors are cast to float32 when `device` is MPS.
  """
  if torch.is_tensor(obj):
    if device.type == 'mps' and obj.dtype == torch.float64:
      obj = obj.float()
    return obj.to(device)
  if isinstance(obj, dict):
    out = type(obj)((k, _to_device(v, device)) for k, v in obj.items())
    if hasattr(obj, '_metadata'):  # keep the version info that nn.Module.state_dict() attaches
      out._metadata = obj._metadata
    return out
  if isinstance(obj, (list, tuple)):
    return type(obj)(_to_device(v, device) for v in obj)
  return obj


def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    device = torch.device(device)
    # Load on the CPU first (MPS can't receive float64 tensors), then move what needs moving.
    loaded_state = torch.load(ckpt_dir, map_location='cpu', weights_only=False)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])  # moves its state to the params' device
    state['model'].load_state_dict(_to_device(loaded_state['model'], device), strict=False)
    state['ema'].load_state_dict(_to_device(loaded_state['ema'], device))
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)