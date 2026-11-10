"""Device selection for the GPU implementations.

The device is chosen once per process: the ``TSB_AD_DEVICE`` environment variable if set
(``cuda``, ``cuda:1``, ``mps`` or ``cpu``), otherwise CUDA, then Apple MPS, then CPU.
"""

import os

import numpy as np
import torch

_DEVICE = None


def get_device():
    global _DEVICE
    if _DEVICE is None:
        name = os.environ.get("TSB_AD_DEVICE")
        if name:
            _DEVICE = torch.device(name)
        elif torch.cuda.is_available():
            _DEVICE = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            _DEVICE = torch.device("mps")
        else:
            _DEVICE = torch.device("cpu")
    return _DEVICE


def set_device(device):
    """Overrides the device used by every GPU routine in this process."""
    global _DEVICE
    _DEVICE = torch.device(device)


def supports_float64(device=None):
    device = device or get_device()
    return device.type != "mps"


def to_device(a, dtype=torch.float32, device=None):
    device = device or get_device()
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)


def synchronize(device=None):
    device = device or get_device()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
