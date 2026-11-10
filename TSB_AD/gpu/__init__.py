"""GPU implementations of TSB-AD's VUS metrics and subsequence detectors (PyTorch; CUDA or Apple MPS)."""

from .device import get_device, set_device
from .vus import vus
from .detectors import SubPCA, SubKNN, KMeansAD, MatrixProfile

__all__ = ["get_device", "set_device", "vus", "SubPCA", "SubKNN", "KMeansAD", "MatrixProfile"]
