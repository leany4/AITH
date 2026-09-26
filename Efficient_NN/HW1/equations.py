"""FP32, S divisible by 16. Scalar inputs or broadcastable NumPy arrays."""
import numpy as np

PARAMETERS = 1_040_324
WEIGHT_BYTES = 4 * PARAMETERS


def _inputs(image_size, batch):
    s, b = np.broadcast_arrays(np.asarray(image_size, dtype=float),
                               np.asarray(batch, dtype=float))
    if np.any(~np.isfinite(s) | ~np.isfinite(b) | (s < 16) | (s % 16 != 0)
              | (b < 1) | (b % 1 != 0)):
        raise ValueError('S must be a positive multiple of 16; B a positive integer')
    return s, b


def flops(image_size, batch):
    """MAC=2; ReLU/MaxPool comparisons=1; include GAP and Linear bias."""
    s, b = _inputs(image_size, batch)
    return b * (17_751 * s**2 + 313_956)


def memory(image_size, batch):
    """Live tensor baseline, bytes; no fitted terms or cuDNN workspace."""
    s, b = _inputs(image_size, batch)
    return WEIGHT_BYTES + 52 * b * s**2


def bytes_moved(image_size, batch):
    """One logical read/write per operator tensor, not measured DRAM traffic."""
    s, b = _inputs(image_size, batch)
    return 4 * (PARAMETERS + b * (91 * s**2 + 2148))


def latency(image_size, batch, theta):
    """Seconds: launch floor + roofline bottleneck."""
    return (theta['launch_s']
            + np.maximum(theta['seconds_per_flop'] * flops(image_size, batch),
                         theta['seconds_per_byte'] * bytes_moved(image_size, batch)))


def energy(image_size, batch, theta_energy):
    """Whole-GPU joules, including idle/background power during the pass."""
    return (theta_energy['base_power_w'] * latency(image_size, batch, theta_energy['latency'])
            + theta_energy['joules_per_flop'] * flops(image_size, batch)
            + theta_energy['joules_per_byte'] * bytes_moved(image_size, batch))
