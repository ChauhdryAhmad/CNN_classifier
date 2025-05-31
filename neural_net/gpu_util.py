import numpy as np

try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False


def to_gpu(array):
    """Safely move data to GPU with proper type checking"""
    if not GPU_AVAILABLE:
        return array

    # Handle None and non-array cases
    if array is None:
        return None

    # Convert numpy arrays
    if isinstance(array, np.ndarray):
        return cp.asarray(array)

    # Convert Python sequences
    if isinstance(array, (list, tuple)):
        return cp.asarray(np.array(array))

    # Return unchanged if not convertible
    return array


def to_cpu(array):
    """Safely move data back to CPU"""
    if not GPU_AVAILABLE:
        return array

    if isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array


def get_xp(array=None):
    """Get correct module (numpy or cupy)"""
    if array is not None and GPU_AVAILABLE and isinstance(array, cp.ndarray):
        return cp
    return np
