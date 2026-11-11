import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    # Write code here
    matrix = np.asarray(matrix, dtype=float)
    ord_ = np.inf if norm_type == "max" else int(norm_type[1:])

    if axis is None:
        norm = np.linalg.norm(matrix.ravel(), ord=ord_)  
    else:
        norm = np.linalg.norm(matrix, ord=ord_, axis=axis, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return matrix / norm