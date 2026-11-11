import numpy as np

def dot_product(x, y) -> float:
    """
    Returns the dot product as a float.
    """
    
    return float(np.asarray(x, dtype=np.float64) @ np.asarray(y, dtype=np.float64))
    