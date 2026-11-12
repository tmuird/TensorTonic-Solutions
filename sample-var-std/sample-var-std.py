import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    std = float(np.std(x, ddof=1))
    return {"variance": std**2, "standard_deviation": std}