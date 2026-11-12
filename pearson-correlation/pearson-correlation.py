import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    sigma = np.cov(X, rowvar=False)
    sigma_sum = np.sum(sigma)
    vars = np.diag(sigma)
    stds = np.sqrt(vars)
    S=np.outer(stds, stds)
    return sigma/S