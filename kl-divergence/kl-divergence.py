import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    # Write code here
    p = np.array(p)
    q = np.array(q)
    ratio = p / np.clip(q,eps,None)
    mask = p > 0
    kld = np.sum(p[mask] * np.log(ratio[mask]))
    return float(kld) 