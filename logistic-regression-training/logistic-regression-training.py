import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here

    n = X.shape[0]
    X = np.column_stack((np.ones(n), X))
    w = np.zeros(X.shape[-1])
    
    print(f"Shapes: {X.shape, w.shape}")
    print(X,w)
    for i in range(steps):
        z = X@np.transpose(w)
        pred = _sigmoid(z)
        diff = ((pred - y))
        # print(f"Difference Shape: {diff.shape}")
        loss_grad = 1/n * np.transpose(X) @  diff
        # print(loss_grad.shape)
        w -= (lr * loss_grad)
    
    return (w[1:],w[0])