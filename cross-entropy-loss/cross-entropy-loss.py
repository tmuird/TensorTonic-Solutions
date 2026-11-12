import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    # Write code here
    classes = np.max(y_true)+1
    onehot = np.eye(classes)[y_true] 
    out = onehot * np.log( y_pred)
    return - np.mean(np.sum(out,axis=1))