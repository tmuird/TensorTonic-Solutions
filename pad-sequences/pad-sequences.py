import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is None:
        max_len = max((len(s) for s in seqs), default=0)
    start = np.full((len(seqs), max_len), pad_value)

    for i, seq in enumerate(seqs):
        print(seq, start[i])
        start[i,:len(seq)] = seq[:max_len]
    # return np.column_stack(np.array(seqs),np.repeat)))
    return start