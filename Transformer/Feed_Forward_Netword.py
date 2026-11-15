import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward output.
    """
    """FFN(x) = ReLU(xW1 + b1)W2 + b2"""

    x_W1_b1 = x @ W1 + b1
    relu = np.maximum(0, x_W1_b1)
    ffn = relu @ W2 + b2

    return ffn
    