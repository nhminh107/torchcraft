import numpy as np

def softmax(x, axis=-1):
    """Provided: numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns x normalized over the last axis, then scaled by gamma and shifted by beta.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    norm = (x - mean) / np.sqrt(var + eps)
    return gamma * norm + beta


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    d_model = Q.shape[-1]
    d_k = d_model // num_heads

    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    heads = []

    for i in range(num_heads):
        start = i * d_k
        end = (i + 1)*d_k

        Q_i = Q_proj[..., start:end]
        K_i = K_proj[..., start:end]
        V_i = V_proj[..., start:end]

        scores = Q_i @ K_i.transpose(0, 2, 1)
        scores = scores / np.sqrt(d_k)


        exp_score = np.exp(
            scores - np.max(scores, axis=-1, keepdims=True)
        )
        attention = exp_score / np.sum(
            exp_score, axis=-1, keepdims=True
        )
        head_i = attention @ V_i
        heads.append(head_i)

    concat_heads = np.concatenate(heads, axis=-1)
    return concat_heads @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward network output with a ReLU hidden layer.
    """
    x_W1_b1 = x @ W1 + b1
    relu = np.maximum(0, x_W1_b1)
    ffn = relu @ W2 + b2
    
    return ffn

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    z = layer_norm(x + multi_head_attention(x,x,x, W_q, W_k, W_v, W_o, num_heads), gamma1, beta1)
    y = layer_norm(z + feed_forward(z, W1, b1, W2, b2), gamma2, beta2)

    return y


