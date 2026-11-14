import numpy as np
import torch
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