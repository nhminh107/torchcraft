import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    position_enc = []

    for idx in range(seq_length): 
        pe = []
        for i in range (d_model // 2):
            param = np.pow(10000, 2*i/d_model)
            
            pe.append(np.sin(idx/param))
            pe.append(np.cos(idx/param))

        pe_np = np.array(pe, dtype='float32')
        position_enc.append(pe_np)

    return np.array(position_enc, dtype='float32')


if __name__ == "__main__":
    print(positional_encoding(seq_length=3, d_model=4))