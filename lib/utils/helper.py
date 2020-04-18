import numpy as np
import torch

def check_numpy(x):
    # Makes sure x is a numpy array
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


def sliced_argmax(inp, slices, out=None):
    # returns max value per specified slice indices for data 'inp', into 'out' array
    # or an ndarray of -1s if None is specified
    if out is None:
        out = np.full(len(slices) - 1, -1, dtype=np.int64)
    for i in range(len(slices) - 1):
        if slices[i] == slices[i + 1]: continue
        out[i] = np.argmax(inp[slices[i]: slices[i + 1]])
    return out