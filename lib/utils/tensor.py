
import contextlib
from warnings import warn
import numpy as np
import torch
import torch.nn as nn


try:
    import numba
    maybe_jit = numba.jit
except Exception as e:
    warn("numba not found or failed to import, some operations may run slowly;\n"
         "Error message: {}".format(e))
    maybe_jit = nop

def check_numpy(x):
    """ Makes sure x is a numpy array """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


@maybe_jit



