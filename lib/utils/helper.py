import numpy as np
import torch


def update_learning_rate(stage:int, t:int, emb, prune_threshold=0.5, decrease_lr:bool = True):
    # reduce LR and prune PRODIGE graph
    if decrease_lr is False:
        emb = emb.pruned(threshold=prune_threshold)
        opt = torch.optim.SparseAdam(emb.parameters(), lr=0.01)
    else:
        if stage == 0:
            opt = torch.optim.SparseAdam(emb.parameters(), lr=0.1)
            stage += 1

        elif stage == 1: # and t >= 300:
            emb = emb.pruned(threshold=prune_threshold)
            opt = torch.optim.SparseAdam(emb.parameters(), lr=0.05)
            stage += 1

        elif stage == 2: # and t >= 1000:
            emb = emb.pruned(threshold=prune_threshold)
            opt = torch.optim.SparseAdam(emb.parameters(), lr=0.01)
            stage += 1

        elif stage == 3:
            emb = emb.pruned(threshold=prune_threshold)
            opt = torch.optim.SparseAdam(emb.parameters(), lr=0.005)
            stage += 1
    return stage, emb, opt


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


def inverse_softplus(x):
    return np.where(x > 10.0, x, np.log(np.expm1(np.clip(x, 1e-6, 10.0))))


def inverse_sigmoid(x):
    return -np.log(1. / np.clip(x, 1e-6, 1.0 - 1e-6) - 1)

