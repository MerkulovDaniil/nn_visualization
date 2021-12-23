import numpy as np


def to_data(value):
    if value.__class__.__module__.startswith("torch"):
        import torch
        if isinstance(value, torch.nn.parameter.Parameter):
            value = value.data
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                value = value.detach()
            value = value.cpu().numpy().copy()
        if not value.shape:
            value = value.item()
    if value.__class__.__module__ == "numpy" and value.__class__.__name__ != "ndarray":
        value = value.item()
    return value


def write(*args):
    s = ""
    for a in args:
        a = to_data(a)

        if isinstance(a, np.ndarray):
            s += ("\t" if s else "") + "Tensor  {} {}  min: {:.3f}  max: {:.3f}".format(
                a.dtype, a.shape, a.min(), a.max())
            print(s)
            s = ""
        elif isinstance(a, list):
            s += ("\t" if s else "") + "list    len: {}  {}".format(len(a), a[:10])
        else:
            s += (" " if s else "") + str(a)
    if s:
        print(s)
