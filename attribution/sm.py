import torch


def sm(model, x):
    x.requires_grad_()
    scores = model(x)
    score_max_index = scores.argmax()
    score_max = scores[0, score_max_index]
    score_max.backward()
    res, _ = torch.max(x.grad.data.abs(), dim=1)
    return res[0]
