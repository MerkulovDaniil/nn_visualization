import torch


def sm(model, x):
    """Метод атрибуции Saliency Map (SM).

        Args:
            model (torch.nn.Module): нейронная сеть.
            x (torch.tensor): входной вектор для сети.

        Returns:
            x (torch.tensor): карта атрибуции для входного вектора.

    """
    x.requires_grad_()
    scores = model(x)
    score_max_index = scores.argmax()
    score_max = scores[0, score_max_index]
    score_max.backward()
    res, _ = torch.max(x.grad.data.abs(), dim=1)
    return res[0]
