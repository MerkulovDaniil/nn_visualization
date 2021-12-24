import torch


def am(model, layer, filter, x0, device, lr=0.1, iters=30, eps=1.E-7):
    """Метод анализа активации Activation Maximization (AM).

        Args:
            model (torch.nn.Module): нейронная сеть.
            layer (str): имя слоя сети.
            filter (int): номер фильтра (нумерация с нуля).
            x0 (torch.tensor): начальный входной вектор для сети.
            device (str): используемое устройство ('cpu' или 'gpu').
            lr (float): параметр скорости обучения.
            iters (int): количество итераций.
            eps (float): параметр шума.

        Returns:
            bool: входной вектор для сети, максимизирующий активацию.

    """
    x = x0.detach().clone().to(device)
    x.requires_grad = True

    layer = model.features[layer]

    if type(layer) != torch.nn.modules.conv.Conv2d:
        raise ValueError('Метод AM работает только для сверточных слоев')
    if filter < 0 or filter >= layer.out_channels:
        raise ValueError('Указан несуществующий номер фильтра для AM')

    class AmHook():
        def __init__(self, filter, shape, device):
            self.filter = filter
            self.A = None

        def forward(self, module, inp, out):
            self.A = torch.mean(out[:, self.filter, :, :])

    hook = AmHook(filter, x.shape, device)
    handlers = [layer.register_forward_hook(hook.forward)]
    for i in range(iters):
        model(x)
        A = hook.A
        G = (torch.autograd.grad(A, x))[0]
        G /= torch.sqrt(torch.mean(torch.mul(G, G))) + eps
        x = x + G * lr

    while len(handlers) > 0:
        handlers.pop().remove()

    return x
