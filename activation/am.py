import torch


class AmHook():
    def __init__(self, filter, shape, device):
        self.filter = filter
        self.A = None
        self.G = torch.zeros(shape).to(device)

    def forward(self, module, inp, out):
        self.A = torch.mean(out[:, self.filter, :, :])

    def backward(self, module, inp, out):
        if self.G.shape != inp[0].shape:
            return
        self.G = inp[0]


def am(model, layer, filter, x0, device, lr=0.1, iters=30, eps=1.E-7):
    """Activation maximization for selected layer and filter for the model."""
    x = x0.detach().clone().to(device)
    x.requires_grad = True

    layer = model.features[layer]

    assert(type(layer) == torch.nn.modules.conv.Conv2d)
    assert(filter >= 0 and filter < layer.out_channels)

    hook = AmHook(filter, x0.shape, device)
    handlers = [layer.register_forward_hook(hook.forward)]
    for _, module in model.named_modules():
        if not isinstance(module, torch.nn.modules.conv.Conv2d):
            continue
        if module.in_channels != 3:
            continue
        handlers.append(module.register_backward_hook(hook.backward))
        break

    for i in range(iters):
        model(x)
        hook.A.backward()
        hook.G /= torch.sqrt(torch.mean(torch.mul(hook.G, hook.G))) + eps
        x = x + hook.G * lr

    while len(handlers) > 0:
        handlers.pop().remove()

    return x




def plot_ams_old(Y, filters=None, title='Conv2d', num_cols=4, fname=None):
    """Visualization of one or more results."""
    if not isinstance(Y, list): # Plot only one filter
        plt.figure(figsize=(4, 4))
        plt.axis('off')
        plt.title(title + f' | Filter {filters}' if filters else title)
        plt.imshow(tensor_to_plot(tensor_to_img(y, sat=0.2, br=0.8)))
        if fname:
            plt.savefig(fname, bbox_inches='tight')
        plt.show()
        return

    num_rows = int(np.ceil(len(Y) / num_cols))

    fig = plt.figure(figsize=(16, num_rows * (num_cols + 1)))
    plt.title(title)
    plt.axis('off')

    for i, filter in enumerate(filters):
        ax = fig.add_subplot(num_rows, num_cols, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Filter {filter}')
        ax.imshow(tensor_to_plot(tensor_to_img(Y[i], sat=0.2, br=0.8)))

    plt.subplots_adjust(wspace=0, hspace=0)

    if fname:
        plt.savefig(fname, bbox_inches='tight')

    plt.show()


def plot_dream_old(x, y):
    """Visualization of deep dream result."""
    fig = plt.figure(figsize=(10, 5))
    plt.title('DeepDream')
    plt.axis('off')

    ax = fig.add_subplot(1, 2, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Original')
    ax.imshow(tensor_to_plot(tensor_to_img(x, sat=0.2, br=0.8)))

    ax = fig.add_subplot(1, 2, 2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Result')
    ax.imshow(tensor_to_plot(tensor_to_img(y, sat=0.2, br=0.8)))

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
