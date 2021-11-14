import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


DPATH = f'./../data/'


def build(A, img, with_mask=False):

    def linear_transform(A, clip_above=99, clip_below=0, low=0.):

        def calc_threshold(A, p=60):
            if p < 0 or p > 100:
                raise ValueError('Percentage (p) should be in >= 0 and <= 100')
            if p == 100:
                return np.min(A)
            A_flat = A.flatten()
            A_sum = np.sum(A_flat)
            A_sort = np.sort(np.abs(A_flat))[::-1]
            cum_sum = 100. * np.cumsum(A_sort) / A_sum
            ind = np.where(cum_sum >= p)[0][0]
            return A_sort[ind]

        m = calc_threshold(A, 100-clip_above)
        e = calc_threshold(A, 100-clip_below)
        res = (1 - low) * (np.abs(A) - e) / (m - e) + low
        res*= np.sign(A)
        res*= (res >= low)
        res = np.clip(res, 0., 1.)
        return res

    A = np.clip(A, 0., 1.)
    A = np.average(A, axis=2)
    A = linear_transform(A)
    A_mask = A.copy()
    A = np.expand_dims(A, 2) * [0, 255, 0]
    if with_mask:
        A = np.expand_dims(A_mask, 2)
        A = np.clip(A * img, 0, 255)
        A = A[:, :, (2, 1, 0)]
    return A


def prep(obs, device):
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    obs = obs / 255
    obs = (obs - mean) / std
    obs = np.transpose(obs, (2, 0, 1))
    obs = np.expand_dims(obs, 0)
    obs = np.array(obs)
    obs = torch.tensor(obs, dtype=torch.float32, device=device, requires_grad=True)
    return obs


def run(inputs, model, device, target_label_idx=None):
    G_list = []
    for input in inputs:
        input = prep(input, device)
        output = model(input)
        output = F.softmax(output, dim=1)
        if target_label_idx is None:
            target_label_idx = torch.argmax(output, 1).item()
        index = np.ones((output.size()[0], 1)) * target_label_idx
        index = torch.tensor(index, dtype=torch.int64, device=device)
        output = output.gather(1, index)
        model.zero_grad()
        output.backward()
        G = input.grad.detach().cpu().numpy()[0]
        G_list.append(G)
    return np.array(G_list), target_label_idx


def run_ig(inputs, model, device, target_label_idx, steps=50, num_random_trials=10):
    all_intgrads = []
    for i in range(num_random_trials):
        baseline = 255.0 *np.random.random(inputs.shape)
        scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
        grads, _ = run(scaled_inputs, model, device, target_label_idx)
        avg_grads = np.average(grads[:-1], axis=0)
        avg_grads = np.transpose(avg_grads, (1, 2, 0))
        delta_X = (prep(inputs, device) - prep(baseline, device)).detach().squeeze(0).cpu().numpy()
        delta_X = np.transpose(delta_X, (1, 2, 0))
        integrated_grad = delta_X * avg_grads
        all_intgrads.append(integrated_grad)
    return np.average(np.array(all_intgrads), axis=0)


def visualize(img, res, res_overlay, title, fname=None):
    fig = plt.figure(figsize=(16, 5))
    plt.axis('off')

    ax = fig.add_subplot(1, 3, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Original image')
    ax.imshow(np.uint8(img))

    ax = fig.add_subplot(1, 3, 2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Overlay for ' + title)
    ax.imshow(np.uint8(res_overlay))

    ax = fig.add_subplot(1, 3, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Result for ' + title)
    ax.imshow(np.uint8(res))

    plt.subplots_adjust(wspace=0, hspace=0)

    if fname:
        plt.savefig(fname, bbox_inches='tight')

    plt.show()


def ig(model, img, steps, device):
    model.eval()

    # import cv2
    # img = cv2.imread(DPATH + './img1.jpg').astype(np.float32)[:, :, (2, 1, 0)]

    G, k = run([img], model, device)
    A = run_ig(img, model, device, k, steps)
    res = build(A, img)
    # res_overlay = build(A, img, with_mask=True)
    return res

    visualize(img, res, res_overlay, 'Integrated Gradients', 'res1.png')

    G = np.transpose(G[0], (1, 2, 0))
    res = build(G, img)
    res_overlay = build(G, img, with_mask=True)

    visualize(img, res, res_overlay, 'Base Gradients', 'res1_grad.png')
