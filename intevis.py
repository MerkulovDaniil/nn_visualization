from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
import torch
import torchvision
import urllib
import yaml


import sys
sys.path.append('./activation')
sys.path.append('./attribution')
sys.path.append('./gui')
sys.path.append('./gui/components')
sys.path.append('./gui/elements')


from am import am
from ig import ig
from sm import sm
from gui import Gui
from opts import opts


class Intevis:
    def __init__(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.opts = opts

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = None
        self.model_name = ''

        # Входной вектор модели:
        self.x_raw = None
        self.x = None

        # Выходной вектор модели:
        self.y = None
        self.y_name = ''

        self.am = None
        self.ig = None
        self.sm = None

    def download_imagenet_classes(self):
        classes = ''
        for f in urllib.request.urlopen(self.opts.url.imagenet_classes):
            classes = classes + f.decode('utf-8')
        self.classes = yaml.safe_load(classes)

    def gui(self, width='1200px', height='600px'):
        self.opts.app_width = width
        self.opts.app_height = height
        Gui(self, self.opts)

    def preprocess(self, image, size=224):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((size, size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            torchvision.transforms.Lambda(lambda x: x[None]),
        ])
        return transform(image)

    def run(self):
        self.model.eval()
        self.y = self.model(self.x)
        args = torch.argmax(self.y[0, :]).cpu().numpy().tolist()
        self.y_name = self.classes[args]

    def run_am(self, layer, filter, lr, iters, is_random, sz=224):
        if is_random:
            x = tensor_rand(sz)
        else:
            x = self.x

        self.am = am(self.model, layer, filter, x, self.device, lr, iters)

    def run_ar(self):
        return

    def run_ig(self, steps):
        x = torch.squeeze(self.x, 0)
        x = torch.swapaxes(x, 0, 1)
        x = torch.swapaxes(x, 1, 2)
        x = x.detach().numpy()

        self.ig = ig(self.model, x, steps, self.device)

    def run_sm(self):
        self.sm = sm(self.model, self.x)

    def set_image(self, data=None, link=None):
        if data is not None:
            self.x_raw = Image.open(BytesIO(data))
            self.x = self.preprocess(self.x_raw)
        elif link is not None:
            response = requests.get(link)
            self.x_raw = Image.open(BytesIO(response.content))
            self.x = self.preprocess(self.x_raw)
        self.x = self.x.to(self.device)

    def set_model(self, model=None, name=None):
        if model is None:
            self.model = torch.hub.load('pytorch/vision', name, pretrained=True)
            # self.model = torchvision.models.vgg16(pretrained=True)
            self.model.to(self.device)
        else:
            self.model = model

        self.model_name = name

    def plot(self, fpath=None):
        if self.x_raw is None:
            return False

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(self.x_raw)
        plt.axis('off')

        if fpath:
            plt.savefig(fpath, bbox_inches='tight')
            clear_plt()
            return True
        else:
            plt.show()

    def plot_am(self, fpath=None):
        if self.am is None:
            return False

        x = self.am.cpu()

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(tensor_to_plot(tensor_to_img(x, sat=0.2, br=0.8)))
        plt.axis('off')

        if fpath:
            plt.savefig(fpath, bbox_inches='tight')
            clear_plt()
            return True
        else:
            plt.show()

    def plot_ig(self, fpath=None):
        if self.ig is None:
            return False

        x = self.ig # It is numpy!

        fig = plt.figure(figsize=(6, 6))
        # TODO Check why np.uint8 below
        plt.imshow(np.uint8(x), cmap=plt.cm.hot)
        plt.axis('off')

        if fpath:
            plt.savefig(fpath, bbox_inches='tight')
            clear_plt()
            return True
        else:
            plt.show()

    def plot_sm(self, fpath=None):
        if self.sm is None:
            return False

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(self.sm, cmap=plt.cm.hot)
        plt.axis('off')

        if fpath:
            plt.savefig(fpath, bbox_inches='tight')
            clear_plt()
            return True
        else:
            plt.show()

    def plot_tmp(self, fname=None, is_cap=False):
        """Old plot."""
        img1 = self.x_raw.resize((224, 224), Image.ANTIALIAS)
        img2 = self.sm

        fig = plt.figure(figsize=(10, 5))
        plt.title('Saliency Map')
        plt.axis('off')

        ax = fig.add_subplot(1, 2, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Original')
        ax.imshow(img1)

        ax = fig.add_subplot(1, 2, 2)
        ax.set_xticks([])
        ax.set_yticks([])
        if is_cap:
            ax.set_title(f'Result [captum]')
        else:
            ax.set_title(f'Result')
        ax.imshow(img2, cmap=plt.cm.hot)

        plt.subplots_adjust(wspace=0, hspace=0)

        if fname:
            plt.savefig(fname, bbox_inches='tight')
            return True
        else:
            plt.show()

    def vis_arc_1(self):
        """Old architecture visualiztion #1."""
        text = ''
        features = self.model.features.named_children()
        for f in features:
            text += f'>>> {f}\n'
        print(text)

    def vis_arc_2(self):
        """Old architecture visualiztion #2."""
        import torchinfo

        text = torchinfo.summary(self.model)
        print(text)

    def vis_arc_3(self):
        """Old architecture visualiztion #3."""
        import torchsummary

        ch = 3
        sz = 224
        text = torchsummary.summary(self.model, (ch, sz, sz))
        print(text)

    def vis_arc_4(self):
        """Old architecture visualiztion #4."""
        import hiddenlayer

        ch = 3
        sz = 224
        g = hiddenlayer.build_graph(self.model, torch.zeros([1, ch, sz, sz]))
        d = g.build_dot()
        d.render('_tmp', format='png', view=True)
        fig = plt.figure(figsize=(22, 20))
        plt.imshow(Image.open('./_tmp.png'))
        plt.axis('off')
        plt.show()

    def vis_arc_5(self):
        """Old architecture visualiztion #5."""
        import torchviz

        x = tensor_rand(224)
        y = self.model(x)
        return torchviz.make_dot(y.mean(),
            params=dict(self.model.named_parameters()))


def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


def img_to_tensor(img, sz=224):
    """Transform PIL image to tensor.

    It returns a tensor of the shape [1, number of channels, height, width].

    """
    if not isinstance(img, Image.Image):
        img = torchvision.transforms.functional.to_pil_image(img)

    m = [0.485, 0.456, 0.406]
    v = [0.229, 0.224, 0.225]
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(sz),
        torchvision.transforms.CenterCrop(sz),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(m, v)
    ])

    return transform(img).unsqueeze(0)


def img_rand(sz):
    """Create random PIL image."""
    img = np.uint8(np.random.uniform(150, 180, (sz, sz, 3)))
    img = torchvision.transforms.functional.to_pil_image(img)
    return img


def tensor_rand(sz):
    """Create random PIL image and transform it to tensor."""
    return img_to_tensor(img_rand(sz))


def tensor_to_img(x, vmin=0., vmax=1., sat=0.1, br=0.5):
    """Normalise tensor with values between [vmin, vmax]."""
    x = x.detach().cpu()
    m = x.mean()
    s = x.std() or 1.E-8
    return x.sub(m).div(s).mul(sat).add(br).clamp(vmin, vmax)


def tensor_to_plot(x):
    """Prepare tensor for image plot.

    It returns a tensor of the shape [number of channels, height, width].

    """
    y = x.clone().squeeze(0) if len(x.shape) == 4 else x.clone()
    y = y.squeeze(0) if y.shape[0] == 1 else y.permute(1, 2, 0)
    return y.detach()
