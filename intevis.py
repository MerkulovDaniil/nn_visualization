from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
import torch
import torchvision
import urllib
import yaml


from activation.am import am
from attribution.ig import ig
from attribution.sm import sm
from gui.gui import Gui
from opts import opts
import architecture as arch

class Intevis:
    def __init__(self, seed=42):
        """Менеджер программного продукта.

        Args:
            seed (int): опциональный параметр для генератора случайных чисел.

        """
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
        """Загрузка из сети Интернет описания imagenet классов."""
        classes = ''
        for f in urllib.request.urlopen(self.opts.url.imagenet_classes):
            classes = classes + f.decode('utf-8')
        self.classes = yaml.safe_load(classes)

    def gui(self, width='1200px', height='600px'):
        """Запуск графического пользовательского интерфейса.

        Args:
            width (str): ширина элемента (например, "1200px").
            height (str): высота элемента (например, "600px").

        """
        self.opts.app_width = width
        self.opts.app_height = height
        Gui(self, self.opts)

    def plot(self, fpath=None):
        """Построение (графика) входного изображения.

        Args:
            fpath (str): путь к файлу для сохранения изображения. Если не задан,
                то производится непосредственная отрисовка изображения.

        Returns:
            bool: флаг успешности операции (возвращается только при сохранении
                в файл).

        """
        if self.x_raw is None:
            return False

        x = self.x_raw

        return plot(x, fpath)

    def plot_am(self, fpath=None):
        """Построение (графика) результата метода AM.

        Args:
            fpath (str): путь к файлу для сохранения изображения. Если не задан,
                то производится непосредственная отрисовка изображения.

        Returns:
            bool: флаг успешности операции (возвращается только при сохранении
                в файл).

        """
        if self.am is None:
            return False

        x = self.am.cpu()
        x = tensor_to_img(x, sat=0.2, br=0.8)
        x = tensor_to_plot(x)

        return plot(x, fpath)

    def plot_ig(self, fpath=None):
        """Построение (графика) результата метода IG.

        Args:
            fpath (str): путь к файлу для сохранения изображения. Если не задан,
                то производится непосредственная отрисовка изображения.

        Returns:
            bool: флаг успешности операции (возвращается только при сохранении
                в файл).

        """
        if self.ig is None:
            return False

        x = self.ig
        x = np.uint8(x)

        return plot(x, fpath, with_cmap=True)

    def plot_sm(self, fpath=None):
        """Построение (графика) результата метода SM.

        Args:
            fpath (str): путь к файлу для сохранения изображения. Если не задан,
                то производится непосредственная отрисовка изображения.

        Returns:
            bool: флаг успешности операции (возвращается только при сохранении
                в файл).

        """
        if self.sm is None:
            return False

        x = self.sm.cpu()

        return plot(x, fpath, with_cmap=True)

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
        """Запуск метода максимизации активаций."""
        if is_random:
            x = tensor_rand(sz)
        else:
            x = self.x

        self.am = am(self.model, layer, filter, x, self.device, lr, iters)

    def run_ar(self):
        """Запуск метода визуализации архитектуры."""
        self.model.eval()
        graph = arch.build_graph(self.model, (torch.zeros([1, 3, 228, 228]).to(self.device)))
        dot=graph.build_dot()
        dot.attr("graph", rankdir="TD") #Topdown
        # dot.attr("graph", rankdir="LR") #Left-Right
        dot.format = 'png'
        dot.render('./tmp/architecture')
        return

    def run_ig(self, steps):
        """Запуск метода атрибуции IG."""
        x = np.array(self.x_raw)

        self.ig = ig(self.model, x, steps, self.device)

    def run_sm(self):
        """Запуск метода атрибуции SM."""
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
            self.model.to(self.device)
        else:
            self.model = model

        self.model_name = name

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


def plot(x, fpath=None, with_cmap=False):
    fig = plt.figure(figsize=(6, 6))
    if with_cmap:
        plt.imshow(x, cmap=plt.cm.hot)
    else:
        plt.imshow(x)
    plt.axis('off')

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
        plot_clear()
        return True
    else:
        plt.show()


def plot_clear():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


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
