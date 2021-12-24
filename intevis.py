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
from architecture.canvas import arch
from attribution.ig import ig
from attribution.sm import sm
from attribution.sc import sc
from gui.gui import Gui
from opts import opts


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

        # Результаты работы средств анализа ИНС:
        self.am = None
        self.ig = None
        self.sm = None
        self.sc = None

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

    def plot_sc(self, fpath=None):
        """Построение (графика) результата метода Score-CAM.

        Args:
            fpath (str): путь к файлу для сохранения изображения. Если не задан,
                то производится непосредственная отрисовка изображения.

        Returns:
            bool: флаг успешности операции (возвращается только при сохранении
                в файл).

        """
        if self.sc is None:
            return False

        x = self.sc.cpu().squeeze()
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
        self.model.eval()

        if is_random:
            x = tensor_rand(sz)
        else:
            x = self.x

        self.am = am(self.model, layer, filter, x, self.device, lr, iters)

    def run_ar(self, dir=None):
        """Запуск метода визуализации архитектуры."""
        self.model.eval()

        x0 = torch.zeros([1, 3, 228, 228]).to(self.device)
        is_hor = dir == 'Горизонтально'
        arch(self.model, x0, './tmp/architecture', 'png', is_hor)

    def run_ig(self, steps):
        """Запуск метода атрибуции IG."""
        self.model.eval()

        x = np.array(self.x_raw)

        self.ig = ig(self.model, x, steps, self.device)

    def run_sm(self):
        """Запуск метода атрибуции SM."""
        self.model.eval()

        self.sm = sm(self.model, self.x)

    def run_sc(self):
        """Запуск метода атрибуции Score-CAM"""
        self.model.eval()

        # TODO: сделать универсальный выбор слоя. Сейчас подразумевается, что первый блок это backbone или features и берется его выход
        # ['vgg13', 'vgg16', 'vgg19', 'resnet18', 'own']
        if self.model_name in ['vgg13', 'vgg16', 'vgg19']:
            target_layer = list(self.model.children())[0]
        elif self.model_name in ['resnet18']:
            target_layer = list(self.model.children())[-3]
        else:
            try:
                target_layer = list(self.model.children())[0][-1]
            except:
                target_layer = list(self.model.children())[-3]

        self.sc = sc(self.model, target_layer, self.x, None, self.device)

    def set_image(self, data=None, link=None):
        """Задание входного изображения.

        Args:
            data (byte): непосредственно изображение в raw формате.
            link (str): url-адрес изображения.

        Note:
            Должен быть задан только из аргументов data, link. Если заданы оба
            аргумента, то будет использован data.

        """
        if data is not None:
            self.x_raw = Image.open(BytesIO(data))
        elif link is not None:
            response = requests.get(link)
            self.x_raw = Image.open(BytesIO(response.content))

        self.x = self.preprocess(self.x_raw)

        self.x = self.x.to(self.device)

    def set_model(self, model=None, name=None):
        """Задание модели ИНС."""
        if model is None:
            self.model = torch.hub.load('pytorch/vision', name, pretrained=True)
        else:
            self.model = model

        self.model_name = name
        self.model.to(self.device)


def img_to_tensor(img, sz=224):
    """Преобразование PIL изображения в тензор.

    Note:
        Возвращает случайный тензор формы [1, number of channels, height,
        width].

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
    """Создание случайного PIL изображения."""
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
    """Создание случайного PIL изображения и трансформация в тензор."""
    return img_to_tensor(img_rand(sz))


def tensor_to_img(x, vmin=0., vmax=1., sat=0.1, br=0.5):
    """Нормализация тензора со значениями между [vmin, vmax]."""
    x = x.detach().cpu()
    m = x.mean()
    s = x.std() or 1.E-8
    return x.sub(m).div(s).mul(sat).add(br).clamp(vmin, vmax)


def tensor_to_plot(x):
    """Подготовка тензора для отрисовки как изобаражения.

    Note:
        Возвращает тензор формы [number of channels, height, width].

    """
    y = x.clone().squeeze(0) if len(x.shape) == 4 else x.clone()
    y = y.squeeze(0) if y.shape[0] == 1 else y.permute(1, 2, 0)
    return y.detach()
