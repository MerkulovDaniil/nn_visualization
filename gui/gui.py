from IPython.display import HTML
from IPython.display import display
import os
import random


from .elements.tabs import Tabs


from .components.ac import Ac
from .components.ar import Ar
from .components.at import At
from .components.cs import Cs
from .components.dg import Dg
from .components.md import Md


class Gui:
    def __init__(self, iv, opts):
        """Менеджер пользовательского интерфейса.

        Args:
            iv (Intevis): экземпляр класса менеджера платформы.
            opts (dict): словарь с глобальными опциями.

        """
        self.iv = iv
        self.opts = opts

        self.build_css()
        self.build()

    def build(self):
        self.md = Md(self.opts, self.run_md) # Вкладка "Модель"
        self.ar = Ar(self.opts, self.run_ar) # Вкладка "Архитектура"
        self.ac = Ac(self.opts, self.run_ac) # Вкладка "Активация"
        self.at = At(self.opts, self.run_at) # Вкладка "Атрибуция"
        self.cs = Cs(self.opts, self.run_cs) # Вкладка "Консоль"
        self.dg = Dg(self.opts, self.run_dg) # Вкладка "DEBUG"

        self.tabs = Tabs({
            'Модель': self.md.wgt,
            'Архитектура': self.ar.wgt,
            'Активация': self.ac.wgt,
            'Атрибуция': self.at.wgt,
            'Консоль': self.cs.wgt,
            'DEBUG': self.dg.wgt,
        })
        self.tabs.set(w=self.opts.app_width).build()

        display(self.tabs.wgt)
        self.cs.log(f'Система запущена ({self.cs.get_date()})')

        return self

    def build_css(self):
        fpath = os.path.dirname(__file__)
        fpath = os.path.join(fpath, './style.css')
        with open(fpath, 'r') as f:
            css = f.read()
        css = f'<style>{css}</style>'
        display(HTML(css))

    def run_ac(self, data):
        """Запуск метода максимизации активаций."""
        with self.dg.log:
            for i, f in enumerate(data['filters']):
                self.cs.log(f'Построение Act. Max. для фильтра {f}', 'prc')
                self.iv.run_am(data['layer'], f, data['lr'], data['iters'],
                    data['is_random'])
                self.cs.log(f'Act. Max. для фильтра {f} построена', 'res')

                self.cs.log(f'Отрисовка Act. Max. для фильтра {f}', 'prc')
                image_file = random_image_path()
                if not self.iv.plot_am(image_file):
                    self.cs.log('Не удалось нарисовать Act. Max. для фильтра {f}',
                        'err')
                    return
                self.ac.set_image(i, image_file)
                self.cs.log(f'Act. Max. для фильтра {f} нарисована', 'res')

    def run_at(self, data):
        """Запуск метода атрибуции."""
        with self.dg.log:
            self.cs.log('Отрисовка исходного изображения', 'prc')
            image_file = random_image_path()
            if not self.iv.plot(image_file):
                self.cs.log('Не удалось нарисовать исходное изображение', 'err')
                return
            self.at.set_image_bs(image_file)
            self.cs.log('Иходное изображение нарисовано', 'res')

            self.cs.log('Построение Saliency Map', 'prc')
            self.iv.run_sm()
            self.cs.log('Saliency Map построена', 'res')

            self.cs.log('Отрисовка Saliency Map', 'prc')
            image_file = random_image_path()
            if not self.iv.plot_sm(image_file):
                self.cs.log('Не удалось нарисовать Saliency Map', 'err')
                return
            self.at.set_image_sm(image_file)
            self.cs.log('Saliency Map нарисована', 'res')

            self.cs.log('Построение Integrated Gradients', 'prc')
            self.iv.run_ig(data['steps'])
            self.cs.log('Integrated Gradients построен', 'res')

            self.cs.log('Отрисовка Integrated Gradients', 'prc')
            image_file = random_image_path()
            if not self.iv.plot_ig(image_file):
                self.cs.log('Не удалось нарисовать Integrated Gradients', 'err')
                return
            self.at.set_image_ig(image_file)
            self.cs.log('Integrated Gradients нарисован', 'res')

            self.cs.log('Построение Score-CAM', 'prc')
            self.iv.run_sc()
            self.cs.log('Score-CAM построена', 'res')

            self.cs.log('Отрисовка Score-CAM', 'prc')
            image_file = random_image_path()
            if not self.iv.plot_sc(image_file):
                self.cs.log('Не удалось нарисовать Score-CAM', 'err')
                return
            self.at.set_image_sc(image_file)
            self.cs.log('Score-CAM нарисована', 'res')

    def run_ar(self, data):
        """Запуск метода построения архитектуры."""
        with self.dg.log:
            self.cs.log('Построение вычислительного графа', 'prc')
            self.iv.run_ar(data['dir'])
            self.cs.log('Построение вычислительного графа завершено', 'res')
            self.cs.log('Построение архитектуры', 'prc')
            self.ar.set_image("./tmp/architecture.png")
            self.cs.log('Построение архитектуры завершено', 'res')
            return

    def run_cs(self, data):
        """Запуск метода вывода в консоль."""
        return

    def run_dg(self, data):
        """Запуск метода вывода в 'debug'."""
        return

    def run_md(self, data):
        """Запуск метода подготовки модели."""
        with self.dg.log:
            self.cs.log('Обработка модели и данных', 'prc')

            if data['data'] == 'imagenet':
                self.cs.log('Загрузка меток классов', 'prc')
                self.iv.download_imagenet_classes()
                self.cs.log('Метки классов загружены', 'res')
            else:
                self.cs.log('Неизвестный набор данных', 'ftl')
                return

            if data['model'] == self.opts.model_list[-1]:
                # self.iv.set_model(name=data['model'])
                # self.cs.log('Собственные модели пока что не поддерживаются', 'ftl')
                self.cs.log('Загрузка модели из файла', 'prc')
                try:
                    from custom_models.custom_model import custom_model
                    import torch
                    if torch.cuda.is_available():
                        custom_model.cuda()
                    self.iv.set_model(model=custom_model, name=data['model'])
                except:
                    self.cs.log('Неудачная загрузка модели', 'err')
                    return
            else:
                self.cs.log('Загрузка модели', 'prc')
                self.iv.set_model(name=data['model'])
                self.cs.log('Модель загружена', 'res')

            self.cs.log('Загрузка изображения', 'prc')
            if data['image_link'] and data['image_file']:
                self.cs.log('Заданы и ссылка, и файл изображения', 'wrn')
            if not data['image_link'] and not data['image_file']:
                self.cs.log('Изображение не задано', 'err')
                return
            self.iv.set_image(data['image_file'], data['image_link'])
            if self.iv.x_raw is None:
                self.cs.log('Не удалось загрузить изображение', 'err')
                return
            self.cs.log('Изображение загружено', 'res')

            self.cs.log('Построение предсказания модели', 'prc')
            self.iv.run()
            self.md.set_result(self.iv.y_name)
            self.cs.log('Предсказание модели построено', 'res')

            self.cs.log('Отрисовка исходного изображения', 'prc')
            image_file = random_image_path()
            if not self.iv.plot(image_file):
                self.cs.log('Не удалось нарисовать исходное изображение', 'err')
                return
            self.md.set_image(image_file)
            self.cs.log('Иходное изображение нарисовано', 'res')


def random_image_path(folder='./tmp/'):
    num = int(random.uniform(1, 10000000000))
    return os.path.join(folder, f'{num}.png')
