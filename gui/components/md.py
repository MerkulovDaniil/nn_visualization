from ..elements.button import Button
from ..elements.dropdown import Dropdown
from ..elements.figure import Figure
from ..elements.input import Input
from ..elements.output import Output
from ..elements.panel import Panel
from ..elements.table import Table
from ..elements.text import Text
from ..elements.upload import Upload
from ..elements.window import Window


from .component import Component


class Md(Component):
    """Вкладка "Модель"."""
    def build(self):
        self.build_content()
        self.build_panel()

        conts = [self.win_image, self.win_table]
        self.wgt = Window(self.panel, conts)
        self.wgt.set(h=self.opts.app_height).build()
        return self

    def build_content(self):
        self.win_image = Output()
        self.win_table = Output()

    def build_panel(self):
        self.model = Dropdown(self.opts.model_list)

        self.model_file = Upload(types='.py')

        self.weights_files = Upload(types='*', multiple=True)

        self.btn_delete_model = Button(self.on_delete_model, 'Удалить модель и данные') #TODO - добавить кнопку почистить

        self.data = Dropdown(self.opts.data_list)

        self.image_link = Input()

        self.image_file = Upload()

        self.btn_run = Button(self.on_run, 'Загрузить')

        self.panel = Panel({
            'Нейронная сеть': self.model,
            'Файл модели .py': self.model_file,
            'Файлы модели доп.': self.weights_files,
            'Набор данных': self.data,
            'URL изображения': self.image_link,
            'Файл изображения': self.image_file,
        }, self.btn_run)

    def clear(self):
        self.win_image.clear()
        self.win_table.clear()

    def on_delete_model(self):
        import os
        import glob
        files = glob.glob('custom_models/*')
        for f in files:
            os.remove(f)

    def on_run(self):
        self.clear()

        image_file = (self.image_file.wgt.value or {}).values()
        image_name = None
        image_type = None
        if len(image_file) > 0:
            image_file = next(iter(image_file))
            image_name = image_file.get('metadata', {}).get('name')
            image_type = image_file.get('metadata', {}).get('type')
            image_file = image_file.get('content')
        else:
            image_file = None

        model_file = (self.model_file.wgt.value or {}).values()
        if len(model_file) > 0:
            model_file = next(iter(model_file))
            with open("custom_models/custom_model.py", "wb+") as fp:
                fp.write(model_file['content'])

        weights_files = self.weights_files.wgt.value
        # print(weights_files)
        for weights_name, weights_file in (weights_files or {}).items():
            print(weights_name)
            with open(f"custom_models/{weights_name}", "wb+") as fp:
                fp.write(weights_file['content'])

        data = {
            'data': self.data.wgt.value,
            'model': self.model.wgt.value,
            'image_link': self.image_link.wgt.value,
            'image_file': image_file,
            'image_name': image_name,
            'image_type': image_type,
        }

        self.run(data)

    def set_image(self, fpath):
        figure = Figure(fpath, 'Загруженное изображение')
        self.win_image.add(figure)

    def set_result(self, pred):
        table = Table({
            'Предсказание': Text(pred),
            #'Важная инфо 1': Text('Информация'),
            #'Важная инфо 2': Text('Информация'),
            #'Важная инфо 3': Text('Информация'),
            #'Важная инфо 4': Text('Информация'),
            #'Важная инфо 5': Text('Информация'),
            #'Важная инфо 6': Text('Информация'),
            #'Важная инфо 7': Text('Информация'),
        })
        self.win_table.add(table)
