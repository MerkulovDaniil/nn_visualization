from elements.button import Button
from elements.dropdown import Dropdown
from elements.figure import Figure
from elements.input import Input
from elements.output import Output
from elements.panel import Panel
from elements.table import Table
from elements.text import Text
from elements.upload import Upload
from elements.window import Window


from component import Component


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

        self.data = Dropdown(self.opts.data_list)

        self.image_link = Input()

        self.image_file = Upload()

        self.btn_run = Button(self.on_run, 'Загрузить')

        self.panel = Panel({
            'Нейронная сеть': self.model,
            'Набор данных': self.data,
            'URL изображения': self.image_link,
            'Файл изображения': self.image_file,
        }, self.btn_run)

    def clear(self):
        self.win_image.clear()
        self.win_table.clear()

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
            'Важная инфо 1': Text('Информация'),
            'Важная инфо 2': Text('Информация'),
            'Важная инфо 3': Text('Информация'),
            'Важная инфо 4': Text('Информация'),
            'Важная инфо 5': Text('Информация'),
            'Важная инфо 6': Text('Информация'),
            'Важная инфо 7': Text('Информация'),
        })
        self.win_table.add(table)
