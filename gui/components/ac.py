from ..elements.button import Button
from ..elements.dropdown import Dropdown
from ..elements.figure import Figure
from ..elements.input_bool import InputBool
from ..elements.input_int import InputInt
from ..elements.input_float import InputFloat
from ..elements.output import Output
from ..elements.panel import Panel
from ..elements.window import Window


from .component import Component


class Ac(Component):
    """Вкладка "Активация"."""
    def build(self):
        self.build_content()
        self.build_panel()

        self.wgt = Window(self.panel, self.wins)
        self.wgt.set(h=self.opts.app_height).build()

        self.filters = []

        return self

    def build_content(self):
        self.wins = [Output(), Output(), Output(), Output()]

    def build_panel(self):
        self.layer = InputInt(v=2, v_min=0, v_max=99)
        self.filter1 = InputInt(v=10, v_min=0, v_max=1000)
        self.filter2 = InputInt(v=20, v_min=0, v_max=1000)
        self.filter3 = InputInt(v=30, v_min=0, v_max=1000)
        self.filter4 = InputInt(v=40, v_min=0, v_max=1000)
        self.iters = InputInt(v=5, v_min=1, v_max=100)
        self.lr = InputFloat(v=0.5, v_min=-4, v_max=1.)

        help = 'Если выбрано, то будет использован шум в качестве начального приближения, иначе будет использовано загруженное изображение. По умолчанию не выбрано.'
        # self.image_random = InputBool(False , help)
        self.image_random = Dropdown(['Изображение', 'Шум'])

        self.btn_run = Button(self.on_run, 'Запустить')

        self.panel = Panel({
            'Слой сети': self.layer,
            'Фильтр #1': self.filter1,
            'Фильтр #2': self.filter2,
            'Фильтр #3': self.filter3,
            'Фильтр #4': self.filter4,
            'Итерации': self.iters,
            'Скорость обучения': self.lr,
            'Инициализация': self.image_random,
        }, self.btn_run)

    def clear(self):
        for win in self.wins:
            win.clear()

    def get_filters(self):
        return [
            self.filter1.value,
            self.filter2.value,
            self.filter3.value,
            self.filter4.value]

    def on_run(self):
        self.clear()
        self.filters = self.get_filters()
        self.run({
            'layer': self.layer.value,
            'filters': self.filters,
            'iters': self.iters.value,
            'lr': self.lr.value,
            'is_random': self.image_random.wgt.value == 'Шум',
        })

    def set_image(self, i, fpath):
        label = f'Filter {self.filters[i]}'
        figure = Figure(fpath, label)
        self.wins[i].add(figure)
