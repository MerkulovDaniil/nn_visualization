from ..elements.button import Button
from ..elements.figure import Figure
from ..elements.input_int import InputInt
from ..elements.output import Output
from ..elements.panel import Panel
from ..elements.window import Window


from .component import Component


class At(Component):
    """Вкладка "Атрибуция"."""
    def build(self):
        self.build_content()
        self.build_panel()

        conts = [self.win_bs, self.win_sm, self.win_ig, self.win_sc]
        self.wgt = Window(self.panel, conts)
        self.wgt.set(h=self.opts.app_height).build()

        return self

    def build_content(self):
        self.win_bs = Output()
        self.win_sm = Output()
        self.win_ig = Output()
        self.win_sc = Output()

    def build_panel(self):
        self.steps = InputInt(v=3, v_min=1, v_max=500)

        self.btn_run = Button(self.on_run, 'Запустить')

        self.panel = Panel({
            'Количество точек': self.steps,
        }, self.btn_run)

    def clear(self):
        self.win_bs.clear()
        self.win_sm.clear()
        self.win_ig.clear()
        self.win_sc.clear()

    def on_run(self):
        self.clear()

        self.run({
            'steps': self.steps.value
        })

    def set_image_bs(self, fpath):
        figure = Figure(fpath, 'Исходное изображение')
        self.win_bs.add(figure)

    def set_image_sm(self, fpath):
        figure = Figure(fpath, 'Saliency Map')
        self.win_sm.add(figure)

    def set_image_ig(self, fpath):
        figure = Figure(fpath, 'Integrated Gradients')
        self.win_ig.add(figure)

    def set_image_sc(self, fpath):
        figure = Figure(fpath, 'Score-CAM')
        self.win_sc.add(figure)
