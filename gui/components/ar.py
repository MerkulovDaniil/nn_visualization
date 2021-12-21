from elements.button import Button
from elements.figure import Figure
from elements.input_int import InputInt
from elements.output import Output
from elements.panel import Panel
from elements.window import Window

from component import Component


class Ar(Component):
    """Вкладка "Архитектура"."""
    def build(self):
        self.build_content()
        self.build_panel()

        self.wgt = Window(self.panel, [self.cont])
        self.wgt.set(h=self.opts.app_height).build()

        return self

    def build_content(self):
        self.cont = Output()

    def build_panel(self):
        self.btn_run = Button(self.on_run, 'Построить')

        self.panel = Panel({}, self.btn_run)

    def clear(self):
        self.cont.clear()

    def on_run(self):
        self.clear()

        data = {}

        self.run(data)
    
    def set_image(self, fpath="./tmp/architecture.png"):
        figure = Figure(fpath, 'Архитектура нейросети')
        self.cont.add(figure)
