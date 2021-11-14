from elements.button import Button
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
        self.cont.add_text('TODO')

    def build_panel(self):
        self.btn_run = Button(self.on_run, 'Запустить')

        self.panel = Panel({}, self.btn_run)

    def clear(self):
        self.cont.clear()

    def on_run(self):
        self.clear()

        data = {}

        self.run(data)
