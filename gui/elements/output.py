from IPython.display import display
import ipywidgets


from element import Element


class Output(Element):
    def add(self, item):
        with self.wgt:
            display(item.wgt)

    def add_text(self, text):
        self.wgt.append_stdout(text + '\n')

    def build(self):
        self.wgt = ipywidgets.Output(layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-output')

        # self.wgt.layout.border = '1px solid green' # DEBUG
        return self

    def clear(self):
        self.wgt.clear_output()

    def layout(self):
        return ipywidgets.Layout(
            margin                  = '0px',
            width                   = self.w,
            height                  = self.h,
            min_width               = '50px',
            padding                 = '0px',
            overflow                = 'scroll',
        )

    def set(self, w='100%', h='auto'):
        self.w = w
        self.h = h

        return self
