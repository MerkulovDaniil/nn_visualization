from IPython.display import display
import ipywidgets


from element import Element


class Html(Element):
    def __init__(self, content, kind='p'):
        self.content = content
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        self.wgt = ipywidgets.HTML(value=self.content, layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-html')

        return self

    def clear(self):
        self.wgt.clear_output()

    def layout(self):
        return ipywidgets.Layout(
            margin                  = '0px',
            padding                 = '0px',
        )

    def set(self):
        return self
