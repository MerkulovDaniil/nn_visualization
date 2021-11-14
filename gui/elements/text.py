import ipywidgets


from element import Element


class Text(Element):
    def __init__(self, text, kind='p'):
        self.text = text
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        self.wgt = ipywidgets.Label(
            value=self.text,
            layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-text')

        return self

    def layout(self):
        return ipywidgets.Layout(
            margin                  = '0px',
            width                   = self.w,
            height                  = self.h,
            min_width               = '50px',
            padding                 = '0px',
        )
