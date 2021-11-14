import ipywidgets


from element import Element


class Dropdown(Element):
    def __init__(self, values=[], kind='p'):
        self.values = values
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        self.wgt = ipywidgets.Dropdown(
            options=self.values,
            value=self.values[0] if len(self.values) > 0 else None,
            layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-dropdown')

        return self

    def layout(self):
        return ipywidgets.Layout(
            margin                  = '0px',
            width                   = self.w,
            height                  = self.h,
            min_width               = '50px',
            padding                 = '0px',
        )

    def set(self, w='auto', h='auto'):
        self.w = w
        self.h = h

        return self
