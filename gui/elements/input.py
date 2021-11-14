import ipywidgets


from element import Element


class Input(Element):
    def __init__(self, v=None, kind='p'):
        self.v = v
        self.kind = kind

        self.value = v

        self.set()
        self.build()

    def build(self):
        self.wgt = ipywidgets.Text(layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-input')

        self.wgt.observe(self.on_change, names='value')

        return self

    def layout(self):
        return ipywidgets.Layout(
            margin                  = '0px',
            width                   = self.w,
            height                  = self.h,
            min_width               = '50px',
            padding                 = '0px',
        )

    def on_change(self, change):
        self.value = change['new']
