import ipywidgets


from element import Element


class InputInt(Element):
    def __init__(self, v=None, v_min=None, v_max=None, kind='p'):
        self.v = v
        self.v_min = v_min
        self.v_max = v_max
        self.kind = kind

        self.value = v

        self.set()
        self.build()

    def build(self):
        self.wgt = ipywidgets.IntSlider(
            value=self.v,
            min=self.v_min,
            max=self.v_max,
            step=1,
            layout=self.layout())

        self.wgt.observe(self.on_change, names='value')

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-input-int')

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
