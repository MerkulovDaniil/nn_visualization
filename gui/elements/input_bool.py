import ipywidgets


from element import Element


class InputBool(Element):
    def __init__(self, v=None, help='', kind='p'):
        self.v = v
        self.help = help
        self.kind = kind

        self.value = v

        self.set()
        self.build()

    def build(self):
        self.wgt = ipywidgets.ToggleButton(
            value=self.v,
            tooltip=self.help,
            icon='check',
            layout=self.layout())

        self.wgt.observe(self.on_change, names='value')

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-input-bool')

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
