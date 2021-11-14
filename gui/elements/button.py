import ipywidgets


from element import Element


class Button(Element):
    def __init__(self, callback=None, text='', help='', kind='p'):
        self.callback = callback
        self.text = text
        self.help = help
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        self.wgt = ipywidgets.Button(
            description=self.text,
            tooltip=self.help,
            button_style='warning' if self.kind == 'p' else 'primary',
            layout=self.layout())

        self.wgt.on_click(self.on_click)

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-button')

        return self

    def layout(self):
        return ipywidgets.Layout(
            margin                  = '0px',
            width                   = self.w,
            height                  = self.h,
            min_width               = '50px',
            padding                 = '0px',
        )

    def on_click(self, b):
        if self.callback:
            self.callback()
