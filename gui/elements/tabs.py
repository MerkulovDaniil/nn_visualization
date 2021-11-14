import ipywidgets


from element import Element


class Tabs(Element):
    def __init__(self, items={}, kind='p'):
        self.items = items
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        self.wgt = ipywidgets.Tab(layout=self.layout())

        self.wgt.children = [item.wgt for item in self.items.values()]
        for i, name in enumerate(list(self.items.keys())):
            self.wgt.set_title(i, name)

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-tabs')

        return self

    def layout(self):
        return ipywidgets.Layout(
            margin                  = '0px',
            width                   = self.w,
            height                  = self.h,
            min_width               = '50px',
            padding                 = '0px',
        )
