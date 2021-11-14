import ipywidgets


from element import Element
from text import Text


class Table(Element):
    def __init__(self, items={}, kind='p'):
        self.items = items
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        children = []
        for i, [name, item] in enumerate(self.items.items(), 1):
            name = Text(name)
            name.wgt.layout.grid_area = f'lbl{i}'
            name.wgt.add_class('e-table__name')
            children.append(name.wgt)

            item.wgt.layout.grid_area = f'val{i}'
            item.wgt.add_class('e-table__text')
            children.append(item.wgt)

        self.wgt = ipywidgets.GridBox(children=children, layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-table')

        return self

    def layout(self):
        n = len(self.items.keys())
        gtc = '5fr 5fr'
        gtr = 'min-content ' * n
        gta = ''
        for i in range(1, n+1):
            gta += f' "lbl{i} val{i}"'

        return ipywidgets.Layout(
            margin                  = '0px',
            width                   = self.w,
            height                  = self.h,
            min_width               = '50px',
            padding                 = '15px',
            display                 = 'grid',
            grid_gap                = '0px 5px',
            grid_template_columns   = gtc,
            grid_template_rows      = gtr,
            grid_template_areas     = gta
        )

    def set(self, w='350px', h='auto'):
        self.w = w
        self.h = h

        return self
