import ipywidgets


from element import Element


class Panel(Element):
    def __init__(self, items={}, button=None, kind='p'):
        self.items = items
        self.button = button
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        children = []
        for i, [name, item] in enumerate(self.items.items(), 1):
            text_wgt = ipywidgets.Label(value=name)
            text_wgt.layout.grid_area = f'lbl{i}'
            children.append(text_wgt)

            item.wgt.layout.grid_area = f'val{i}'
            children.append(item.wgt)

        if self.button:
            self.button.wgt.layout.grid_area = 'btn'
            self.button.wgt.layout.align_self = 'flex-end'
            children.append(self.button.wgt)

        self.wgt = ipywidgets.GridBox(children=children, layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-panel')

        return self

    def layout(self):
        n = len(self.items.keys())
        gtc = '4fr 5fr'
        gtr = 'min-content ' * n
        if self.button:
            gtr += '1fr '
        gta = ''
        for i in range(1, n+1):
            gta += f' "lbl{i} val{i}"'
        if self.button:
            gta += ' "btn btn"'

        return ipywidgets.Layout(
            margin                  = '0px',
            width                   = self.w,
            height                  = self.h,
            min_width               = '50px',
            padding                 = '15px',
            display                 = 'grid',
            grid_gap                = '15px 15px',
            grid_template_columns   = gtc,
            grid_template_rows      = gtr,
            grid_template_areas     = gta
        )

    def set(self, w='350px', h='auto'):
        self.w = w
        self.h = h

        return self
