import ipywidgets


from element import Element


class Window(Element):
    def __init__(self, panel, conts, kind='p'):
        self.panel = panel
        self.conts = conts
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        self.wgt = ipywidgets.GridBox(layout=self.layout())

        self.panel.wgt.layout.grid_area = 'panel'
        children = [self.panel.wgt]

        for i, cont in enumerate(self.conts, 1):
            cont.wgt.layout.grid_area = f'cont{i}'
            children.append(cont.wgt)

        self.wgt.children = children

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-window')

        # self.wgt.layout.border = '1px solid blue' # DEBUG
        return self

    def layout(self):
        if len(self.conts) == 1:
            gtc = '350px 1fr'
            gtr = '1fr'
            gta = '''
                "panel cont1"
            '''
        elif len(self.conts) == 2:
            gtc = '350px 1fr 1fr'
            gtr = '1fr'
            gta = '''
                "panel cont1 cont2"
            '''
        elif len(self.conts) == 3:
            gtc = '350px 1fr 1fr'
            gtr = '1fr 1fr'
            gta = '''
                "panel cont1 cont2"
                "panel cont3 cont3"
            '''
        elif len(self.conts) == 4:
            gtc = '350px 1fr 1fr'
            gtr = '1fr 1fr'
            gta = '''
                "panel cont1 cont2"
                "panel cont3 cont4"
            '''
        else:
            raise ValueError('Слишком много элементов')

        return ipywidgets.Layout(
            margin                  = '0px',
            width                   = self.w,
            height                  = self.h,
            padding                 = '15px',
            display                 = 'grid',
            #justify_items           = 'stretch',
            #align_content           = 'stretch',
            #justify_content         = 'stretch',
            grid_gap                = '15px 15px',
            grid_template_columns   = gtc,
            grid_template_rows      = gtr,
            grid_template_areas     = gta
        )

    def set(self, w='100%', h='100%'):
        self.w = w
        self.h = h

        return self
