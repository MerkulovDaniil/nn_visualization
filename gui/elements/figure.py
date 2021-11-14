import ipywidgets


from element import Element
from image import Image
from text import Text


class Figure(Element):
    def __init__(self, fpath, title, kind='p'):
        self.fpath = fpath
        self.title = title
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        self.label = Text(self.title)
        self.label.wgt.add_class('e-figure__label')

        self.image = Image(self.fpath).set(h='250px').build()
        self.image.wgt.add_class('e-figure__image')

        children = [self.label.wgt, self.image.wgt]
        self.wgt = ipywidgets.GridBox(children=children, layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-figure')

        return self

    def layout(self):
        return ipywidgets.Layout(
            margin                  = '0px',
            width                   = self.w,
            height                  = self.h,
            padding                 = '1px',
            display                 = 'grid',
            grid_gap                = '2px 2px',
            grid_template_columns   = '1fr',
            grid_template_rows      = 'min-content min-content',
            justify_items           = 'center',
        )

    def set(self, w='100%', h='300px'):
        self.w = w
        self.h = h

        return self
