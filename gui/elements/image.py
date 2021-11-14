import ipywidgets


from element import Element


class Image(Element):
    def __init__(self, fpath, kind='p'):
        self.fpath = fpath
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        with open(self.fpath, 'rb') as f:
            image = f.read()

        self.wgt = ipywidgets.Image(
            value=image,
            format='png',
            layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-image')

        return self

    def layout(self):
        return ipywidgets.Layout(
            margin                  = '0px',
            width                   = self.w,
            height                  = self.h,
            min_width               = '50px',
            padding                 = '0px',
            object_fit              = 'cover',
        )

    def set(self, w='auto', h='100%'):
        self.w = w
        self.h = h

        return self
