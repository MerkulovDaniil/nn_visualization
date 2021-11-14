import ipywidgets


from element import Element


class Upload(Element):
    def __init__(self, types='.png,.jpg', kind='p'):
        self.types = types
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        self.wgt = ipywidgets.FileUpload(
            accept=self.types,
            multiple=False,
            layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-upload')

        return self

    def layout(self):
        return ipywidgets.Layout(
            margin                  = '0px',
            width                   = self.w,
            height                  = self.h,
            min_width               = '50px',
        )
