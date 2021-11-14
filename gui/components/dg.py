from component import Component


from elements.output import Output


class Dg(Component):
    """Вкладка "DEBUG"."""
    def build(self):
        self.wgt = Output()
        self.wgt.set(h=self.opts.app_height).build()
        
        self.log = self.wgt.wgt

        return self
