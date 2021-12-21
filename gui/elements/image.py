import ipywidgets


from .element import Element


class Image(Element):
    def __init__(self, fpath, kind='p'):
        """Изображение, элемент пользовательского интерфейса.

        Args:
            fpath (str): путь к файлу с изображением.
            kind (str): тип элемента ('p' - 'primary', 's' - 'secondary',
                't' - tertiary, 'a' - 'accent', 'w' - warning), используется для
                выбора способа стилизации.

        """
        self.fpath = fpath
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        """Метод осуществляет непосредственно построение элемента.

        Returns:
            Image: текущий экземпляр класса.

        """
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
        """Метод возвращает лейаут (разметку) для элемента.

        Returns:
            ipywidgets.Layout: разметка для элемента.

        """
        return ipywidgets.Layout(
            margin                  = '0px',
            width                   = self.w,
            height                  = self.h,
            min_width               = '50px',
            padding                 = '0px',
            object_fit              = 'cover',
        )

    def set(self, w='auto', h='100%'):
        """Метод для задания дополнительных свойств элемента.

        Args:
            w (str): ширина элемента (например, "100px" или "auto").
            h (str): высота элемента (например, "100px" или "auto").

        Returns:
            Image: текущий экземпляр класса.

        """
        self.w = w
        self.h = h

        return self
