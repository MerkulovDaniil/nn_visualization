import ipywidgets


from .element import Element


class Text(Element):
    def __init__(self, text, kind='p'):
        """Отображаемый текст, элемент пользовательского интерфейса.

        Args:
            text (str): текст для отображения.
            kind (str): тип элемента ('p' - 'primary', 's' - 'secondary',
                't' - tertiary, 'a' - 'accent', 'w' - warning), используется для
                выбора способа стилизации.

        """
        self.text = text
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        """Метод осуществляет непосредственно построение элемента.

        Returns:
            Text: текущий экземпляр класса.

        """
        self.wgt = ipywidgets.Label(
            value=self.text,
            layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-text')

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
        )
