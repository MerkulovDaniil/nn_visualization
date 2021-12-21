import ipywidgets


from .element import Element


class Input(Element):
    def __init__(self, v=None, kind='p'):
        """Поле ввода str, элемент пользовательского интерфейса.

        Args:
            v (int): начальное значение.
            kind (str): тип элемента ('p' - 'primary', 's' - 'secondary',
                't' - tertiary, 'a' - 'accent', 'w' - warning), используется для
                выбора способа стилизации.

        """
        self.v = v
        self.kind = kind

        self.value = v

        self.set()
        self.build()

    def build(self):
        """Метод осуществляет непосредственно построение элемента.

        Returns:
            Input: текущий экземпляр класса.

        """
        self.wgt = ipywidgets.Text(layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-input')

        self.wgt.observe(self.on_change, names='value')

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

    def on_change(self, change):
        """Метод вызывается при изменении пользователем содержимого элемента."""
        self.value = change['new']
