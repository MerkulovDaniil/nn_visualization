import ipywidgets


from .element import Element


class InputBool(Element):
    def __init__(self, v=None, help='', kind='p'):
        """Поле выбора bool, элемент пользовательского интерфейса.

        Args:
            v (bool): начальное значение.
            help (str): Текст всплывающей подсказки.
            kind (str): тип элемента ('p' - 'primary', 's' - 'secondary',
                't' - tertiary, 'a' - 'accent', 'w' - warning), используется для
                выбора способа стилизации.

        """
        self.v = v
        self.help = help
        self.kind = kind

        self.value = v

        self.set()
        self.build()

    def build(self):
        """Метод осуществляет непосредственно построение элемента.

        Returns:
            InputBool: текущий экземпляр класса.

        """

        self.wgt = ipywidgets.Checkbox(
            value=self.v,
            #tooltip=self.help,
            #icon='check',
            layout=self.layout())

        self.wgt.observe(self.on_change, names='value')

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-input-bool')

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
