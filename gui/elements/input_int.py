import ipywidgets


from .element import Element


class InputInt(Element):
    def __init__(self, v=None, v_min=None, v_max=None, kind='p'):
        """Поле ввода int, элемент пользовательского интерфейса.

        Args:
            v (int): начальное значение.
            v_min (int): минимальное возможное значение.
            v_max (int): максимальное возможное значение.
            kind (str): тип элемента ('p' - 'primary', 's' - 'secondary',
                't' - tertiary, 'a' - 'accent', 'w' - warning), используется для
                выбора способа стилизации.

        """
        self.v = v
        self.v_min = v_min
        self.v_max = v_max
        self.kind = kind

        self.value = v

        self.set()
        self.build()

    def build(self):
        """Метод осуществляет непосредственно построение элемента.

        Returns:
            InputInt: текущий экземпляр класса.

        """
        self.wgt = ipywidgets.IntSlider(
            value=self.v,
            min=self.v_min,
            max=self.v_max,
            step=1,
            layout=self.layout())

        self.wgt.observe(self.on_change, names='value')

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-input-int')

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
