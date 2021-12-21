import ipywidgets


from .element import Element


class InputFloat(Element):
    def __init__(self, v=None, v_min=None, v_max=None, kind='p'):
        """Поле ввода float, элемент пользовательского интерфейса.

        Args:
            v (float): начальное значение.
            v_min (float): Минимальное возможное значение.
            v_max (float): Максимальное возможное значение.
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
            InputFloat: текущий экземпляр класса.

        """
        self.wgt = ipywidgets.FloatLogSlider(
            value=self.v,
            base=10,
            min=self.v_min,
            max=self.v_max,
            step=0.1,
            layout=self.layout())

        self.wgt.observe(self.on_change, names='value')

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-input-float')

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
