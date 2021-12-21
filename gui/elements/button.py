import ipywidgets


from .element import Element


class Button(Element):
    def __init__(self, callback=None, text='', help='', kind='p'):
        """Кнопка, элемент пользовательского интерфейса.

        Args:
            callback (func): функция (без аргументов) вызывается при клике на
                кнопку.
            text (str): отображаемый текст на кнопке.
            help (str): текст всплывающей подсказки.
            kind (str): тип элемента ('p' - 'primary', 's' - 'secondary',
                't' - tertiary, 'a' - 'accent', 'w' - warning), используется для
                выбора способа стилизации.

        """
        self.callback = callback
        self.text = text
        self.help = help
        self.kind = kind

        self.set()
        self.build()

    def _on_click(self, b):
        """Метод вызывается при клике на кнопку."""
        if self.callback:
            self.callback()

    def build(self):
        """Метод осуществляет непосредственно построение элемента.

        Returns:
            Button: текущий экземпляр класса.

        """
        self.wgt = ipywidgets.Button(
            description=self.text,
            tooltip=self.help,
            button_style='warning' if self.kind == 'p' else 'primary',
            layout=self.layout())

        self.wgt.on_click(self._on_click)

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-button')

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
