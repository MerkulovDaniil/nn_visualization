import ipywidgets


from .element import Element


class Dropdown(Element):
    def __init__(self, values=[], kind='p'):
        """Раскрывающееся меню, элемент пользовательского интерфейса.

        Args:
            values (list): возможные опции для выбора. Каждый элемент списка
                должен быть строкой.
            kind (str): тип элемента ('p' - 'primary', 's' - 'secondary',
                't' - tertiary, 'a' - 'accent', 'w' - warning), используется для
                выбора способа стилизации.

        """
        self.values = values
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        """Метод осуществляет непосредственно построение элемента.

        Returns:
            Dropdown: текущий экземпляр класса.

        """
        self.wgt = ipywidgets.Dropdown(
            options=self.values,
            value=self.values[0] if len(self.values) > 0 else None,
            layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-dropdown')

        return self
