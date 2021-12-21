import ipywidgets


from .element import Element


class Tabs(Element):
    def __init__(self, items={}, kind='p'):
        """Вкладки с элементами, элемент пользовательского интерфейса.

        Args:
            items (dict): словарь с именованными элементами, входящими в
                таблицу.
            kind (str): тип элемента ('p' - 'primary', 's' - 'secondary',
                't' - tertiary, 'a' - 'accent', 'w' - warning), используется для
                выбора способа стилизации.

        """
        self.items = items
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        """Метод осуществляет непосредственно построение элемента.

        Returns:
            Table: текущий экземпляр класса.

        """
        self.wgt = ipywidgets.Tab(layout=self.layout())

        self.wgt.children = [item.wgt for item in self.items.values()]
        for i, name in enumerate(list(self.items.keys())):
            self.wgt.set_title(i, name)

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-tabs')

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
