import ipywidgets


from .element import Element


class Panel(Element):
    def __init__(self, items={}, button=None, kind='p'):
        """Упорядоченный набор элементов, элемент пользовательского интерфейса.

        Args:
            items (dict): словарь с именованными элементами, входящими в панель.
            button (Element): опциональная кнопка, добавляемая в конец панели.
            kind (str): тип элемента ('p' - 'primary', 's' - 'secondary',
                't' - tertiary, 'a' - 'accent', 'w' - warning), используется для
                выбора способа стилизации.

        """
        self.items = items
        self.button = button
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        """Метод осуществляет непосредственно построение элемента.

        Returns:
            Panel: текущий экземпляр класса.

        """
        children = []
        for i, [name, item] in enumerate(self.items.items(), 1):
            text_wgt = ipywidgets.Label(value=name)
            text_wgt.layout.grid_area = f'lbl{i}'
            children.append(text_wgt)

            item.wgt.layout.grid_area = f'val{i}'
            children.append(item.wgt)

        if self.button:
            self.button.wgt.layout.grid_area = 'btn'
            self.button.wgt.layout.align_self = 'flex-end'
            children.append(self.button.wgt)

        self.wgt = ipywidgets.GridBox(children=children, layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-panel')

        return self

    def layout(self):
        """Метод возвращает лейаут (разметку) для элемента.

        Returns:
            ipywidgets.Layout: разметка для элемента.

        """
        n = len(self.items.keys())
        gtc = '4fr 5fr'
        gtr = 'min-content ' * n
        if self.button:
            gtr += '1fr '
        gta = ''
        for i in range(1, n+1):
            gta += f' "lbl{i} val{i}"'
        if self.button:
            gta += ' "btn btn"'

        return ipywidgets.Layout(
            margin                  = '0px',
            width                   = self.w,
            height                  = self.h,
            min_width               = '50px',
            padding                 = '15px',
            display                 = 'grid',
            grid_gap                = '15px 15px',
            grid_template_columns   = gtc,
            grid_template_rows      = gtr,
            grid_template_areas     = gta
        )

    def set(self, w='350px', h='auto'):
        """Метод для задания дополнительных свойств элемента.

        Args:
            w (str): ширина элемента (например, "100px" или "auto").
            h (str): высота элемента (например, "100px" или "auto").

        Returns:
            Panel: текущий экземпляр класса.

        """
        self.w = w
        self.h = h

        return self
