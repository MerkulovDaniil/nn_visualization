from IPython.display import display
import ipywidgets


from .element import Element


class Html(Element):
    def __init__(self, content, kind='p'):
        """Контейнер с любым содержимым, элемент пользовательского интерфейса.

        Args:
            content (str): произвольное содержимое.
            kind (str): тип элемента ('p' - 'primary', 's' - 'secondary',
                't' - tertiary, 'a' - 'accent', 'w' - warning), используется для
                выбора способа стилизации.

        Note:
            При необходимости содержимое может добавляться в рамках контекстного
            вызова ("with element: print('some new content')").

        """
        self.content = content
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        """Метод осуществляет непосредственно построение элемента.

        Returns:
            Html: текущий экземпляр класса.

        """
        self.wgt = ipywidgets.HTML(value=self.content, layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-html')

        return self

    def clear(self):
        """Метод очищает содержимое элемента."""
        self.wgt.clear_output()

    def layout(self):
        """Метод возвращает лейаут (разметку) для элемента.

        Returns:
            ipywidgets.Layout: разметка для элемента.

        """
        return ipywidgets.Layout(
            margin                  = '0px',
            padding                 = '0px',
        )

    def set(self):
        """Метод для задания дополнительных свойств элемента.

        Returns:
            Html: текущий экземпляр класса.

        """
        return self
