from IPython.display import display
import ipywidgets


from .element import Element


class Output(Element):
    def add(self, item):
        """Метод для добавления элемента в контейнер.

        Args:
            item (Element): добавляемый элемент.

        """
        with self.wgt:
            display(item.wgt)

    def add_text(self, text):
        """Метод для добавления текста в контейнер.

        Args:
            text (str): добавляемый текст.

        """
        self.wgt.append_stdout(text + '\n')

    def build(self):
        """Метод осуществляет непосредственно построение элемента.

        Returns:
            Output: текущий экземпляр класса.

        """
        self.wgt = ipywidgets.Output(layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-output')

        # self.wgt.layout.border = '1px solid green' # DEBUG
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
            width                   = self.w,
            height                  = self.h,
            min_width               = '50px',
            padding                 = '0px',
            overflow                = 'scroll',
        )

    def set(self, w='100%', h='auto'):
        """Метод для задания дополнительных свойств элемента.

        Args:
            w (str): ширина элемента (например, "100px" или "auto").
            h (str): высота элемента (например, "100px" или "auto").

        Returns:
            Output: текущий экземпляр класса.

        """
        self.w = w
        self.h = h

        return self
