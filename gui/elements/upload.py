import ipywidgets


from .element import Element


class Upload(Element):
    def __init__(self, types='.png,.jpg', kind='p', multiple=False):
        """Загрузка файла, элемент пользовательского интерфейса.

        Args:
            types (str): поддерживаемые типы файлов.
            kind (str): тип элемента ('p' - 'primary', 's' - 'secondary',
                't' - tertiary, 'a' - 'accent', 'w' - warning), используется для
                выбора способа стилизации.
            multiple (bool): если True, то поддерживается загрузка сразу
                нескольких файлов (в текущей версии не поддерживается).

        """
        self.types = types
        self.kind = kind

        self.set()
        self.build(multiple)

    def build(self, multiple=False):
        """Метод осуществляет непосредственно построение элемента.

        Returns:
            Upload: текущий экземпляр класса.

        """
        self.wgt = ipywidgets.FileUpload(
            accept=self.types,
            multiple=multiple,
            layout=self.layout())

        self.wgt.add_class('e-element')
        self.wgt.add_class('e-upload')

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
        )
