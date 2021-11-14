class Element():
    """Базовый класс, представляющий элемент графического интерфейса.

        Все конкретные элементы графического интерфейса наследуют данный класс.
        Каждый элемент должен принимать в качестве аргументов конструктора
        основные свойства (например, текст для текстового поля); вспомогательные
        аргументы (например, ширина) должны передаваться в качестве аргументов
        метода "set". В методе "build" в каждом элементе должна быть создана
        переменная "self.wgt", представляющая непосредственно виджет класса
        ipywidgets (эта переменная будет использоваться при создании вложенных
        элементов, при отрисовке элементов и т.п.).

        Note:
            Простейший способ создания элемента: "elm = Element(*СВОЙСТВА*)".
            Если необходимо указать вспомогательные аргументы, то создавать
            элемент необходимо следующим образом:
            "elm = Element(*ВСПОМОГАТЕЛЬНЫЕ СВОЙСТВА*).set(*СВОЙСТВА*).build()".

            Конструктор каждого элемента в качестве последнего аргумента должен
            иметь "kind", соответствующий типу элемента, который может быть:
            "p" ("primary"), "s" ("secondary"), "t" ("tertiary"), "a"
            ("accent"), "w" ("warning").

    """
    def __init__(self, kind='p'):
        self.kind = kind

        self.set()
        self.build()

    def build(self):
        self.wgt = None

        return self

    def layout(self):
        return ipywidgets.Layout(
            margin                  = '0px',
            width                   = self.w,
            height                  = self.h,
            min_width               = '50px',
            padding                 = '0px',
        )
        
    def set(self, w='auto', h='auto'):
        self.w = w
        self.h = h

        return self