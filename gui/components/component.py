class Component:
    """Базовый класс, представляющий компонент графического интерфейса."""
    def __init__(self, opts, run):
        self.opts = opts
        self.run = run

        self.build()

    def build(self):
        self.wgt = None

        return self

    def clear(self):
        return

    def on_run(self):
        self.clear()

        data = {}

        self.run(data)
