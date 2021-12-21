from datetime import datetime


from ..elements.html import Html
from ..elements.output import Output


from .component import Component


class Cs(Component):
    """Вкладка "Консоль"."""
    def build(self):
        self.wgt = Output()
        self.wgt.set(h=self.opts.app_height).build()

        return self

    def get_date(self):
        return datetime.now().strftime('%m.%d.%Y; %H:%M')

    def get_time(self):
        return datetime.now().strftime('%H:%M:%S')

    def log(self, text, kind='res'):
        text_out = self.get_time() + f' [{kind.upper()}] > ' + text
        #self.wgt.add_text(text_out)

        cl_kind = 'cs-msg__kind-' + kind

        html = ''
        html += '<div class="cs-msg">'
        html += f'<span class="cs-msg__date">{self.get_time()}</span>'
        html += f'<span class="cs-msg__kind {cl_kind}">[{kind.upper()}]</span>'
        html += f'<span class="cs-msg__text">{text}</span>'
        self.wgt.add(Html(html))
