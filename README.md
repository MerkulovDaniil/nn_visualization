# intevis

> Платформа **intevis** (**INTE**llectual **VIS**ualization) для визуализации искусственных нейронных сетей.


## Работа с программой

## "Облачный" запуск

Для запуска достаточно выполнить/запустить первую ячейку с кодом в colab ноутбуке [intevis](https://drive.google.com/file/d/1qrfXf1Oze0J2RoaodDYwUlTJ8aEwIzZ_/view?usp=sharing).

## "Локальный" запуск

Для установки и запуска программы достаточно выполнить следующие шаги:
1. Создать виртуальное окружение:
    ```bash
    conda create --name intevis python=3.7
    ```
1. Перейти в виртуальное окружение `intevis`:
    ```bash
    conda activate intevis
    ```
1. Установить все необходимые зависимости:
    ```bash
    pip install -r requirements.txt
    ```
1. Запустить jupyter-ноутбук:
    ```bash
    jupyter lab
    ```
1. В интерфейсе jupyter ноутбука открыть файл `intevis.ipynb` и выполнить/запустить первую ячейку с кодом.
1. По окончании работы при необходимости можно удалить окружение:
    ```bash
    conda activate && conda remove --name intevis --all
    ```


## Документация

Документация находится в папке `doc`. Для просмотра документации достаточно открыть в web-браузере файл `doc/_build/html/index.html`. Для самостоятельной сборки документации в формате `html`, должна быть выполнена команда `sphinx-build ./doc ./doc/_build/html` из корня проекта. Для самостоятельной сборки документации в формате `pdf`, должна быть выполнена команда `sphinx-build -M latexpdf ./doc doc/_build/pdf` из корня проекта.

Оформление программного кода и документации осуществлялось нами с использованием популярного [набора стилей от google](https://google.github.io/styleguide/pyguide.html). Для генерации документации используется популярная система [sphinx](http://sphinxsearch.com/) (тема [alabaster](https://alabaster.readthedocs.io/en/latest/)) и плагин [napoleon](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), поддерживающий документацию, оформленную в системе стилей google.


## Структура файлов и папок

- `intevis.py` - основной скрипт ПО, содержащий класс `Intevis`, представляющий менеджер ПО. Для запуска графического пользовательского интерфейса следует использовать метод класса `gui`;
- `activation` - папка содержит программный код, связанный с активаций отдельных нейронов и слоев ИНС;
- `architecture` - папка содержит программный код, связанный c визуализацией архитектуры ИНС;
- `attribution` - папка содержит программный код, связанный с построением атрибуции ИНС;
- `doc` - папка содержит программный код автоматически генерируемой документации:
- `gui` - папка содержит программный код, связанный с графическим пользовательским интерфейсом, в том числе:
  - `gui.py` - основной скрипт, содержащий класс `Gui`, представляющий менеджер графического пользовательского интерфейса. В качестве основы используется популярный python пакет `ipywidgets`;
  - `style.css` - файл содержит css-стили пользовательского интерфейса приложения;
  - `components` - папка содержит основные компоненты (вкладки) пользовательского интерфейса. Подробные комментарии по использованию элементов приведены в базовом классе `Component` в файле `component.py`;
  - `elements` - папка содержит элементы пользовательского интерфейса (кнопки, выпадающие меню и т.п.). Подробные комментарии по использованию элементов приведены в базовом классе `Element` в файле `element.py`.
