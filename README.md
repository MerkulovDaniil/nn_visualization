# Neural Networks visualization

## Architecture visualization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17oDLdK_3wu_MYJNRzK29WRmmptPTtF-q?usp=sharing)

Попробовали разные способы отрисовки архитектуры с использованием разных библиотек (torchinfo, torchsummary, hiddenlayer, torchviz, tensorboard). Для того, чтобы, например, корректно выбрать ("тыкнуть пальцем") слой/фильтр в методе максимизации активации, нужно аккуратно отрисовать архитектуру ANN. Мы демонстрируем разные варианты, однако оптимальный (достаточно удобный) не нашли пока что. В частности, удивляет, что в colab граф сети в tensorboard рисуется в крошечном виджете (это баг или так и должно быть?).

**Plan**:
1. Найти оптимальный способ визуализации архитектуры.
1. Подготовить функцию, осуществляющую визуализацию архитектуры.
1. Сделать симпатичную визуализацию используемой нами сети.

- [ ] TODO [**Д**] Изучаем существующие средства визуализации архитектуры сетей. Размышляем о перспективности создания собственной полноценной супер-пупер-системы (конкурентные преимущества!). Особенно обращаем внимание на jax/flax (открытая ниша).

## Inner visualization

### Analysis of existing software products

Краткий анализ проводился в нашей обзорной статье. Для простоты и конкретики мы пока что ограничимся библиотекой `captum`.

### Captum

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tnS6UncQPrJohfio7nP20w54b6JQcdTE?usp=sharing)

На данный момент, для сети VGG-16 посредством простейшего метода Saliency Maps (Heatmaps), вроде как согласно работе Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps (2013; Cited 2398), сгенерирована соответствующая классическая картинка. Captum пока вообще не используется!

**Plan**:
1. Изучить возможности и ограничения captum.
1. Реализовать Saliency Maps (Heatmaps) с captum и сравнить с результатом, полученным без captum (есть в данном ноутбуке). См. также раздел `Saliency Maps` ниже.
1. Реализовать Integrated Gradients для VGG-16 с captum и сравнить с нашим результатом. См. также раздел `Integrated Gradients` ниже.
1. Реализовать что-то еще с помощью captum.
1. Изучить captum_insights, обдумать, нужна ли она нам, если нужна, то научиться включать ее в colab.

- [ ] TODO [**Е**] Выявляем недостатки и достоинства captum.
- [ ] TODO [**Е**] Реализуем с captum Saliency Maps (Heatmaps), сравниваем с нашей собственной наивной реализацией.
- [ ] TODO [**Е**] Реализуем с captum Integrated Gradients, сравниваем с нашей собственной реализацией.

### Analysis of existing visualization methods

Краткий анализ проводился в нашей обзорной статье. Для простоты и конкретики мы пока что ограничимся тремя методами:
1. `Saliency Maps`
1. `Activation Maximization`
1. `Integrated Gradients`

### Saliency Maps

> См. также раздел `Analysis of existing software products > Captum` выше.

### Activation Maximization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Jwhw8paNR1DAgWO7jx5kPKh5SLK2JKA4?usp=sharing)

**Plan**:
1. Устранить баг, который препятствует запуску на GPU.
1. Расписать теорию метода.
1. Улучшить/уточнить алгоритм (какой-то он слишком простой получается).
1. Улучшить результаты (визуализация для фильтров возможно недостаточно яркая - не заметно, что последние фильтры настроены на очень высокоорганизованные фичи).
1. Улучшить результаты для DeepDream (DeepDream дает убогую картинку, возможно есть смысл поэкспериментировать, чтобы получалось что-то эффектное).

- [ ] TODO [**Е, Д** ?] Уточняем код Activation Maximization, добавляем поддержку работы на GPU (исправляем баг).
- [ ] TODO [**Е** ?] Улучшаем метод Activation Maximization: вводим различные регуляризации и т.п.

### Integrated Gradients

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15xZFBi_FCWCgPeG0yFHhkSPjkw4hnkCB?usp=sharing)

Основная статья: Axiomatic Attribution for Deep Networks (ICML-2017; Cited 1149).

**Plan**:
1. Более детально разобраться в коде и сделать масштабный рефакторинг.
1. Расписать теорию метода.
1. Улучшить/уточнить алгоритм.
1. Получить больше симпатичных результатов.

- [ ] TODO [**A**] Уточняем код и результаты нашей реализации Integrated Gradients, а также расписываем теорию по методу.

> См. также раздел `Analysis of existing software products > Captum` выше.

## Adversarial examples

- [ ] TODO [**С**] Обдумываем и формализуем актуальные алгоритмы построения злонамеренных примеров.

## Report
### Informal report # 1

Возможная структура (черновик!):
1. Есть много языков, мы выбрали python (ибо...)
1. Есть много фреймворков машинного обучения, мы выбрали pytorch (ибо...)
1. Есть много способов организации пользовательского интерфейса, мы выбрали jupyter / colab (ибо - нефиг заморачиваться)
1. Есть много библиотек для визуализации, мы выбрали captum (ибо он активно развивается и т.п.)
1. Есть много методов визуализации, мы выбрали три: Saliency Maps, Activation Maximization, Integrated Gradients (ибо...)
1. Мы реализовали Saliency Maps в captum - вот стандартный результат
1. Мы реализовали Activation Maximization самостоятельно, поскольку в captum он не поддерживается - вот стандартный результат
1. Мы реализовали Integrated Gradients самостоятельно, поскольку нам пока что лень разбираться с его реализацией в captum (или она там кривая или что-то еще) - вот стандартный результат
1. Еще мы посмотрели разные библиотеки для визуализации архитектуры сети - вот разные примеры - пока они нас не очень устраивают
1. Приводим предварительный план на будущее: доработать, допилить, объединить,
1. ...

- [ ] TODO [**Е, Д, А**] Готовим презентацию.
- [ ] TODO [**Е, Д, А**] Отправляем готовую презентацию заказчику (**не позднее 1 марта**).
- [ ] TODO [**Е, Д, А**] Просим заказчика поделиться идеями о перспективных реальных задачах визуализации (связь с исследованиями мозга.

## Various

- См. нашу [обзорную статью](https://www.overleaf.com/project/5fac61b07bd15b41cfd99811) (или в папке `text` файл `матвеев2021обзор.pdf`).
- См. нашу обширную [excel таблицу](https://docs.google.com/spreadsheets/d/1HZTYd0SyoVlbXnfCxUmTdv6-HaOXMsOmqVyKSTvdUjo/edit?usp=sharing) с перечнем публикаций и методов.

- [ ] TODO [**ALL**] Обдумываем и формализуем задачу анализа "энергоэффективности" сетей в контексте визуализации.
- [ ] TODO [**Д, Е, А**] Размышляем о разумности перехода с pytorch на jax / flax (?).
