## Визуализация деревьев

Реализован алгоритм Layered-Tree-Draw для произвольных деревьев.

Дерево считывается из формата GraphML, на выходе алгоритма SVG-картинка с визуализацией.

## Запуск кода

### Запуск ноутбука на колабе [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/S4nh1seR/DataViz/blob/main/treeviz/treeviz.ipynb)

### Запуск скрипта
python treeviz.py -gp ./graphml_trees/tree-dense.xml -ip ./results/tree-dense.svg

Параметры:
1. GraphPath (-gp) - путь к xml-файлу с графом в GraphML-формате
2. ImagePath (-ip) - путь к файлу с изображением-результатом (предпочтительный формат - SVG)

## Примеры работы

Примеры деревьев содержатся в директории ./graphml_trees, результаты для них - в директории ./results .

Бинарное дерево

![results](./results/bintree-84.svg)

Небинарное плотное дерево

![results](./results/tree-dense.svg)
