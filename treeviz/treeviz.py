from xml.dom import minidom
import plotly.graph_objects as go
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument('--graph_path', '-gp', type=str, required=True)
argparser.add_argument('--img_path', '-ip', type=str, required=True)


# Узел дерева
class CTreeNode:
    def __init__(self, name=None):
        self.name = name
        self.parent = None
        self.children = []


# Дерево, поддерживает загрузку из GraphML-формата
class CTree:
    def __init__(self, load_path=None):
        self.name_to_node = dict()
        self.tree_name = None
        self.root = None
        if load_path is not None:
            self.load(load_path)


    def load(self, path):
        dom = minidom.parse(path)
        root = dom.getElementsByTagName("graphml")[0]
        graph = root.getElementsByTagName("graph")[0]
        self.tree_name = graph.getAttribute('id')
        # Инициализация вершин
        for node in graph.getElementsByTagName("node"):
            node_name = node.getAttribute('id')
            tree_node = CTreeNode(node_name)
            self.name_to_node[node_name] = tree_node
        # Инициализация ребер
        for edge in graph.getElementsByTagName("edge"):
            src_node = self.name_to_node[edge.getAttribute('source')]
            tgt_node = self.name_to_node[edge.getAttribute('target')]
            src_node.children.append(tgt_node)
            tgt_node.parent = src_node
        # Поиск корня дерева
        for node in self.name_to_node.values():
            if node.parent is None:
                self.root = node
                break
        assert self.root is not None


# Механизм для отрисовки дерева
class CTreeDrawer:
    # Минимальное расстояние по оси x между узлами дерева
    MIN_DISTANCE = 2


    # Вспомогательная структура, хранящая контуры поддерева
    # контур - последовательность относительных смещений в порядке убывания глубины
    class CTreeContours:
        def __init__(self):
            self.left = [0]
            self.right = [0]


        def move(self, shift):
            self.left = list(map(lambda x: x + shift, self.left))
            self.right = list(map(lambda x: x + shift, self.right))


        def __len__(self):
            assert len(self.left) == len(self.right)
            return len(self.left)


    def build_displace(self, node):
        if len(node.children) == 0:
            return CTreeDrawer.CTreeContours()
        contours = [self.build_displace(child) for child in node.children]
        child_displacements = [0]
        contour = contours[0]
        for child_contour in contours[1:]:
            # Найдем расстояние между накопленным поддеревом и текущим дочерним поддеревом
            min_height = min(len(contour), len(child_contour))
            min_diff = min([r - l for r, l in zip(child_contour.left[-min_height:], contour.right[-min_height:])])
            curr_distance = CTreeDrawer.MIN_DISTANCE - min_diff
            assert curr_distance >= CTreeDrawer.MIN_DISTANCE
            child_displacements.append(child_displacements[-1] + curr_distance)
            # Сдвинем точку отсчета смещений в текущий дочерний узел и пересчитаем контуры
            contour.move(-curr_distance)
            if len(contour) == len(child_contour):
                contour.right = child_contour.right
            elif len(child_contour) > min_height:
                contour.right = child_contour.right
                contour.left = child_contour.left[:-min_height] + contour.left
            else:
                contour.right = contour.right[:-min_height] + child_contour.right
        # Сдвинем точку отсчета в текущий узел (он будет лежать посередине дочерних)
        x_range = child_displacements[-1]
        parent_position = x_range // 2
        contour.move(parent_position)
        # Заполним финальные относительные смещения дочерних узлов
        for i, (child, displace) in enumerate(zip(node.children, child_displacements)):
            self.displacements[child] = displace - x_range + parent_position
        # Добавим в контур текущий узел
        contour.left.append(0)
        contour.right.append(0)
        return contour


    def set_coord(self, node, depth, cum_shift):
        self.coord[node] = (depth, cum_shift)
        for child in node.children:
            self.set_coord(child, depth + 1, cum_shift + self.displacements[child])


    def build_plot(self, tree, ranges, save_path):
        # Построение картинки-визуализации
        node_y, node_x = zip(*self.coord.values())
        edge_y, edge_x = [], []
        for node in tree.name_to_node.values():
            y_from, x_from = self.coord[node]
            for child in node.children:
                y_to, x_to = self.coord[child]
                edge_x += [x_from, x_to, None]
                edge_y += [y_from, y_to, None]

        labels = [node.name[1:] for node in self.coord.keys()]
        annot = [dict(text=l, x=x, y=y,
                      xref='x1', yref='y1', font=dict(color='rgb(250,250,250)', size=10), showarrow=False)
                 for x, y, l in zip(node_x, node_y, labels)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                 line=dict(color='rgb(210,210,210)', width=5), hoverinfo='none'))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers',
                                 marker=dict(symbol='circle-dot', size=18, color='#6175c1',
                                             line=dict(color='rgb(50,50,50)', width=1)),
                                 text=labels, hoverinfo='text', opacity=0.8))
        pix_per_y = 50
        pix_per_x = 30
        fig.update_layout(showlegend=False,
                          width=pix_per_x * ranges[1], height=pix_per_y * ranges[0],
                          annotations=annot, margin=dict(t=0,b=0,l=0,r=0))

        if save_path is not None:
            fig.write_image(save_path)
        return fig


    def draw_tree(self, tree, save_path=None):
        # Строим смещения по x для каждого узла относительно его родителя
        self.displacements = dict()
        self.build_displace(tree.root)
        # Строим полные координаты каждого узла
        self.coord = dict()
        self.set_coord(tree.root, 0, 0)

        min_x = min(self.coord.values(), key=lambda yx: yx[1])[1]
        max_x = max(self.coord.values(), key=lambda yx: yx[1])[1]
        max_y = max(self.coord.values(), key=lambda yx: yx[0])[0]

        # Трансформируем координаты так,
        # чтобы дерево располагалось сверху вниз от корня к листьям,
        # и полностью лежало внутри первой четверти декартовой системы
        for node, (y, x) in self.coord.items():
            self.coord[node] = (max_y - y + 1, x - min_x + 1)

        return self.build_plot(tree, (max_y, max_x - min_x), save_path)


def main():
    args = argparser.parse_args()
    tree = CTree(args.graph_path)
    drawer = CTreeDrawer()
    drawer.draw_tree(tree, args.img_path)


if __name__ == '__main__':
    main()