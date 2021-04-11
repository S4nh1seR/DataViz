from xml.dom import minidom
from typing import List, Optional
import copy
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import linprog
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--graph_path', '-gp', type=str, required=True)
argparser.add_argument('--img_path', '-ip', type=str, required=True)
argparser.add_argument('--max_width', '-mw', type=int, required=False)

class CDigraphNode:
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.ingoing = []
        self.outgoing = []


class CDigraph:
    def __init__(self, load_path: Optional[str] = None):
        self.name_to_node = dict()
        if load_path is not None:
            self.load(load_path)

    def load(self, path: str):
        dom = minidom.parse(path)
        root = dom.getElementsByTagName("graphml")[0]
        graph = root.getElementsByTagName("graph")[0]
        self.graph_name = graph.getAttribute('id')
        for node in graph.getElementsByTagName("node"):
            node_name = node.getAttribute('id')
            tree_node = CDigraphNode(node_name)
            self.name_to_node[node_name] = tree_node
        for edge in graph.getElementsByTagName("edge"):
            src_node = self.name_to_node[edge.getAttribute('source')]
            tgt_node = self.name_to_node[edge.getAttribute('target')]
            src_node.outgoing.append(tgt_node)
            tgt_node.ingoing.append(src_node)


def less(ordered_lhs: List[int], ordered_rhs: List[int]):
    if len(ordered_lhs) == 0:
        return True
    if len(ordered_rhs) == 0:
        return False
    if ordered_lhs[-1] == ordered_rhs[-1]:
        return less(ordered_lhs[:-1], ordered_rhs[:-1])
    elif ordered_lhs[-1] < ordered_rhs[-1]:
        return True
    else:
        return False


class CDigraphDrawer:
    def graham_coffman(self, max_width: int):
        n_nodes = len(self.nodes)
        undefined_label = n_nodes
        labels = {node: undefined_label for node in self.nodes}
        for i in range(n_nodes):
            min_in_labels = [undefined_label]
            best_node = None
            for node in self.nodes:
                if labels[node] != undefined_label:
                    continue
                curr_in_labels = sorted([labels[v_in] for v_in in node.ingoing])
                if less(curr_in_labels, min_in_labels):
                    min_in_labels = curr_in_labels
                    best_node = node
            labels[best_node] = i

        used = set()
        self.node_to_layer = dict()
        self.layers = []
        while len(used) != n_nodes:
            curr_node = None
            curr_label = -1
            for node in self.nodes:
                if node in used:
                    continue
                if (len(used) == 0 or set(node.outgoing) < used) and labels[node] > curr_label:
                    curr_node = node
                    curr_label = labels[node]

            assert curr_node is not None
            max_child_layer = -1
            for child in curr_node.outgoing:
                max_child_layer = max(max_child_layer, self.node_to_layer[child])
            layer_to_ext = max_child_layer + 1
            while layer_to_ext < len(self.layers) and len(self.layers[layer_to_ext]) == max_width:
                layer_to_ext += 1

            if layer_to_ext == len(self.layers):
                self.layers.append([curr_node])
            else:
                self.layers[layer_to_ext].append(curr_node)
            self.node_to_layer[curr_node] = layer_to_ext
            used.add(curr_node)

    def minimize_dummy(self):
        n_nodes = len(self.nodes)
        n_edges = sum([len(node.outgoing) for node in self.nodes])

        A = np.zeros(shape=(n_edges, n_nodes))
        b = np.ones(shape=n_edges) * -1.0
        c = np.zeros(shape=n_nodes)

        edge_index = 0
        for i, node in enumerate(self.nodes):
            indeg, outdeg = len(node.ingoing), len(node.outgoing)
            c[i] = outdeg - indeg
            for child in node.outgoing:
                j = self.nodes.index(child)
                A[edge_index, j] = 1.0
                A[edge_index, i] = -1.0
                edge_index += 1
        bounds = [(1.0, None) for _ in range(n_nodes)]
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
        min_y = min(res.x)
        eps = 1e-5
        y = list(map(lambda c: int(c - min_y + eps), res.x))
        n_layers = max(y) + 1

        self.node_to_layer = dict()
        self.layers = []
        for i in range(n_layers):
            self.layers.append([])
            layer = self.layers[-1]
            for node, value in zip(self.nodes, y):
                if value == i:
                    layer.append(node)
                    self.node_to_layer[node] = i

    def add_dummy(self):
        self.dummy_nodes = set()
        for node in self.nodes:
            new_outgoing = node.outgoing.copy()
            for child in node.outgoing:
                lower_layer = self.node_to_layer[child]
                upper_layer = self.node_to_layer[node]
                vertices_between = upper_layer - lower_layer - 1
                if vertices_between > 0:
                    new_outgoing.remove(child)
                    child.ingoing.remove(node)
                    prev_node = child
                    for i in range(vertices_between):
                        new_node = CDigraphNode(name=f'{node.name}_{child.name}[{i}]')
                        self.dummy_nodes.add(new_node)
                        prev_node.ingoing.append(new_node)
                        new_node.outgoing.append(prev_node)
                        dummy_layer_index = lower_layer + 1 + i
                        self.layers[dummy_layer_index].append(new_node)
                        self.node_to_layer[new_node] = dummy_layer_index
                        prev_node = new_node
                    prev_node.ingoing.append(node)
                    new_outgoing.append(prev_node)
            node.outgoing = new_outgoing

    def build_plot(self, save_path: Optional[str] = None):
        edge_y, edge_x = [], []
        node_y, node_x = [], []
        for node in self.nodes:
            y_from, x_from = self.y_coord[node], self.x_coord[node]
            node_y.append(y_from)
            node_x.append(x_from)
            for child in node.outgoing:
                y_to, x_to = self.y_coord[child], self.x_coord[child]
                edge_x += [x_from, x_to, None]
                edge_y += [y_from, y_to, None]

        dummy_y, dummy_x = [], []
        for node in self.dummy_nodes:
            y_from, x_from = self.y_coord[node], self.x_coord[node]
            dummy_y.append(self.y_coord[node])
            dummy_x.append(self.x_coord[node])
            for child in node.outgoing:
                y_to, x_to = self.y_coord[child], self.x_coord[child]
                edge_x += [x_from, x_to, None]
                edge_y += [y_from, y_to, None]
        labels = [node.name[1:] for node in self.nodes]
        annot = [dict(text=l, x=x, y=y,
                      xref='x1', yref='y1', font=dict(color='rgb(250,250,250)', size=10), showarrow=False)
                 for x, y, l in zip(node_x, node_y, labels)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                 line=dict(color='rgb(210,210,210)', width=2), hoverinfo='none'))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers',
                                 marker=dict(symbol='circle-dot', size=18, color='#6175c1',
                                             line=dict(color='rgb(50,50,50)', width=1)),
                                 text=labels, hoverinfo='text', opacity=0.8))
        fig.add_trace(go.Scatter(x=dummy_x, y=dummy_y, mode='markers',
                                 marker=dict(symbol='circle-dot', size=9, color='#6175c1',
                                             line=dict(color='rgb(50,50,50)', width=1)), hoverinfo='none'))
        min_x = min(node_x + dummy_x)
        max_x = max(node_x + dummy_x)
        min_y = min(node_y + dummy_y)
        max_y = max(node_y + dummy_y)
        pix_per_y = 80
        pix_per_x = 50
        fig.update_layout(showlegend=False,
                          width=pix_per_x * (max_x - min_x), height=pix_per_y * (max_y - min_y),
                          annotations=annot, margin=dict(t=0, b=0, l=0, r=0))
        if save_path is not None:
            fig.write_image(save_path)
        return fig

    def fill_coord(self):
        x_delta = 1
        self.y_coord = dict()
        self.x_coord = dict()

        max_width_in_nodes = max(list(map(len, self.layers)))
        max_width = x_delta * (max_width_in_nodes - 1)
        center_x = 1 + max_width // 2

        for y, layer in enumerate(reversed(self.layers), start=1):
            layer_width = x_delta * (len(layer) - 1)
            layer_width_half = layer_width // 2
            for i, node in enumerate(layer):
                self.y_coord[node] = len(self.layers) - y
                self.x_coord[node] = center_x - layer_width_half + i * x_delta

    def order_barycenter(self, anchor_layer: List[CDigraphNode], layer: List[CDigraphNode], link_type: str):
        anchor_ord = dict([(node, i) for i, node in enumerate(anchor_layer)])
        layer_ord = []
        for node in layer:
            linked = getattr(node, link_type)
            avg = 0
            if len(linked) != 0:
                avg = sum([anchor_ord[el] for el in linked]) // len(linked)
            layer_ord.append((node, avg))
        layer_ord = sorted(layer_ord, key=lambda el: el[1])
        layer = list(zip(*layer_ord))[0]

    def minimize_crossings(self, anchor_layer: List[CDigraphNode], layer: List[CDigraphNode], link_type: str):
        anchor_ord = dict([(node, i) for i, node in enumerate(anchor_layer)])
        for i in range(len(layer) - 1):
            for j in range(i + 1, len(layer)):
                u = layer[i]
                v = layer[i + 1]
                prev_u = getattr(u, link_type)
                prev_v = getattr(v, link_type)
                c_uv = 0
                for k in prev_u:
                    for l in prev_v:
                        if anchor_ord[l] < anchor_ord[k]:
                            c_uv += 1
                c_vu = len(prev_u) * len(prev_v) - c_uv
                if c_uv > c_vu:
                    layer[i] = v
                    layer[i + 1] = u

    def order_layers_by_x(self):
        for i in range(len(self.layers) - 1):
            self.order_barycenter(self.layers[i], self.layers[i + 1], "outgoing")

        n_steps = 5
        step = 0
        while step < n_steps:
            for i in range(len(self.layers) - 1):
                self.minimize_crossings(self.layers[i], self.layers[i + 1], "outgoing")
            for i in range(len(self.layers) - 1, -1, -1):
                self.minimize_crossings(self.layers[i], self.layers[i - 1], "ingoing")
            step += 1

    def draw_digraph(self, digraph: CDigraph,
                     max_width: Optional[int] = None, save_path: Optional[str] = None):
        self.graph = copy.deepcopy(digraph)
        self.nodes = list(self.graph.name_to_node.values())
        if max_width is None:
            self.minimize_dummy()
        else:
            self.graham_coffman(max_width)

        self.add_dummy()

        self.order_layers_by_x()

        self.fill_coord()

        return self.build_plot(save_path)

def main():
    args = argparser.parse_args()
    digraph = CDigraph(args.graph_path)
    drawer = CDigraphDrawer()
    drawer.draw_digraph(digraph, max_width=args.max_width, save_path=args.img_path)

if __name__ == '__main__':
    main()