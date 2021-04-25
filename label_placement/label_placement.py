import plotly.graph_objects as go
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--rects_path', '-rp', type=str, required=True)
argparser.add_argument('--img_path', '-ip', type=str, required=True)

import plotly.graph_objects as go


class Pos:
    def __init__(self, *arr):
        assert len(arr) == 2
        self.xy = arr

    def __add__(self, that):
        return [a + b for a, b in zip(self.xy, that.xy)]

    def __sub__(self, that):
        return [a - b for a, b in zip(self.xy, that.xy)]

    def __neg__(self):
        return [-a for a in self.xy]

    def __str__(self):
        return str(self.xy)

    def __repr__(self):
        return str(self)

    def parse(s):
        arr = [int(a) for a in s.split(',')]
        return Pos(*arr)


class Box:
    def __init__(self, line):
        arr = line.split('\t')
        self.pos = Pos.parse(arr[0])
        self.size = Pos.parse(arr[1])
        self.offsets = [Pos.parse(pos) for pos in arr[2].split(' ')]
        assert len(self.offsets) > 0

    def __str__(self):
        return 'Box(pos={}, size={}, offsets={})'.format(self.pos, self.size, self.offsets)

    def __repr__(self):
        return str(self)


not_found_val = -1


def dfs(vertex_index, edges, used, on_enter_callback, on_exit_callback=None):
    on_enter_callback(vertex_index)
    for next_index in edges[vertex_index]:
        if used[next_index] == not_found_val:
            dfs(next_index, edges, used, on_enter_callback, on_exit_callback)
    if on_exit_callback is not None:
        on_exit_callback(vertex_index)


def kosaraju_alg(edges, inv_edges):
    n_vertices = len(edges)

    order = []
    used = [not_found_val] * n_vertices

    def inv_on_enter(vertex):
        used[vertex] = 1

    def inv_on_exit(vertex):
        order.append(vertex)

    for i in range(n_vertices):
        if used[i] == not_found_val:
            dfs(i, edges, used, inv_on_enter, inv_on_exit)

    comp = [not_found_val] * n_vertices
    comp_index = 0

    def on_enter(vertex):
        comp[vertex] = comp_index

    for i in reversed(order):
        if comp[i] == not_found_val:
            dfs(i, inv_edges, comp, on_enter)
        comp_index += 1
    return comp


def have_intersection(box_lhs, lhs_i, box_rhs, rhs_i):
    lhs_bottom_left = box_lhs.pos - box_lhs.offsets[lhs_i]
    rhs_bottom_left = box_rhs.pos - box_rhs.offsets[rhs_i]
    lhs_top_right = Pos(*lhs_bottom_left) + box_lhs.size
    rhs_top_right = Pos(*rhs_bottom_left) + box_rhs.size
    return \
        (rhs_bottom_left[0] <= lhs_top_right[0] and rhs_top_right[0] >= lhs_bottom_left[0]) and \
        (rhs_bottom_left[1] <= lhs_top_right[1] and rhs_top_right[1] >= lhs_bottom_left[1])


def draw_label_rects(boxes, choosed_indices, save_path=None):
    fig = go.Figure()

    axis_values = []
    for box, index in zip(boxes, choosed_indices):
        bottom_left = box.pos - box.offsets[index]
        top_right = Pos(*bottom_left) + box.size
        axis_values.append((bottom_left[0], bottom_left[1], top_right[0], top_right[1]))

    pad = 10
    min_x = min(list(zip(*axis_values))[0]) - pad
    max_x = max(list(zip(*axis_values))[2]) + pad
    min_y = min(list(zip(*axis_values))[1]) - pad
    max_y = max(list(zip(*axis_values))[3]) + pad

    fig.update_xaxes(range=[min_x, max_x])
    fig.update_yaxes(range=[min_y, max_y])

    for box_values in axis_values:
        left, bottom, right, top = box_values
        fig.add_shape(type="rect",
                      x0=left, y0=bottom, x1=right, y1=top,
                      line=dict(color="RoyalBlue", width=2, )
                      )
    pos_x = [box.pos.xy[0] for box in boxes]
    pos_y = [box.pos.xy[1] for box in boxes]
    fig.add_trace(go.Scatter(x=pos_x, y=pos_y, mode='markers',
                             marker=dict(symbol='circle-dot', size=5, color='rgb(255, 0, 0)'), opacity=1.0))
    fig.update_layout(showlegend=False,
                      margin=dict(t=0, b=0, l=0, r=0))

    if save_path is not None:
        fig.write_image(save_path)
    return fig


def solve_bin_label_placement(path, img_save_path=None):
    boxes = []
    with open(path, "r") as f:
        boxes = [Box(line) for line in f.read().strip().split('\n')]

    n_vertices = len(boxes) * 2
    edges = [[] for _ in range(n_vertices)]
    inv_edges = [[] for _ in range(n_vertices)]
    for i, box_i in enumerate(boxes):
        for j, box_j in enumerate(boxes[i + 1:], start=i + 1):
            for i_offset in range(len(box_i.offsets)):
                for j_offset in range(len(box_j.offsets)):
                    if have_intersection(box_i, i_offset, box_j, j_offset):
                        edges[2 * i + (1 - i_offset)].append(2 * j + j_offset)
                        edges[2 * j + (1 - j_offset)].append(2 * i + i_offset)

                        inv_edges[2 * j + j_offset].append(2 * i + (1 - i_offset))
                        inv_edges[2 * i + i_offset].append(2 * j + (1 - j_offset))

    comp = kosaraju_alg(edges, inv_edges)
    answer = []
    for i in range(len(boxes)):
        if comp[2 * i] == comp[2 * i + 1]:
            print("Solution doesn't exist!")
            return None
        else:
            answer.append(int(comp[2 * i] > comp[2 * i + 1]))
    return draw_label_rects(boxes, answer, img_save_path)


def main():
    args = argparser.parse_args()
    solve_bin_label_placement(args.rects_path, args.img_path)


if __name__ == '__main__':
    main()