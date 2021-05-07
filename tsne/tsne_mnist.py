import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import math
import pickle
import plotly.graph_objects as go
from timeit import default_timer as timer
from sklearn.decomposition import PCA
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument('--mnist_path', '-mp', type=str, required=True)
argparser.add_argument('--img_path', '-ip', type=str, required=True)
argparser.add_argument('--use_pca', '-up', action="store_true")


def update_p(P, distances, beta, entropy, sample_index):
    P[sample_index] = np.exp(-distances[sample_index] * beta)
    P[sample_index][sample_index] = 0.0

    Psum = np.sum(P[sample_index])
    P[sample_index] /= Psum

    curr_entropy = math.log(Psum) + beta * np.sum(distances[sample_index] * P[sample_index])
    return curr_entropy - entropy


def calculate_p(X, perplexity):
    n_samples = X.shape[0]
    entropy = math.log(perplexity)
    entropy_tol = 1e-5
    max_bin_search_iterations = 100

    distances = euclidean_distances(X, X, squared=True)
    P = np.zeros(shape=(n_samples, n_samples))
    for i in range(n_samples):
        beta = 1.0
        beta_min = np.NINF
        beta_max = np.PINF

        entropy_diff = update_p(P, distances, beta, entropy, i)
        iteration = 0
        while math.fabs(entropy_diff) > entropy_tol and iteration < max_bin_search_iterations:
            if entropy_diff > 0.0:
                beta_min = beta
                if np.isposinf(beta_max):
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if np.isneginf(beta_min):
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0
            entropy_diff = update_p(P, distances, beta, entropy, i)
            iteration += 1
    return P


def update_gains_by_jacobs(last_update, grad, gains, min_gain):
    pos_mask = np.sign(last_update) != np.sign(grad)
    neg_mask = np.invert(pos_mask)
    gains[pos_mask] += 0.2
    gains[neg_mask] *= 0.8
    gains = np.clip(gains, min_gain, np.inf)


def t_sne(X, perplexity=40):
    n_iterations = 1000

    stage1_momentum = 0.5
    stage2_momentum = 0.8
    stage_switch_iteration = 250

    initial_lr = 200
    early_exaggeration_coef = 4.0
    early_exaggeration_n_iterations = 100
    min_gain = 0.01

    n_samples, src_dim = X.shape
    emb_dim = 2

    P = calculate_p(X, perplexity)
    P = 0.5 * (P + P.T)
    P /= P.sum()
    P *= early_exaggeration_coef
    momentum = stage1_momentum

    np.random.seed(42)
    Y = np.random.randn(n_samples, emb_dim) * 0.01
    last_update = np.zeros(shape=(n_samples, emb_dim))
    gains = np.ones(shape=(n_samples, emb_dim))

    for i in range(n_iterations):
        print('\rTSNE iteration: {}/{}...'.format(i + 1, n_iterations), end='')

        ydist = euclidean_distances(Y, Y, squared=True)
        ydist += 1.0
        ydist **= -1.0
        Q = ydist / np.sum(ydist)

        pqd = (P - Q) * ydist
        grad = 4.0 * np.dot(np.diag(np.sum(pqd, axis=1)) - pqd, Y)
        update_gains_by_jacobs(last_update, grad, gains, min_gain)
        grad *= gains

        last_update = momentum * last_update - initial_lr * grad
        Y += last_update

        if i == stage_switch_iteration:
            momentum = stage2_momentum
        if i == early_exaggeration_n_iterations:
            P /= early_exaggeration_coef
    print()
    return Y


def load_mnist(mnist_path):
    with open(mnist_path,'rb') as f:
        mnist = pickle.load(f)
    return mnist["train_data"], mnist["train_labels"], mnist["test_data"], mnist["test_labels"]


def draw_mnist_viz(embedded, y, save_path=None, pix_per_axis_val=(4,4)):
    x_coord = embedded[:,0]
    y_coord = embedded[:,1]

    fig = go.Figure()
    for i in range(10):
        digit_mask = y == i
        x_c = list(x_coord[digit_mask])
        y_c = list(y_coord[digit_mask])
        fig.add_trace(go.Scatter(x=x_c, y=y_c, mode='markers', name=str(i)))

        x_text, y_text = np.median(embedded[y == i, :], axis=0)
        fig.add_annotation(dict(font=dict(size=20)), bgcolor="white", x=x_text, y=y_text,text=str(i), showarrow=False)

    pad = 10
    min_x = min(x_coord) - pad
    max_x = max(x_coord) + pad
    min_y = min(y_coord) - pad
    max_y = max(y_coord) + pad

    fig.update_layout(showlegend=True,
                      width=pix_per_axis_val[0] * (max_x - min_x), height=pix_per_axis_val[1] * (max_y - min_y),
                      margin=dict(t=0,b=0,l=0,r=0))

    if save_path is not None:
        fig.write_image(save_path)
    return fig


def main():
    args = argparser.parse_args()

    train_x, train_y, test_x, test_y = load_mnist(args.mnist_path)
    n_samples = 3000
    data_x, data_y = train_x[:n_samples], train_y[:n_samples]
    data_x = data_x / 255.0

    if args.use_pca:
        reduced_dim = 30
        pca = PCA(n_components=reduced_dim)
        data_x = pca.fit_transform(data_x)

    t1 = timer()
    embedded = t_sne(data_x)
    t2 = timer()
    print("TSNE took {} seconds".format(t2 - t1))

    draw_mnist_viz(embedded, data_y, save_path=args.img_path)

if __name__ == '__main__':
    main()