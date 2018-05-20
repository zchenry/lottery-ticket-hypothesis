import warnings
warnings.filterwarnings("ignore")

import pdb
import time
import chainer
import argparse
import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer.cuda import to_cpu
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel
import multiprocessing
from joblib import Parallel, delayed

def pr(s, l=100):
    print(' ' * l, end='\r')
    print(s, end='\r')

def tri(dic, key):
    return None if dic is None else dic[key]

class MLP(chainer.Chain):
    def __init__(self, H, weights):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(2, H, nobias=True, initialW=tri(weights, 'w1'))
            self.l2 = L.Linear(H, 1, nobias=True, initialW=tri(weights, 'w2'))

    def forward(self, x, keep_val=False):
        h1 = F.relu(self.l1(x))
        h2 = self.l2(h1)

        if keep_val:
            return h2, [h1, h2]
        else:
            return h2

    def res_y(self):
        return self.forward(self.x)

    def set_xy(self):
        self.x = self.xp.array(
            [[0, 0], [1, 1], [0, 1], [1, 0]], dtype=self.xp.float32)
        self.y = self.xp.array(
            [-1, -1, 1, 1], dtype=self.xp.int32)[:, None]

    def loss(self, x, y):
        loss = F.sigmoid_cross_entropy(self.res_y(), y)
        return loss

    def if_db(self):
        return not ((self.y * self.res_y()).data < 0).any()

class MNIST(chainer.Chain):
    def __init__(self, weights, p):
        super(MNIST, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(784, 300,
                               initialW=weights['w1'],
                               initial_bias=weights['b1'])
            self.l2 = L.Linear(300, 100,
                               initialW=weights['w2'],
                               initial_bias=weights['b2'])
            self.l3 = L.Linear(100, 10, nobias=True,
                               initialW=weights['w3'])
            self.p = p

    def forward(self, x, keep_val=False):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = self.l3(h2)

        if keep_val:
            return h3, [h1, h2, h3]
        else:
            return h3

    def loss(self, x, y):
        loss = F.softmax_cross_entropy(self.forward(x), y)
        return loss

def tuple2array(data, xp):
    xs = xp.array([d[0] for d in data], dtype=xp.float32)
    ys = xp.array([d[1] for d in data])
    return xs, ys

def inits(d1, d2, xp, s='', mu=0., sigma=0.1):
    samples = xp.random.normal(mu, sigma, d1 * d2)
    outliers = xp.abs(samples) > sigma * 2
    while outliers.any():
        pr('{} {}/{} left'.format(s, sum(outliers), d1 * d2))
        samples[outliers] = xp.random.normal(mu, sigma, sum(outliers))
        outliers = xp.abs(samples) > sigma * 2
    return samples.reshape((d2, d1))

def mesh_inputs(*args):
    ls = np.linspace(*args)
    xs, ys = np.meshgrid(ls, ls)
    return xs, ys, np.append(xs.reshape([-1, 1]), ys.reshape([-1, 1]), 1).astype(np.float32)

def calc_mi(xs, ys, ts, bins):
    ys = ys.astype(np.float32)
    pys, pys1, py_x, b1, b, \
        unique_a, unique_x, unique_y, pxs = extract_probs(ys, xs)
    return Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(calc_information_for_layer_with_other)(
            ts[e][l], bins, unique_x, unique_y, ys,
            b, b1, len(unique_a), pxs, py_x, pys1, e, l)
        for e in range(len(ts)) for l in range(len(ts[e])))

def mmd(x1, y1):
    g = 1. / (2. * median(x1)**2)
    kxx = np.mean(rbf_kernel(x1, x1, g))
    kxy = np.mean(rbf_kernel(x1, y1, g))
    kyy = np.mean(rbf_kernel(y1, y1, g))
    return kxx - 2. * kxy + kyy

def median(x1):
    particle = len(x1)
    sq_dist = pdist(x1)
    pairwise_dists = squareform(sq_dist)**2
    band = np.median(pairwise_dists)
    band = np.sqrt(0.5 * band / np.log(particle + 1))
    return band

def pdist2(x1):
    r = tf.reduce_sum(x1*x1, 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2.*tf.matmul(x1, tf.transpose(x1)) + tf.transpose(r)
    return D

def calc_mmds(xs, ys, ts):
    return Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(calc_mmd)(ts[e][l], xs, ys, e, l)
        for e in range(len(ts)) for l in range(len(ts[e])))

def calc_hsics(xs, ys, ts):
    return Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(calc_hsic)(ts[e][l], xs, ys, e, l)
        for e in range(len(ts)) for l in range(len(ts[e])))

def calc_hsic(ts, xs, ys, e, t):
    pr('Calculating HSIC for SNAP {} LAYER {}...'.format(e + 1, t + 1))
    m = len(ts)
    kx = rbf_kernel(xs, xs, 1./(2*median(xs)**2))
    ky = rbf_kernel(ys, ys, 1./(2*median(ys)**2))
    l = rbf_kernel(ts, ts, 1./(2*median(ts)**2))
    h = np.ones((m, m)) * (1. - 1./m)
    hlh = h @ l @ h
    params = {}
    params['ixt'] = np.trace(kx @ hlh) / m**2
    params['ity'] = np.trace(ky @ hlh) / m**2
    params['epoch'] = e
    params['layer'] = t
    return params

def calc_information_for_layer_with_other(data, bins, unique_inverse_x,
                                          unique_inverse_y, label, b, b1,
                                          len_unique_a, pxs, p_YgX, pys1, e, l,
                                          percent_of_sampling=50):
    pr('Calculating MI for SNAP {} LAYER {}...'.format(e + 1, l + 1))
    IXT, ITY = calc_information_sampling(data, bins, pys1, pxs, label, b, b1,
                                         len_unique_a, p_YgX, unique_inverse_x,
                                         unique_inverse_y)
    number_of_indexs = int(data.shape[1] * (1. / 100 * percent_of_sampling))
    indexs_of_sampls = np.random.choice(data.shape[1], number_of_indexs,
                                        replace=False)
    if percent_of_sampling != 100:
        sampled_data = data[:, indexs_of_sampls]
        sampled_IXT, sampled_ITY = calc_information_sampling(
            sampled_data, bins, pys1, pxs, label, b, b1,
            len_unique_a, p_YgX, unique_inverse_x, unique_inverse_y)

    params = {}
    params['ixt'] = IXT
    params['ity'] = ITY
    params['epoch'] = e
    params['layer'] = l
    return params

def calc_entropy_for_specipic_t(current_ts, px_i):
    """Calc entropy for specipic t"""
    b2 = np.ascontiguousarray(current_ts).view(
        np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False,
                  return_inverse=True, return_counts=True)
    p_current_ts = unique_counts / float(sum(unique_counts))
    p_current_ts = np.asarray(p_current_ts, dtype=np.float64).T
    H2X = px_i * (-np.sum(p_current_ts * np.log2(p_current_ts)))
    return H2X

def calc_condtion_entropy(px, t_data, unique_inverse_x):
    H2X_array = np.array(
        Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(calc_entropy_for_specipic_t)(t_data[unique_inverse_x == i, :], px[i])
                for i in range(px.shape[0])))
    H2X = np.sum(H2X_array)
    return H2X

def calc_information_from_mat(px, py, ps2, data, unique_inverse_x,
                              unique_inverse_y, unique_array):
    H2 = -np.sum(ps2 * np.log2(ps2))
    H2X = calc_condtion_entropy(px, data, unique_inverse_x)
    H2Y = calc_condtion_entropy(py.T, data, unique_inverse_y)
    IX = H2 - H2X
    IY = H2 - H2Y
    return IX, IY

def calc_information_sampling(data, bins, pys1, pxs, label, b, b1,
                              len_unique_a, p_YgX, unique_inverse_x,
                              unique_inverse_y):
    bins = bins.astype(np.float32)
    nbins = bins.shape[0]
    digitized = bins[np.digitize(
        np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
    b2 = np.ascontiguousarray(digitized).view(
        np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False,
                  return_inverse=True, return_counts=True)
    p_ts = unique_counts / float(sum(unique_counts))
    PXs, PYs = np.asarray(pxs).T, np.asarray(pys1).T
    if False:
        pxy_given_T = np.array(
            [calc_probs(i, unique_inverse_t, label, b, b1, len_unique_a) for i in range(0, len(unique_array))]
        )
        p_XgT = np.vstack(pxy_given_T[:, 0])
        p_YgT = pxy_given_T[:, 1]
        p_YgT = np.vstack(p_YgT).T
        DKL_YgX_YgT = np.sum(
            [KL(c_p_YgX, p_YgT.T) for c_p_YgX in p_YgX.T], axis=0)
        H_Xgt = np.nansum(p_XgT * np.log2(p_XgT), axis=1)
    IXT, ITY = calc_information_from_mat(PXs, PYs, p_ts, digitized,
                                         unique_inverse_x, unique_inverse_y,
                                         unique_array)
    return IXT, ITY

def extract_probs(label, x):
    pys = np.sum(label, axis=0) / float(label.shape[0])
    b = np.ascontiguousarray(x).view(
        np.dtype((np.void, x.dtype.itemsize * x.shape[1])))

    unique_array, unique_indices, unique_inverse_x, unique_counts = \
        np.unique(b, return_index=True, return_inverse=True, return_counts=True)
    unique_a = x[unique_indices]

    b1 = np.ascontiguousarray(unique_a).view(
        np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))

    pxs = unique_counts / float(np.sum(unique_counts))

    p_y_given_x = []
    for i in range(0, len(unique_array)):
        indexs = unique_inverse_x == i
        py_x_current = np.mean(label[indexs, :], axis=0)
        p_y_given_x.append(py_x_current)
    p_y_given_x = np.array(p_y_given_x).T

    b_y = np.ascontiguousarray(label).view(
        np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
    unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
        np.unique(b_y, return_index=True,
                  return_inverse=True, return_counts=True)
    pys1 = unique_counts_y / float(np.sum(unique_counts_y))
    return pys, pys1, p_y_given_x, b1, b, unique_a, unique_inverse_x, unique_inverse_y, pxs
