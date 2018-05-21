#!/usr/bin/env python

import time
from lib import *
from chainer.iterators import SerialIterator

def inits(d1, d2, s, mu=0., sigma=0.1):
    samples = np.random.normal(mu, sigma, d1 * d2)
    outliers = np.abs(samples) > sigma * 2
    while outliers.any():
        pr('{} {}/{} left'.format(s, sum(outliers), d1 * d2))
        samples[outliers] = np.random.normal(mu, sigma, sum(outliers))
        outliers = np.abs(samples) > sigma * 2
    return samples.reshape((d2, d1))

def initial_weights():
    d1, d2, d3, d4 = 784, 300, 100, 1
    weights = {}
    s = 'Initializing weights.'
    weights['w1'] = inits(d1, d2, s)
    weights['b1'] = np.zeros((300,))
    weights['w2'] = inits(d2, d3, s + '.')
    weights['b2'] = np.zeros((100,))
    weights['w3'] = inits(d3, d4, s + '..')
    pr(' ')
    return weights

def prepare(gpu, weights, p):
    model = MNIST(weights, p)
    if gpu >= 0:
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()
    return model

def extract_weights(model):
    weights = {}
    weights['w1'] = model.l1.W.data[()]
    weights['b1'] = model.l1.b.data[()]
    weights['w2'] = model.l2.W.data[()]
    weights['b2'] = model.l2.b.data[()]
    weights['w3'] = model.l3.W.data[()]
    return weights

def shuffle_weight(w):
    _w = w
    if len(w.shape) == 1:
        iss = np.nonzero(w)[0]
        if len(iss) > 0:
            _extract = np.extract(w != 0, w)
            np.random.shuffle(_extract)
            for idx in range(len(iss)):
                _w[ iss[idx] ] = _extract[idx]
    else:
        iss, jss = np.nonzero(w)
        if len(iss) > 0:
            _extract = np.extract(w != 0, w)
            np.random.shuffle(_extract)
            for idx in range(len(iss)):
                _w[ iss[idx], jss[idx] ] = _extract[idx]
    return _w

def shuffle_weights(weights):
    ws = {}
    for key in weights.keys():
        ws[key] = shuffle_weight(weights[key])
    return ws

def reset_zeros(model, weights):
    model.l1.W.data[weights['w1'] == 0] = 0
    model.l1.b.data[weights['b1'] == 0] = 0
    model.l2.W.data[weights['w2'] == 0] = 0
    model.l2.b.data[weights['b2'] == 0] = 0
    model.l3.W.data[weights['w3'] == 0] = 0
    return model

def map_zeros(_weights, weights):
    _weights['w1'][weights['w1'] == 0] = 0
    _weights['b1'][weights['b1'] == 0] = 0
    _weights['w2'][weights['w2'] == 0] = 0
    _weights['b2'][weights['b2'] == 0] = 0
    _weights['w3'][weights['w3'] == 0] = 0
    return _weights

def train_model(model, train, test_xs, test_ys, s, epochs):
    optimizer = chainer.optimizers.SGD(0.05)
    optimizer.setup(model)
    weights = extract_weights(model)
    train_iter = chainer.iterators.SerialIterator(train, 100)

    accs, its, ts, tes = [], [], [], []
    for epoch in range(epochs):
        train_xs, train_ys = tuple2array(train_iter.next(), model.xp)
        loss = model.loss(train_xs, train_ys)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        model = reset_zeros(model, weights)

        idx = (np.random.rand(1000) * len(test_xs)).astype(np.int)
        pred, hiddens_gpu = model.forward(test_xs[idx], keep_val=True)
        acc = F.accuracy(pred, test_ys[idx])
        acc.to_cpu()
        acc = acc.data[()]
        pr('{}{:.5}%: EPOCH {}, ACC {:.10f}'.format(
            s, model.p, epoch, acc))
        accs.append(acc)
        its.append(epoch)

        if (epoch + 1) % 10 == 0:
            hiddens = []
            for hidden in hiddens_gpu:
                hidden.to_cpu()
                hiddens.append(hidden.data[()])
            ts.append(hiddens)
            tes.append(epoch)

    print('{}{:.5}%: EPOCH {}, ACC {:.7f}'.format(
        s, model.p, epoch, acc))
    return accs, its, ts, tes

def prune(ws, p):
    d1 = ws.shape[0]
    d2 = ws.shape[1]
    values = np.abs(ws.flatten())
    values = np.extract(values > 0, values)
    pivot = np.sort(values)[int(len(values) * p / 100)]
    ws[np.abs(ws) < pivot] = 0.
    return ws

def plot_weights(weights, percent):
    def plot_weight(key):
        data = weights[key]
        ws = []
        for di in range(data.shape[0]):
            ws.append(np.sqrt(np.sum(data[di] ** 2)))
        y, xs = np.histogram(ws, bins=100)
        xs = 0.5 * (xs[1:] + xs[:-1])
        plt.plot(xs, y, '-')

    plt.clf()
    for key in ['w1', 'w2', 'w3']:
        plot_weight(key)
    plt.savefig('{}P_weights.png'.format(percent))

def plot_points(points, es, feature):
    pr('Plotting {}...'.format(feature))
    _xs = np.array([p['ixt'] for p in points])
    _ys = np.array([p['ity'] for p in points])
    _ls = np.array([p['layer'] for p in points])

    plt.clf()
    for l in range(2):
        mask = _ls == l
        plt.plot(es, _xs[mask], label=str(l))
    plt.legend()
    plt.savefig('{}_x.png'.format(feature))

    plt.clf()
    for l in range(2):
        mask = _ls == l
        plt.plot(es, _ys[mask], label=str(l))
    plt.legend()
    plt.savefig('{}_y.png'.format(feature))

def run_mnist(P, gpu, epochs, analysis):
    weights = initial_weights()
    train, test = chainer.datasets.get_mnist()
    model_p = 100.
    to_plot = {}
    bins = np.linspace(-1, 1, 30)
    if gpu >= 0:
        import cupy as xp
    else: xp = np
    if analysis >= 0:
        import multiprocessing
        from joblib import Parallel, delayed
    test_xs, test_ys = tuple2array(test, xp)
    for i, p in enumerate(P + [0]):
        model = prepare(gpu, weights, model_p)
        acc, it, ts, tes = train_model(model, train, test_xs,
                                       test_ys, 'Prune: ', epochs)
        dic = {}
        dic['accuracies'] = acc
        dic['iters'] = it
        to_plot['Prun: {:.5}%'.format(model_p)] = dic

        cxs = to_cpu(test_xs)
        cys = to_cpu(test_ys)[:, None]

        if analysis >= 0:
            idx = (np.random.rand(1000) * len(test_xs)).astype(np.int)
            idy = (np.random.rand(1000) * len(test_ys)).astype(np.int)
            hsic_points = calc_hsics(cxs[idx], cys[idy], ts)
            plot_points(hsic_points, tes, 'l_prun_{}_hsic'.format(model_p))
            mi_points = calc_mi(cxs[idx], cys[idy], ts, bins)
            plot_points(mi_points, tes, 'l_prun_{}_mi'.format(model_p))

        if i > 0:
            _weights = shuffle_weights(weights)
            _model = prepare(gpu, _weights, model_p)
            cacc, cit,  cts, ctes = train_model(_model, train,
                                        test_xs, test_ys, 'Shuffle: ', epochs)
            dic = {}
            dic['accuracies'] = cacc
            dic['iters'] = cit
            to_plot['Shuffle {:.5}%'.format(model_p)] = dic

            if analysis >= 0:
                idx = (np.random.rand(1000) * len(test_xs)).astype(np.int)
                idy = (np.random.rand(1000) * len(test_ys)).astype(np.int)
                hsic_points = calc_hsics(cxs[idx], cys[idy], cts)
                plot_points(hsic_points, ctes,
                            'l_shuffle_{}_hsic'.format(model_p))
                mi_points = calc_mi(cxs[idx], cys[idy], cts, bins)
                plot_points(mi_points, ctes, 'l_shuffle_{}_mi'.format(model_p))

            _weights = initial_weights()
            _weights = map_zeros(_weights, weights)

            _model = prepare(gpu, _weights, model_p)
            cacc, cit,  cts, ctes = train_model(_model, train,
                                        test_xs, test_ys, 'Normal: ', epochs)
            dic = {}
            dic['accuracies'] = cacc
            dic['iters'] = cit
            to_plot['Normal {:.5}%'.format(model_p)] = dic

            if analysis >= 0:
                idx = (np.random.rand(1000) * len(test_xs)).astype(np.int)
                idy = (np.random.rand(1000) * len(test_ys)).astype(np.int)
                hsic_points = calc_hsics(cxs[idx], cys[idy], cts)
                plot_points(hsic_points, ctes,
                            'l_normal_{}_hsic'.format(model_p))
                mi_points = calc_mi(cxs[idx], cys[idy], cts, bins)
                plot_points(mi_points, ctes, 'l_normal_{}_mi'.format(model_p))

        if i < len(P):
            model_p = model_p * (100 - p) / 100
            weights['w1'] = prune(weights['w1'], p)
            weights['w2'] = prune(weights['w2'], p)
            weights['w3'] = prune(weights['w3'], p / 2)

    plt.clf()
    for key, value in to_plot.items():
        plt.plot(value['iters'], value['accuracies'], label=key)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('iter-accuracy.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--percent', nargs='+', type=int, default=[])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--analyse', type=int, default=-1)
    args = parser.parse_args()

    run_mnist(args.percent, args.gpu, args.epoch, args.analyse)
