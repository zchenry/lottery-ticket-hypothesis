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

def train_model(model, train, test, s, I, minacc, patient):
    optimizer = chainer.optimizers.SGD(0.05)
    optimizer.setup(model)
    weights = extract_weights(model)
    train_iter = chainer.iterators.SerialIterator(train, 100)
    test_xs, test_ys = tuple2array(test, model.xp)

    old_acc = None
    acc_diff = 100
    p = 0
    t = 0
    times = []
    accs = []
    its = []
    train_xs, train_ys = tuple2array(train_iter.next(), model.xp)
    for i in range(I):
        start_time = time.time()
        loss = model.loss(train_xs, train_ys)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        model = reset_zeros(model, weights)
        t += time.time() - start_time

        acc = F.accuracy(model.forward(test_xs), test_ys)
        acc.to_cpu()
        acc = acc.data[()]
        pr('{}{:.5}%: ITER {}, TIME {:.2f}s, ACC {:.10f}'.format(
            s, model.p, i, t, acc))
        if old_acc is None:
            old_acc = acc
        else:
            acc_diff = np.abs(old_acc - acc)
            old_acc = acc
        accs.append(acc)
        times.append(t)
        its.append(i)
        if acc_diff < minacc:
            if p > patient:
                pass
            else:
                p += 1
        else:
            p = 0
    print('{}{:.5}%: ITER {}, TIME {:.2f}s, ACC {:.7f}'.format(
        s, model.p, i, t, acc))
    return accs, its, times

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

def run_mnist(P, gpu, *args):
    weights = initial_weights()
    train, test = chainer.datasets.get_mnist()
    model_p = 100.
    to_plot = {}
    for i, p in enumerate(P + [0]):
        model = prepare(gpu, weights, model_p)
        acc, it, t = train_model(model, train, test, 'Prune: ', *args)
        dic = {}
        dic['accuracies'] = acc
        dic['iters'] = it
        dic['times'] = t
        to_plot['Prun: {:.5}%'.format(model_p)] = dic

        _weights = shuffle_weights(weights)
        _model = prepare(gpu, _weights, model_p)
        cacc, cit, ct = train_model(_model, train,
                                    test, 'Shuffle: ', *args)
        dic = {}
        dic['accuracies'] = cacc
        dic['iters'] = cit
        dic['times'] = ct
        to_plot['Shuffle {:.5}%'.format(model_p)] = dic

        _weights = initial_weights()
        _weights = map_zeros(_weights, weights)
        w1 = _weights['w1']
        out = sum(np.abs(w1).sum(1) > 0)
        for wi in range(w1.shape[0]):
            inn = sum(w1[wi] != 0)
            rg = 0.7 * np.power(out, 1./inn)
            if np.sum(w1[wi] ** 2) > 0:
                w1[wi] *= rg / np.sqrt( np.sum(w1[wi] ** 2) )
        _weights['w1'] = w1

        w2 = _weights['w2']
        out = sum(np.abs(w2).sum(1) > 0)
        for wi in range(w2.shape[0]):
            inn = sum(w2[wi] != 0)
            rg = 0.7 * np.power(out, 1./inn)
            if np.sum(w2[wi] ** 2) > 0:
                w2[wi] *= rg / np.sqrt( np.sum(w2[wi] ** 2) )
        _weights['w2'] = w2
        _weights['b1'] = np.ones(_weights['b1'].shape)
        _weights['b2'] = np.ones(_weights['b2'].shape)

        _model = prepare(gpu, _weights, model_p)
        cacc, cit, ct = train_model(_model, train,
                                    test, 'My: ', *args)
        dic = {}
        dic['accuracies'] = cacc
        dic['iters'] = cit
        dic['times'] = ct
        to_plot['My {:.5}%'.format(model_p)] = dic

        if i < len(P):
            model_p = model_p * (100 - p) / 100
            weights['w1'] = prune(weights['w1'], p)
            weights['w2'] = prune(weights['w2'], p)
            weights['w3'] = prune(weights['w3'], p / 2)

    plt.clf()
    for key, value in to_plot.items():
        plt.plot(value['times'], value['accuracies'], label=key)
    plt.legend()
    plt.xlabel('Wall Clock Time (s)')
    plt.ylabel('Accuracy')
    plt.savefig('time-accuracy.png')

    plt.clf()
    for key, value in to_plot.items():
        plt.plot(value['iters'], value['accuracies'], label=key)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig('iter-accuracy.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--percent', nargs='+', type=int, default=[85, 85])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--minacc', type=float, default=1e-3)
    parser.add_argument('--patient', type=int, default=3)
    args = parser.parse_args()

    run_mnist(args.percent, args.gpu, args.iter, args.minacc, args.patient)
