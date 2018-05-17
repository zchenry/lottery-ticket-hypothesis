#!/usr/bin/env python
from mpl_toolkits.mplot3d.axes3d import Axes3D
from lib import *
import cupy as cp

class FM(chainer.Chain):
    def __init__(self, hs, fast=False):
        super(FM, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(784, hs[0])
            self.l2 = L.Linear(hs[0], hs[1])
            self.l3 = L.Linear(hs[1], 10, nobias=True)
            if fast:
                self.good_weights(hs)

    def good_weights(self, hs):
        self.l1.b.data = self.xp.ones(self.l1.b.shape, dtype=self.xp.float32)
        self.l2.b.data = self.xp.ones(self.l2.b.shape, dtype=self.xp.float32)

        self.l1.W.data = self.good_weight(784, hs[0], self.l1.b.data)
        self.l2.W.data = self.good_weight(hs[0], hs[1], self.l2.b.data)

    def good_weight(self, inn, out, b):
        w = self.xp.random.normal(0, 0.1, (out, inn))
        for i in range(out):
            rg = self.xp.power(out, 1 / inn) * b[i] * 0.8
            w[i] *= np.sqrt(rg / (self.xp.sum(w[i] ** 2) * len(w[i])))
        return w.astype(self.xp.float32)

    def forward(self, x):
        return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))

    def loss(self, x, y):
        return F.softmax_cross_entropy(self.forward(x), y)

    def accuracy(self, x, y):
        return F.accuracy(self.forward(x), y)

    def vis(self, xs):
        return (F.relu(self.l2(xs))).data

def train_model(model, i, I, E, train, xs, ys, s):
    optimizer = chainer.optimizers.SGD(0.05)
    optimizer.setup(model)
    train_iter = chainer.iterators.SerialIterator(train, 128)

    t, times, accs, es = 0, [], [], []
    for e in range(E):
        start_time = time.time()
        xs, ys = tuple2array(train_iter.next(), model.xp)
        loss = model.loss(xs, ys)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        t += time.time() - start_time

        acc = model.accuracy(xs, ys)
        acc.to_cpu()
        acc = acc.data[()]
        accs.append(acc)
        times.append(t)
        es.append(e)
        pr('{}/{} {} EPOCH {}, TIME {:.5f}s, ACC {:.10f}'.format(
            i, I, s, e, t, acc))
    pr('')
    return times, es, accs, model

def compare(hs, I, E):
    chainer.backends.cuda.get_device_from_id(0).use()
    train, test = chainer.datasets.get_mnist()
    xs, ys = tuple2array(test, cp)
    Ts, Es, As = np.zeros((I, E)), np.zeros((I, E)), np.zeros((I, E))
    _Ts, _Es, _As = np.zeros((I, E)), np.zeros((I, E)), np.zeros((I, E))
    for i in range(I):
        times, es, accs, m = train_model(FM(hs, fast=True).to_gpu(),
                                         i, I, E, train, xs, ys, 'Fast')
        _times, _es, _accs, _ = train_model(FM(hs).to_gpu(),
                                            i, I, E, train, xs, ys, 'Slow')
        Ts[i] = np.array(times)
        Es[i] = np.array(es)
        As[i] = np.array(accs)
        _Ts[i] = np.array(_times)
        _Es[i] = np.array(_es)
        _As[i] = np.array(_accs)
        '''
        _x, _y, xs = mesh_inputs(-1, 1, 50)
        zs = m.to_cpu().vis(xs)
        cs = ['red', 'c', 'g', 'm', 'blue', 'black', 'yellow']
        cls, xs, ys = [], [], []
        for _i in range(len(_x)):
            for _j in range(len(_y)):
                xs.append(_x[_i])
                ys.append(_y[_j])
                cls.append(cs[np.argmax(zs[(_i * 50 + _j), :])])
        plt.clf()
        plt.scatter(x=xs, y=ys, color=cls, marker='s')
        plt.savefig('surface_iter{}.png'.format(i))
        '''
    plt.clf()
    sns.tsplot(As, time=_Ts.mean(0), condition='fast')
    sns.tsplot(_As, time=_Ts.mean(0), condition='normal', color='g')
    plt.savefig('time.png')
    plt.clf()
    sns.tsplot(As, time=Es.mean(0), condition='fast')
    sns.tsplot(_As, time=_Es.mean(0), condition='normal', color='g')
    plt.savefig('epoch.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', nargs='+', type=int, default=[2, 5])
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--iter', type=int, default=10)
    args = parser.parse_args()

    compare(args.hidden, args.iter, args.epoch)
