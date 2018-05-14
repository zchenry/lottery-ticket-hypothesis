#!/usr/bin/env python

import argparse
from chainer.datasets import TupleDataset
import numpy as np
from lib import *

def extract_weights(model):
    weights = {}
    weights['w1'] = model.l1.W.data[()]
    weights['w2'] = model.l2.W.data[()]
    return weights

def prepare(h, gpu, weights):
    model = MLP(h, weights)
    if weights is None:
        weights = extract_weights(model)

    if gpu >= 0:
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    model.set_xy()
    train_y = model.xp.array([0, 0, 1, 1],
                             dtype=model.xp.int32)[:, None]
    return model, weights, model.x, train_y

def train_model(model, train_x, train_y, epoch, minloss, patient):
    optimizer = chainer.optimizers.SGD(1.)
    optimizer.setup(model)

    old_loss = None
    loss_diff = 100
    p = 0
    for e in range(epoch):
        loss = model.loss(train_x, train_y)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        loss.to_cpu()
        if old_loss is None:
            old_loss = loss.data
        else:
            loss_diff = np.abs(old_loss - loss.data)
            old_loss = loss.data

        if loss_diff < minloss:
            if p > patient:
                break
            else:
                p += 1
        else:
            p = 0
    return model, loss.data[()]

def one_iter(H, gpu, strategy, *args):
    db = np.zeros(len(H))
    zl = np.zeros(len(H))
    weights = None
    for i, h in enumerate(H):
        model, weights, train_x, train_y = prepare(h, gpu, weights)

        new_model, loss = train_model(model, train_x, train_y, *args)
        db[i] = 1 if new_model.if_db() else 0
        zl[i] = 1 if loss == 0. else 0

        if i < len(H) - 1:
            if strategy == 'in':
                importance = np.mean(weights['w1'], 1)
            elif strategy == 'out':
                importance = weights['w2'][0]
            elif strategy == 'pro':
                importance = (np.sum(weights['w1'], 1) * weights['w2'])[0]

            indices = np.argsort(importance)[::-1]
            weights['w1'] = weights['w1'][indices[:H[i + 1]]]
            weights['w2'] = weights['w2'][0][indices[:H[i + 1]]][None, :]

    return db, zl

def run_xor(H, I, *args):
    db = np.zeros(len(H))
    zl = np.zeros(len(H))

    for i in range(I):
        _db, _zl = one_iter(H, *args)
        db += _db
        zl += _zl
        pr('Iter {}: DB: {}, ZL: {}'.format(i + 1, db, zl))

    pr('')
    print('{} Iterations'.format(I))
    for i in range(len(H)):
        print('H {}: DB {:.2f}, ZL {:.2f}'.format(H[i], db[i]/I, zl[i]/I))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', nargs='+',
                        type=int, default=[8, 4, 2])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--minloss', type=float, default=1e-7)
    parser.add_argument('--patient', type=int, default=7)
    parser.add_argument('--strategy', type=str, default='pro') # or in or out
    args = parser.parse_args()

    run_xor(args.hidden, args.iter, args.gpu, args.strategy,
            args.epoch, args.minloss, args.patient)
