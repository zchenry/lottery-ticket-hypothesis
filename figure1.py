#!/usr/bin/env python

import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import TupleDataset
import numpy as np


class MLP(chainer.Chain):
    def __init__(self, H, i, I, db, zl):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(2, H)
            self.l2 = L.Linear(H, 1)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', nargs='+',
                        type=int, default=[2, 4, 6, 8, 10])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--minloss', type=float, default=1e-7)
    parser.add_argument('--patient', type=int, default=7)
    args = parser.parse_args()

    for h in args.hidden:
        db = 0
        zl = 0
        sl = 0
        for i in range(args.iter):
            model = MLP(h, i, args.iter, db, zl)

            if args.gpu >= 0:
                chainer.backends.cuda.get_device_from_id(args.gpu).use()
                model.to_gpu()

            optimizer = chainer.optimizers.SGD(1.)
            optimizer.setup(model)

            train = model.xp.array([0, 0, 1, 1],
                                   dtype=model.xp.int32)[:, None]

            epoch = 0
            loss_diff = 100
            old_loss = None
            patient = 1
            model.set_xy()
            for epoch in range(args.epoch):
                loss = model.loss(model.x, train)

                model.cleargrads()
                loss.backward()
                optimizer.update()
                print(' ' * 100, end='\r')
                print('H = {} ITER {} / {} iter, DB {:.3f},'\
                      ' ZL {:.3f}, SL {:.3f}, LOSS {}, {}'.format(
                          h, i, args.iter, db / args.iter, zl / args.iter,
                          sl / args.iter, loss.data, epoch), end='\r')

                loss.to_cpu()
                if old_loss is None:
                    old_loss = loss.data
                else:
                    loss_diff = np.abs(old_loss - loss.data)
                    old_loss = loss.data

                if loss_diff < args.minloss:
                    if patient > args.patient:
                        break
                    else:
                        patient += 1
                else:
                    patient = 0

            if model.if_db():
                db += 1
            if loss.data == 0.:
                zl += 1
            if loss.data < args.minloss:
                sl += 1

        print(' ' * 100, end='\r')
        print('H {}: {} iter, DB {:.3f}, ZL {:.3f}, SL {:.3f}'.format(
            h, args.iter, db / args.iter, zl / args.iter, sl / args.iter))


if __name__ == '__main__':
    main()
