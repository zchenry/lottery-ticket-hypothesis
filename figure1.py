#!/usr/bin/env python

import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import TupleDataset
from chainer.training import StandardUpdater, Trainer
import numpy as np

class MLP(chainer.Chain):
    def __init__(self, H):
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

    def __call__(self, x, y):
        res = self.forward(x)
        loss = F.sigmoid_cross_entropy(res, y)
        return loss

    def if_db(self):
        return not ((self.y * self.res_y()).data < 0).any()

    def if_zl(self):
        return F.sigmoid_cross_entropy(self.res_y(), self.y).data[()] == 0.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', nargs='+', type=int, default=[2])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--epoch', type=int, default=1000)
    args = parser.parse_args()

    for h in args.hidden:
        db = 0
        zl = 0
        for i in range(args.iter):
            model = MLP(h)

            if args.gpu >= 0:
                chainer.backends.cuda.get_device_from_id(args.gpu).use()
                model.to_gpu()

            model.set_xy()
            optimizer = chainer.optimizers.SGD()
            optimizer.setup(model)
            train = TupleDataset(
                model.xp.array(
                    [[0, 0], [1, 1], [0, 1], [1, 0]], dtype=model.xp.float32),
                model.xp.array([0, 0, 1, 1], dtype=model.xp.int32)[:, None])
            train_iter = chainer.iterators.SerialIterator(train, batch_size=4)

            updater = StandardUpdater(train_iter, optimizer, device=args.gpu)
            trainer = Trainer(updater, (args.epoch, 'epoch'))
            trainer.run()

            if model.if_db():
                db += 1
            if model.if_zl():
                zl += 1

            print('', end='\r')
            print('H = {}: {}/{} iter, DB {:.3f}, ZL {:.3f}'.format(
                h, i, args.iter, db / (i + 1), zl / (i + 1)), end='\r')

        print('', end='\r')
        print('{} hidden: {} iter, DB {:.3f}, ZL {:.3f}'.format(
            h, args.iter, db / args.iter, zl / args.iter))


if __name__ == '__main__':
    main()
