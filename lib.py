import pdb
import chainer
import argparse
import numpy as np
import chainer.links as L
import chainer.functions as F

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
            self.l3 = L.Linear(100, 10,
                               initialW=weights['w3'],
                               initial_bias=weights['b3'])
            self.p = p

    def forward(self, x):
        return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))

    def loss(self, x, y):
        loss = F.softmax_cross_entropy(self.forward(x), y)
        return loss
