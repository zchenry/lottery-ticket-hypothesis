This is the **Chainer** implementation for the paper [The Lottery Ticket Hypothesis: Training Pruned Neural Network Architectures](https://arxiv.org/abs/1803.03635).

#### Figure 1
`python figure1.py --hidden 2 4 6 8 10 --iter 1000 --epoch 10000 --minloss 1e-7 --patient 7`

- hidden: Number list for the hidden layer.
- iter: Number of iterations for each number of hidden layer.
- epoch: Number of epochs for one iteration.
- minloss: Minimal change of loss for early stopping.
- patient: Counts to stop.
