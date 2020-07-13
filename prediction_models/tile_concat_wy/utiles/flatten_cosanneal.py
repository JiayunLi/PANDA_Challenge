# !/usr/bin/env python
# -*- coding: utf-8 -*-

# It's written by https://github.com/yumatsuoka

from __future__ import print_function
import math
from torch.optim.lr_scheduler import _LRScheduler


class FlatplusAnneal(_LRScheduler):
    def __init__(self, optimizer, max_iter, step_size=0.7, eta_min=0, last_epoch=-1):
        self.flat_range = int(max_iter * step_size)
        self.T_max = max_iter - self.flat_range
        self.eta_min = 0
        super(FlatplusAnneal, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.flat_range:
            return [base_lr for base_lr in self.base_lrs]
        else:
            cr_epoch = self.last_epoch - self.flat_range
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * (cr_epoch / self.T_max)))
                / 2
                for base_lr in self.base_lrs
            ]


if __name__ == "__main__":

    import torch

    # import matplotlib.pyplot as plt

    def check_scheduler(optimizer, scheduler, epochs):
        lr_list = []
        for epoch in range(epochs):
            now_lr = scheduler.get_lr()
            lr_list.append(now_lr)

            optimizer.step()
            scheduler.step()

        return lr_list

    # def show_graph(lr_lists, epochs):
    #    plt.clf()
    #    plt.rcParams["figure.figsize"] = [20, 5]
    #    x = list(range(epochs))
    #    plt.plot(x, lr_lists, label="line L")
    #    plt.plot()
    #    plt.xlabel("iterations")
    #    plt.ylabel("learning rate")
    #    plt.title("Check Flat plus cosine annealing lr")
    #    plt.show()

    lr = 0.1
    epochs = 100
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = FlatplusAnneal(optimizer, max_iter=epochs, step_size=0.7)

    lrs = check_scheduler(optimizer, scheduler, epochs)
    # show_graph(lrs, epochs)