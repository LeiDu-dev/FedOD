import random

from model.data import loader
from model.server import server
from model.client import client
from model.plot import plot


def federated_learning():
    # dataset
    print('Initialize Dataset...')
    data_loader = loader('mnist')
    # data_loader = loader('cifar10', batch_size=batch_size)

    # hyper parameter
    size = 20
    n_epoch_train = 10
    n_epoch_distillation = 100

    # initialize server
    print('Initialize Server...')
    s = server(size=size, n_epoch=n_epoch_distillation, dataset=data_loader.get_dataset([]))

    # initialize client
    print('Initialize Client...')
    clients = []
    for i in range(size):
        clients.append(client(rank=i, n_iter=n_epoch_train, dataset=data_loader.get_dataset(
            random.sample(range(0, 10), 4)
        )))

    # federated learning
    for index, c in enumerate(clients):
        print('\n======================== Rank {:>2} ========================'.format(index))
        c.run()
    print('\n\n======================== Fed_OD ========================')
    s.run()

    # plot
    plot(s.accuracy)


if __name__ == '__main__':
    federated_learning()
