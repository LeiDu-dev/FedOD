import torch

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader


class loader(object):
    def __init__(self, cmd='cifar10', trans_size=1e4):
        self.cmd = cmd
        self.trans_size = int(trans_size)
        self.__load_dataset()
        self.__get_index()

    def __load_dataset(self):

        # mnist
        self.train_mnist = datasets.MNIST('./dataset/',
                                          train=True,
                                          download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                          ]))

        self.test_mnist = datasets.MNIST('./dataset/',
                                         train=False,
                                         download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                         ]))

        # cifar10
        self.train_cifar10 = datasets.CIFAR10('./dataset/',
                                              train=True,
                                              download=True,
                                              transform=transforms.Compose([
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                              ]))
        self.test_cifar10 = datasets.CIFAR10('./dataset/',
                                             train=False,
                                             download=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                             ]))

    def __get_index(self):

        if self.cmd == 'cifar10':
            self.train_size = 50000 - self.trans_size
            self.train_dataset = torch.utils.data.Subset(self.train_cifar10, [i for i in range(0, self.train_size)])
            self.trans_dataset = torch.utils.data.Subset(self.train_cifar10, [i for i in range(self.train_size, 50000)])
            self.test_dataset = self.test_cifar10
        else:
            self.train_size = 60000 - self.trans_size
            self.train_dataset = torch.utils.data.Subset(self.train_mnist, [i for i in range(0, self.train_size)])
            self.trans_dataset = torch.utils.data.Subset(self.train_mnist, [i for i in range(self.train_size, 60000)])
            self.test_dataset = self.test_mnist

        self.indices = [[], [], [], [], [], [], [], [], [], []]
        for index, data in enumerate(self.train_dataset.dataset):
            self.indices[data[1]].append(index)

    def get_dataset(self, rank):
        dataset_indices = []
        difference = list(set(range(10)).difference(set(rank)))
        for i in difference:
            dataset_indices.extend(self.indices[i])

        dataset = torch.utils.data.Subset(self.train_cifar10, dataset_indices)
        if self.cmd != 'cifar10':
            dataset = torch.utils.data.Subset(self.train_mnist, dataset_indices)

        return dataset, self.test_dataset, self.trans_dataset
