import torch

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model.lenet import lenet5


class client(object):
    def __init__(self, rank, n_iter, dataset):
        # seed
        seed = 19201077 + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # rank
        self.rank = rank

        # hyper parameter
        self.n_iter = n_iter

        # local model
        self.model = self._load_global_model_state()

        # data loader
        self.train_loader = DataLoader(dataset[0], batch_size=128, shuffle=True)
        self.test_loader = DataLoader(dataset[1], batch_size=128, shuffle=False)
        self.trans_loader = DataLoader(dataset[2], batch_size=10000, shuffle=False)

    @staticmethod
    def _load_global_model_state():
        model = lenet5().cuda()
        model.load_state_dict(torch.load('./cache/global_model_state.pkl'))
        return model

    def _train(self, optimizer):
        for i in range(self.n_iter):
            self.model.train()
            for data, target in self.train_loader:
                data, target = Variable(data).cuda(), Variable(target).cuda()

                optimizer.zero_grad()
                output = self.model(data)

                loss = nn.CrossEntropyLoss()(output, target)

                loss.backward()
                optimizer.step()

            test_loss = 0
            test_correct = 0
            self.model.eval()
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = Variable(data).cuda(), Variable(target).cuda()

                    output = self.model(data)

                    test_loss += nn.CrossEntropyLoss()(output, target).item()
                    test_loss /= len(self.test_loader.dataset)

                    pred = output.argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()

            print('[Epoch {:>3}]  test_loss: {:.6f},   test_accuracy: {:.4f}'.format(
                i + 1,
                test_loss,
                test_correct / len(self.test_loader.dataset)
            ))

    def _predict(self):
        with torch.no_grad():
            self.model.eval()
            for data, _ in self.trans_loader:
                data = Variable(data).cuda()
                prediction = self.model(data)
        return prediction

    def _save_soft_target(self, soft_target):
        torch.save(soft_target, './cache/soft_target_{}.pkl'.format(self.rank))

    def run(self):
        self.model = self._load_global_model_state()
        optimizer = optim.SGD(params=self.model.parameters(), lr=0.01, momentum=0.5)
        # optimizer = optim.Adam(params=self.model.parameters(), lr=0.001)
        self._train(optimizer=optimizer)
        self._save_soft_target(self._predict())
