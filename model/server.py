import torch

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model.lenet import lenet5


class server(object):
    def __init__(self, size, n_epoch, dataset):
        # seed
        seed = 19201077 + 920
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # hyper parameter
        self.size = size
        self.n_epoch = n_epoch
        self.temperature = 5

        # global model
        self.global_model = self._save_global_model_state()

        # data loader
        self.test_loader = DataLoader(dataset[1], batch_size=128, shuffle=False)
        self.trans_loader = DataLoader(dataset[2], batch_size=128, shuffle=False)

        # accuracy
        self.accuracy = []

    @staticmethod
    def _save_global_model_state():
        model = lenet5().cuda()
        torch.save(model.state_dict(), './cache/global_model_state.pkl')
        return model

    def _load_soft_target(self):
        soft_targets = []
        for s in range(self.size):
            soft_targets.append(torch.load('./cache/soft_target_{}.pkl'.format(s)))
        return soft_targets

    def _aggregate_soft_target(self, soft_targets):
        sum_soft_targets = torch.zeros(soft_targets[0].shape).cuda()
        for s in range(self.size):
            sum_soft_targets += soft_targets[s]
        aggregate_soft_targets = sum_soft_targets / self.size
        return aggregate_soft_targets.split(128, dim=0)

    def _knowledge_distillation(self, soft_target, temperature):
        optimizer = optim.Adam(params=self.global_model.parameters(), lr=0.001)
        for i in range(self.n_epoch):
            self.global_model.train()
            for index, (data, _) in enumerate(self.trans_loader):
                data = Variable(data).cuda()
                optimizer.zero_grad()

                output = self.global_model(data)

                output_student = torch.nn.functional.log_softmax(output / temperature, dim=1)
                output_teacher = torch.nn.functional.softmax(soft_target[index] / temperature, dim=1)
                loss = nn.KLDivLoss()(output_student, output_teacher) * temperature * temperature

                loss.backward()
                optimizer.step()
                index += 1

            test_loss = 0
            test_correct = 0
            self.global_model.eval()
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = Variable(data).cuda(), Variable(target).cuda()

                    output = self.global_model(data)

                    test_loss += nn.CrossEntropyLoss()(output, target).item()
                    test_loss /= len(self.test_loader.dataset)
                    pred = output.argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()

            print('[Epoch {:>3}]  test_loss: {:.6f},  test_accuracy: {:.4f}'.format(
                i + 1,
                test_loss,
                test_correct / len(self.test_loader.dataset)
            ))
            self.accuracy.append(test_correct / len(self.test_loader.dataset))

    def run(self):
        soft_target = self._aggregate_soft_target(self._load_soft_target())
        self._knowledge_distillation(soft_target=soft_target, temperature=self.temperature)
        self._save_global_model_state()
