from copy import deepcopy

import torch
from spikingjelly.activation_based import layer, functional, neuron
from torch import nn


class Net(nn.Module):
    def __init__(self, time_step: int = 16):
        super(Net, self).__init__()
        self.time_step = time_step
        self.l1 = layer.Linear(16, 256, bias=False)
        self.sn1 = neuron.IFNode(step_mode='m', v_reset=None, backend='cupy')
        self.l2 = layer.Linear(256, 256, bias=False)
        self.sn2 = neuron.IFNode(step_mode='m', v_reset=None, backend='cupy')
        self.l3 = layer.Linear(256, 256, bias=False)
        self.sn3 = neuron.IFNode(step_mode='m', v_reset=None, backend='cupy')
        self.l4 = layer.Linear(256, 16, bias=False)

    def forward(self, x):
        x = torch.stack([x] * self.time_step, dim=0)
        x = self.sn1(self.l1(x))
        x = self.sn2(self.l2(x))
        x = self.sn3(self.l3(x))
        x = self.l4(x)
        return x


def train_sdt(model, time_step, inputs, labels, epoch):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_sdt = []
    for _ in range(epoch):
        functional.reset_net(model)
        optim.zero_grad()
        out = model(inputs)
        loss = nn.functional.cross_entropy(out.mean(0), labels)
        total_sdt.append(loss.data.cpu().numpy())
        print(total_sdt[-1])
        loss.backward()
        optim.step()

    return total_sdt


def train_tet(model, time_step, inputs, labels, epoch, batch_size):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_tet = []
    for _ in range(epoch):
        functional.reset_net(model)
        optim.zero_grad()
        out = model(inputs)
        loss = nn.functional.cross_entropy(
            out.view(time_step * batch_size, 16),
            torch.cat([labels] * time_step, dim=0))

        with torch.no_grad():
            # total_tet.append(nn.functional.cross_entropy(out.mean(0), labels).data.cpu().numpy())
            total_tet.append(loss.data.cpu().numpy())
        print(total_tet[-1])
        loss.backward()
        optim.step()

    return total_tet


def train_lts(model, time_step, inputs, labels, epoch):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_lst = []
    for _ in range(epoch):
        functional.reset_net(model)
        optim.zero_grad()
        out = model(inputs)
        loss = nn.functional.cross_entropy(out[-1], labels)
        total_lst.append(loss.data.cpu().numpy())
        print(total_lst[-1])
        loss.backward()
        optim.step()

    return total_lst


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    torch.manual_seed(2024)
    time_step = 4
    batch_size = 1024
    epoch = 1000
    model = Net(time_step).cuda()
    inputs = torch.abs(torch.rand((batch_size, 16), device='cuda:0'))
    labels = torch.abs(torch.randint(low=0, high=16, size=(batch_size,), device='cuda:0'))

    sdt_loss = train_sdt(deepcopy(model), time_step, inputs, labels, epoch)
    tet_loss = train_tet(deepcopy(model), time_step, inputs, labels, epoch, batch_size)
    lst_loss = train_lts(deepcopy(model), time_step, inputs, labels, epoch)

    rcParams.update({
        "font.family": 'Times New Roman',
        "axes.unicode_minus": False,
        "font.size": 18
    })
    plt.figure(figsize=(4, 3), dpi=100)
    plt.gcf().subplots_adjust(top=0.90, bottom=0.22, left=0.19, right=0.90)
    ax = plt.axes()

    plt.plot(sdt_loss, label='SDT')
    plt.plot(tet_loss, label='TET')
    plt.plot(lst_loss, label='LTS')

    ax.set_xlabel('Epoch')
    # plt.xlim(-30, epoch + 30)

    ax.set_ylabel('Loss')
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    plt.ylim(-0.15, 3.0)

    plt.legend()
    plt.savefig(f'T={time_step}.svg')
    plt.show()
