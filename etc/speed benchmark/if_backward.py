from typing import Callable

import torch
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.neuron import IFNode
from torch import nn


class IFIMP(IFNode):
    """
    >>>inputs = torch.randn((2, 1, 4))
    >>>node = IFIMP(inputs.shape)
    >>>out = node(inputs)
    >>>print(node.v)
    tensor([[[ 0.1676, -0.3076, -0.1530, -0.1675]],
            [[-0.0658, -1.4495, -0.3014, -0.2170]]])
    >>>node.reset()
    >>>print(node.v)
    tensor([[[0.4139, 0.1390, 0.8201, 0.3612]],
            [[0.3644, 0.9767, 0.0484, 0.7073]]], grad_fn=<AddBackward0>)
    """

    def __init__(self, init_state_shape=(1, 1, 1, 1), v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False, sigmoid_init=False):
        """
        :param init_state_shape: (B, C, H, W) or (B, N, D)
                        x.shape: (T, B, C, H, W) or (T, B, N, D)
        """
        super(IFIMP, self).__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.init_state = nn.Parameter(nn.init.uniform_(torch.empty(init_state_shape), a=-0.2, b=0.2))
        self.init_func = torch.sigmoid if sigmoid_init else lambda x: x
        self.v += self.init_func(self.init_state)

    def reset(self):
        super(IFIMP, self).reset()
        self.v += self.init_func(self.init_state)

    def forward(self, *args, **kwargs):
        x = args[0]
        if self.step_mode == 's':
            self.v = torch.broadcast_to(self.v, x.shape).to(x)
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            self.v = torch.broadcast_to(self.v, x[0].shape).to(x)
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)


def speed_bench(shape: tuple):
    org_if = IFNode(backend='cupy', step_mode='m').cuda()
    sta_if = IFIMP(shape[1:], backend='cupy', step_mode='m').cuda()

    # ===================== IFNode =====================
    x = torch.randn(shape, requires_grad=True).cuda()
    y = torch.randn(shape, requires_grad=True).cuda()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    org_if.reset()
    out = org_if(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    org_if.reset()
    out = org_if(x)
    for _ in range(9):
        out = out + org_if(out)
    loss = torch.nn.functional.mse_loss(out, y)
    torch.cuda.synchronize()
    start_time.record()
    loss.backward()
    end_time.record()
    torch.cuda.synchronize()
    print(f'IFNode\t{shape} \t: {start_time.elapsed_time(end_time) / 10} ms')
    # =================== IF_IMP ===================
    x = torch.randn(shape, requires_grad=True).cuda()
    y = torch.randn(shape, requires_grad=True).cuda()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    sta_if.reset()
    out = sta_if(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    sta_if.reset()
    out = sta_if(x)
    for _ in range(9):
        out = out + sta_if(out)
    loss = torch.nn.functional.mse_loss(out, y)
    torch.cuda.synchronize()
    start_time.record()
    loss.backward()
    end_time.record()
    torch.cuda.synchronize()
    print(f'IFIMP\t{shape} \t: {start_time.elapsed_time(end_time) / 10} ms')


if __name__ == '__main__':
    shape_list = [
        (2, 2 ** 8), (2, 2 ** 12), (2, 2 ** 16), (2, 2 ** 20),
        (4, 2 ** 8), (4, 2 ** 12), (4, 2 ** 16), (4, 2 ** 20),
        (8, 2 ** 8), (8, 2 ** 12), (8, 2 ** 16), (8, 2 ** 20),
        (16, 2 ** 8), (16, 2 ** 12), (16, 2 ** 16), (16, 2 ** 20),
        (32, 2 ** 8), (32, 2 ** 12), (32, 2 ** 16), (32, 2 ** 20),
    ]
    for test_shape in shape_list:
        speed_bench(test_shape)
