import torch
from torch import nn


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad):
        factor = ctx.alpha - ctx.saved_tensors[0].abs()
        grad *= (1 / ctx.alpha) ** 2 * factor.clamp(min=0)
        return grad, None


class LIFLegacy(nn.Module):
    def __init__(self, thresh=1.0, tau=0.25, gamma=1.0):
        super(LIFLegacy, self).__init__()
        self.heaviside = ZIF.apply
        self.v_th = thresh
        self.tau = tau
        self.gamma = gamma

    def forward(self, x):
        mem_v = []
        mem = 0
        for t in range(x.shape[0]):
            mem = self.tau * mem + x[t, ...]
            spike = self.heaviside(mem - self.v_th, self.gamma)
            mem = mem * (1 - spike)
            mem_v.append(spike)

        return torch.stack(mem_v)


class LIFIMPLegacy(nn.Module):
    def __init__(self, thresh=1.0, tau=0.25, gamma=1.0):
        super(LIFIMPLegacy, self).__init__()
        self.heaviside = ZIF.apply
        self.v_th = thresh
        self.tau = tau
        self.gamma = gamma
        self.mem = None
        self.have_init = False

    def init_membrane_state(self, x):
        if not self.have_init:
            # shape[T, B, C, H, W]->init_shape[1, C, H, W]
            init_shape = (1, *x.shape[2:])
            self.mem = nn.Parameter(nn.init.uniform_(
                torch.empty(init_shape, device=x.device), a=-0.2, b=0.2))
            self.have_init = True
        return self.mem.to(x)

    def forward(self, x):
        mem = self.init_membrane_state(x)
        mem_v = []
        for t in range(x.shape[0]):
            mem = self.tau * mem + x[t, ...]
            spike = self.heaviside(mem - self.v_th, self.gamma)
            mem = mem * (1 - spike)
            mem_v.append(spike)

        return torch.stack(mem_v)


def speed_bench(shape: tuple):
    org_if = LIFLegacy().cuda()
    sta_if = LIFIMPLegacy().cuda()

    # ===================== IFNode =====================
    x = torch.randn(shape, requires_grad=True).cuda()
    y = torch.randn(shape, requires_grad=True).cuda()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    out = org_if(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    out = org_if(x)
    for _ in range(9):
        out = out + org_if(out)
    loss = torch.nn.functional.mse_loss(out, y)
    torch.cuda.synchronize()
    start_time.record()
    loss.backward()
    end_time.record()
    torch.cuda.synchronize()
    print(f'LIFNode\t{shape} \t: {start_time.elapsed_time(end_time) / 10} ms')
    # =================== LIFIMP ===================
    x = torch.randn(shape, requires_grad=True).cuda()
    y = torch.randn(shape, requires_grad=True).cuda()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    out = sta_if(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    out = sta_if(x)
    for _ in range(9):
        out = out + sta_if(out)
    loss = torch.nn.functional.mse_loss(out, y)
    torch.cuda.synchronize()
    start_time.record()
    loss.backward()
    end_time.record()
    torch.cuda.synchronize()
    print(f'LIFIMP\t{shape} \t: {start_time.elapsed_time(end_time) / 10} ms')


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
