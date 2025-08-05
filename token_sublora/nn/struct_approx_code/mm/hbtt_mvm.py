import torch


class HBTeTr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shapes, *cores):
        rs, ms, ns = shapes
        B = x.shape[0]
        ctx.shapes = shapes
        fws = []

        size = ms[1:] + (ms[0], B)
        y = x.transpose(0, 1).reshape(*size)

        for idx, core in enumerate(cores):
            size = ns[:idx] + ms[idx + 1:] + (B, rs[idx] * ms[idx])
            y = y.reshape(*size)
            fws.append(y.contiguous())
            y = y @ core
            size = ns[:idx] + ms[idx + 1:] + (B, rs[idx + 1], ns[idx])
            y = y.reshape(*size)
            y = y.transpose(idx, -1).contiguous()
        y = y.reshape(-1, B).T

        ctx.save_for_backward(x, *cores, *fws)
        return y

    @staticmethod
    def backward(ctx, grad):
        rs, ms, ns = ctx.shapes
        rs_rev = rs[::-1]
        D = len(ms)
        x, *all = ctx.saved_tensors
        cores, fws = all[:D], all[D:]
        B = x.shape[0]
        dWs = []

        prev = grad.T.reshape(*ns, B)
        jdx = [i for i in range(D)][::-1]
        for idx, core in enumerate(cores[::-1]):
            fw = fws.pop()
            neg_idx = -idx if idx > 0 else D
            prev = prev.transpose(jdx[idx], -1).contiguous()
            size = ns[:-(idx + 1)] + ms[neg_idx:] + (B, rs_rev[idx] * ns[-(idx + 1)])
            prev = prev.reshape(*size)
            dW = fw.transpose(-1, -2) @ prev
            dWs.append(dW)
            new = prev @ core.transpose(-1, -2)
            size = ns[:-(idx + 1)] + ms[neg_idx:] + (B, rs_rev[idx + 1], ms[-(idx + 1)])
            prev = new.reshape(*size)
        size = ns[:-(idx + 1)] + ms[neg_idx:] + (ms[-(idx + 1)], B)
        dx = new.reshape(*size)
        dx = dx.reshape(-1, B).transpose(0, 1)

        return dx, None, *dWs[::-1]


hbtt_mvm = HBTeTr.apply


def create_cores(rs, ms, ns, dtype, requires_grad):
    cores = []
    for idx in range(len(ms)):
        size = ns[:idx] + ms[idx + 1:] + (rs[idx] * ms[idx], rs[idx + 1] * ns[idx])
        core = torch.randn(*size, dtype=dtype, requires_grad=requires_grad)
        cores.append(core)
    return cores
