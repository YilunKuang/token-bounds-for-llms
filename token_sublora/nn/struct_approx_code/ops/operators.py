from math import prod
from cola.ops.operator_base import LinearOperator
import torch
from token_sublora.nn.struct_approx_code.mm.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
from token_sublora.nn.struct_approx_code.mm.btt_mvm import btt_mvm
from token_sublora.nn.struct_approx_code.mm.hbtt_mvm import hbtt_mvm


class BlockDiagWithTranspose(LinearOperator):
    """
    Block-diagonal operator with Kron-like transpose
    Args:
        M (array_like): Block-diagonal matrix of shape (b, n, m)
        transpose (bool): Whether to transpose n and m after matmul
    """
    def __init__(self, M, transpose):
        dtype = M.dtype
        self.M = M
        self.b, self.n, self.m = M.shape
        shape = (self.b * self.n, self.b * self.m)
        self.transpose_out = transpose
        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        # v: (batch, d_in)
        v = v.view(-1, self.b, self.n)  # (i, b, n)
        v = torch.einsum('ibn,bnm->ibm', v, self.M)  # (i, b, m)
        if self.transpose_out:
            v = v.transpose(1, 2)
        v = v.reshape(-1, self.b * self.m)
        return v


class Butterfly(LinearOperator):
    def __init__(self, Ms):
        self.Ms = Ms
        dtype = self.Ms[0].dtype
        m = self.Ms[0].shape[0] * self.Ms[0].shape[-1]
        n = self.Ms[1].shape[0] * self.Ms[1].shape[1]
        shape = (m, n)
        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        out = blockdiag_butterfly_multiply(v, self.Ms[0], self.Ms[1])
        return out


# class Banded(LinearOperator):
#     def __init__(self, bands):
#         """ LinearOperator of shape (n,n) with (n,k) matrix of bands."""
#         n, self.k = bands.shape
#         self.bands = bands.reshape(-1)
#         super().__init__(bands.dtype, (n, n))

#     def _rmatmat(self, v):  # v of shape (B, n)
#         n = self.shape[-1]
#         indices = torch.arange(n).view((n, 1)).repeat((1, self.k))
#         shifted_indices = ((arange1 - torch.arange(k)) % n).view(-1)
#         # this line can be made faster by combining multiply and sum into one
#         return (v[:, shifted_indices] * bands).sum(-1)


class TeTrain(LinearOperator):
    """
    Tensor-Train operator

    Args:
        *Ms (array_like): Cores of the TT-decomposition
    """
    def __init__(self, Ms):
        self.Ms = Ms  # (ki, ni, mi, ki+1) ki = rank_in, ki+1 = rank_out
        dtype = self.Ms[0].dtype
        n = prod([M.shape[1] for M in Ms])
        m = prod([M.shape[2] for M in Ms])
        shape = (n, m)
        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        B = v.shape[0]
        v = v.view(B, *[Mi.shape[1] for Mi in self.Ms], -1)  # (B, n1, n2, ..., nd, k1)
        for M in self.Ms:
            v = torch.einsum('bn...k,knml->b...ml', v, M)  # flop count doesn't work for tensordot
        return v.view(B, -1)

    def to_dense(self):
        raise NotImplementedError

    def __str__(self):
        return "TT"


class BlockTeTrain(LinearOperator):
    """
    Tensor-Train operator with weights depending on the idle axes
    Implmented with einsum and einops only

    Args:
        *Ms (array_like): Cores of the TT-decomposition
    """
    def __init__(self, Ms):
        self.Ms = Ms  # (r_i+1, ri, m[:i], mi, ni, n[i+1:])
        dtype = self.Ms[0].dtype
        self.ms = [M.shape[2 + i] for i, M in enumerate(Ms)]
        self.ns = [M.shape[3 + i] for i, M in enumerate(Ms)]
        self.m = prod(self.ms)
        self.n = prod(self.ns)
        self.rank_out = [M.shape[0] for M in Ms]  # (r, r, ..., 1)
        self.rank_in = [M.shape[1] for M in Ms]  # (1, r, r, ..., r)
        assert self.rank_in[1:] == self.rank_out[:-1], "Rank must match"
        assert self.rank_in[0] == 1, "First rank must be 1"
        assert self.rank_out[-1] == 1, "Last rank must be 1"
        shape = (self.n, self.m)
        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        # v: (B, N)
        b = v.shape[0]
        for idx, M in enumerate(self.Ms):
            r = self.rank_out[idx]
            t = self.rank_in[idx]
            m = self.ms[idx]
            n = self.ns[idx]
            p = prod(self.ms[:idx])
            q = prod(self.ns[idx + 1:])
            v = v.reshape(b, t, p, n, q)
            M = M.view(r, t, p, m, n, q)
            v = torch.einsum('rtpmnq,btpnq->brpmq', M, v)
        # v: (b, 1, m1, m2, ..., m_d, 1)
        return v.view(b, -1)

    def to_dense(self):
        raise NotImplementedError

    def __str__(self):
        return "BlockTT"


class OptBlockTT(LinearOperator):
    """
   Args:
        *Ms (array_like): Cores of the TT-decomposition
    """
    def __init__(self, Ms, shapes):
        self.Ms = Ms
        dtype = self.Ms[0].dtype
        self.rs, self.ms, self.ns = shapes
        shape = (prod(self.ms), prod(self.ns))
        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        return btt_mvm(v, self.Ms[0], self.Ms[1], (self.rs, self.ms, self.ns))

    def __str__(self):
        return "BlockTT"


class HBTeTr(LinearOperator):
    """
   Args:
        *Ms (array_like): Cores of the TT-decomposition
    """
    def __init__(self, Ms, shapes):
        self.Ms = Ms
        dtype = self.Ms[0].dtype
        self.rs, self.ms, self.ns = shapes
        shape = (prod(self.ms), prod(self.ns))
        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        return hbtt_mvm(v, (self.rs, self.ms, self.ns), *self.Ms)

    def __str__(self):
        return "HBlockTT"


def init_random_cores(rs, ms, ns):
    if len(ms) == 2:
        core1 = torch.randn(rs[0], ms[0], ns[0], ms[1], rs[1])
        core2 = torch.randn(rs[1], ms[1], ns[1], ns[0], rs[2])
        cores = (core1, core2)
        return cores
    elif len(ms) == 3:
        core1 = torch.randn(rs[0], ms[0], ns[0], ms[1], ms[2], rs[1])
        core2 = torch.randn(rs[1], ms[1], ns[1], ms[2], ns[0], rs[2])
        core3 = torch.randn(rs[2], ms[2], ns[2], ns[0], ns[1], rs[3])
        cores = (core1, core2, core3)
        return cores
    else:
        raise NotImplementedError


class Permutation(LinearOperator):
    def __init__(self, indicies, dtype):
        self.indicies = indicies
        shape = (len(self.indicies), len(self.indicies))
        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        return v[:, self.indicies]
