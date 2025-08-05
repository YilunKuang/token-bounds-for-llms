import itertools
from functools import reduce, partial
from math import prod
from math import ceil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchdistx.deferred_init import deferred_init, materialize_module
from fvcore.nn import FlopCountAnalysis
from torchinfo import summary
from sympy import factorint
from cola.fns import kron
from cola.ops import Dense
from cola.ops import Tridiagonal
from token_sublora.nn.struct_approx_code.ops.operators import TeTrain, OptBlockTT, Butterfly
from token_sublora.nn.struct_approx_code.ops.operators import BlockDiagWithTranspose, BlockTeTrain, Permutation
from token_sublora.nn.struct_approx_code.ops.operators import HBTeTr

from token_sublora.nn.monarch_mat.monarch_linear import MonarchLinear

def is_cola_param(x):
    return x.dtype == torch.float32 or x.dtype == torch.float64 or x.dtype == torch.float16


def dense_init(linear, zero_init=False):
    d_in, _ = linear.in_features, linear.out_features
    if zero_init:
        std = 0
        linear.weight.zero_init = True
    else:
        std = d_in**-0.5
    linear.weight.data.normal_(mean=0, std=std)
    if linear.bias is not None:
        linear.bias.data.zero_()


def cola_init(tensor, zero_init=False):
    assert hasattr(tensor, 'd_in'), 'Learnable CoLA parameter must have d_in attribute'
    if zero_init:
        print(f'Zero init cola param of shape {list(tensor.shape)}')
        with torch.no_grad():
            return tensor.zero_()
    else:
        std = 1 / np.sqrt(tensor.d_in)
        with torch.no_grad():
            return tensor.normal_(0, std)


def factorize(x, n):
    # Get prime factors and their counts
    prime_factors = factorint(x)

    # Initialize the n integers
    numbers = [1] * n

    # Distribute the prime factors
    for prime, count in prime_factors.items():
        for _ in range(count):
            # Find the number with the smallest product to assign the prime factor
            min_index = min(range(n), key=lambda i: numbers[i])
            numbers[min_index] *= prime

    # return in ascending order
    return sorted(numbers)


def get_builder_fn(struct, **kwargs):
    build_fn = partial(build_fns[struct], **kwargs)
    return build_fn


class CoLALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, structure=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.structure = structure
        self.in_features = in_features
        self.out_features = out_features
        if structure is None:
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()
        else:
            self.build_cola = lambda: build_fns[structure](in_features, out_features, bias=bias, **kwargs).to(
                device=device, dtype=dtype)
            self.cola_layer = self.build_cola()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.structure is None:
            nn.Linear.reset_parameters(self)
        else:
            self.cola_layer = self.build_cola()

    def forward(self, input):
        if self.structure is None:
            return F.linear(input, self.weight, self.bias)
        else:
            return self.cola_layer(input)


class CoLALayer(nn.Module):
    def __init__(self, A, bias=True):
        super().__init__()
        d_in_orig, d_out_orig = A.shape
        cola_tensors, self.unflatten = A.flatten()
        # learnable matrices parametrizing A
        self.matrix_params = nn.ParameterList()
        num_mats = sum(is_cola_param(t) for t in cola_tensors)
        # Iterate over cola_tensors and only turn those that meet the condition into Parameters
        # e.g. permutations are excluded
        self.cola_tensors = []
        for t in cola_tensors:
            if is_cola_param(t):
                assert hasattr(t, 'd_in'), f'Learnable CoLA parameter {t} must have d_in attribute'
                d_in = t.d_in
                t = nn.Parameter(t)
                # a heuristic to keep feature updates of similar size compared to a dense layer
                t.lr_mult = d_in_orig / d_in / num_mats
                self.cola_tensors.append(t)
                self.matrix_params.append(t)
            else:
                self.cola_tensors.append(t)

        self.A = self.unflatten(self.cola_tensors)  # (d_out, d_in)
        self.b = nn.Parameter(torch.zeros(d_out_orig)) if bias else None
        if bias:
            self.b.lr_mult = 1

    def _apply(self, fn):
        # _apply is called when doing model.to(device), model.cuda(), model.float(), etc.
        # apply fn to all parameters
        super()._apply(fn)  # TODO: check what happens to non-parameter tensors? (e.g. do they get moved to device?)
        # reconstruct A
        self.A = self.unflatten(self.cola_tensors)
        return self

    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        out = x @ self.A
        if self.b is not None:
            out = out + self.b
        return out.view(*batch_shape, -1)


class WeightedSumOfLayers(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.weights = nn.Parameter(torch.zeros(len(layers)))

    def forward(self, x):
        ys = torch.stack([layer(x) for layer in self.layers], dim=0)  # (n_layers, batch_size, d_out)
        ws = torch.softmax(self.weights, dim=0)[..., None, None]  # (n_layers, 1, 1)
        return torch.sum(ws * ys, dim=0)  # (batch_size, d_out)


def build_dense_test(d_in, d_out, bias=True, **_):
    U = torch.randn(d_out, d_in)
    nn.init.kaiming_normal_(U, mode='fan_in', nonlinearity='relu')
    return CoLALayer(Dense(U), bias=bias)


def build_dense(d_in, d_out, bias=True, zero_init=False, **_):
    U = torch.randn(d_in, d_out)
    U.d_in = d_in
    cola_init(U, zero_init)
    return CoLALayer(Dense(U), bias=bias)


def build_low_rank(d_in, d_out, rank_frac=0, bias=True, zero_init=False, **_):
    if rank_frac == 0:
        rank_frac = 1 / np.sqrt(min(d_in, d_out))
    rank = ceil(rank_frac * min(d_in, d_out))
    U = torch.randn(d_in, rank)
    U.d_in = d_in
    V = torch.randn(rank, d_out)
    V.d_in = rank
    cola_init(U)
    cola_init(V, zero_init)
    A = Dense(U) @ Dense(V)
    return CoLALayer(A, bias=bias)


def build_tridiag(d_in, d_out, bias=True, zero_init=False, **_):
    d = max(d_in, d_out)
    std = 0 if zero_init else np.sqrt(1 / 3)
    diag = torch.randn(d, 1) * std
    diag.d_in = 1
    low_diag = torch.randn(d - 1, 1) * std
    low_diag.d_in = 1
    up_diag = torch.randn(d - 1, 1) * std
    up_diag.d_in = 1
    A = Tridiagonal(low_diag, diag, up_diag)
    A = A[:d_in, :d_out]
    return CoLALayer(A, bias=bias)


def build_tt(d_in, d_out, tt_dim, tt_rank, permute=False, bias=True, zero_init=False, **_):
    ns, ms = factorize(d_in, tt_dim), factorize(d_out, tt_dim)
    print(f'TT shape: {ns} -> {ms}')
    cores = []
    for idx in range(tt_dim):
        rank_prev = 1 if idx == 0 else tt_rank
        rank_next = 1 if idx == tt_dim - 1 else tt_rank
        core = torch.randn(rank_prev, ns[idx], ms[idx], rank_next)
        core.d_in = ns[idx] * rank_prev
        cola_init(core, zero_init and idx == tt_dim - 1)
        cores.append(core)

    A = TeTrain(cores)
    if permute:
        P_in = Permutation(torch.randperm(d_in), dtype=A.dtype)
        P_out = Permutation(torch.randperm(d_out), dtype=A.dtype)
        A = P_in @ A @ P_out
    return CoLALayer(A, bias=bias)


def build_sbtt(d_in, d_out, tt_dim, tt_rank, bias=True, zero_init=False, **_):
    As = []
    for tt_idx in range(2, tt_dim + 1):
        ns, ms = tuple(factorize(d_out, tt_idx)), tuple(factorize(d_in, tt_idx))
        rs = [1] + [tt_rank] * (tt_idx - 1) + [1]
        shapes = (rs, ms, ns)
        cores = []
        for idx in range(tt_idx):
            size = ns[:idx] + ms[idx + 1:] + (rs[idx] * ms[idx], rs[idx + 1] * ns[idx])
            core = torch.randn(*size, dtype=torch.float32)
            core.d_in = rs[idx] * ms[idx]
            cola_init(core, zero_init and idx == tt_idx - 1)
            cores.append(core)
        A = HBTeTr(cores, shapes)
        As.append(A)
    A = sum(As)
    return CoLALayer(A, bias=bias)


def build_opt_hbtt(d_in, d_out, tt_dim, tt_rank, bias=True, zero_init=False, **_):
    ns, ms = tuple(factorize(d_out, tt_dim)), tuple(factorize(d_in, tt_dim))
    rs = (1, tt_rank, 1)
    shapes = (rs, ms, ns)
    cores = []
    for idx in range(tt_dim):
        size = ns[:idx] + ms[idx + 1:] + (rs[idx] * ms[idx], rs[idx + 1] * ns[idx])
        core = torch.randn(*size, dtype=torch.float32)
        core.d_in = rs[idx] * ms[idx]
        cola_init(core, zero_init and idx == tt_dim - 1)
        cores.append(core)
    A = HBTeTr(cores, shapes)
    return CoLALayer(A, bias=bias)


def build_opt_btt(d_in, d_out, tt_dim, tt_rank, bias=True, zero_init=False, **_):
    ns, ms = tuple(factorize(d_out, tt_dim)), tuple(factorize(d_in, tt_dim))
    rs = (1, tt_rank, 1)
    shapes = (rs, ms, ns)
    cores = []
    for idx in range(tt_dim):
        size = ns[:idx] + ms[idx + 1:] + (rs[idx] * ms[idx], rs[idx + 1] * ns[idx])
        core = torch.randn(*size, dtype=torch.float32)
        core.d_in = rs[idx] * ms[idx]
        cola_init(core, zero_init and idx == tt_dim - 1)
        cores.append(core)
    A = OptBlockTT(cores, shapes)
    return CoLALayer(A, bias=bias)


def build_bfly(d_in, d_out, tt_dim, bias=True, **_):
    ns, ms = tuple(factorize(d_out, tt_dim)), tuple(factorize(d_in, tt_dim))
    rs = (1, 1, 1)
    cores = []
    for idx in range(tt_dim):
        size = ns[:idx] + ms[idx + 1:] + (rs[idx] * ms[idx], rs[idx + 1] * ns[idx])
        core = torch.randn(*size, dtype=torch.float32)
        core.d_in = rs[idx] * ms[idx]
        cores.append(core)
    A = Butterfly((cores[0], cores[1]))
    return CoLALayer(A, bias=bias)


def build_block_tt(d_in, d_out, tt_dim, tt_rank, bias=True, zero_init=False, **_):
    # tt_rank^2 should be much smaller than d_in and d_out
    ns = factorize(d_in, tt_dim)
    ms = factorize(d_out, tt_dim)
    print(f'TT shape: {ns} -> {ms}')
    cores = []
    for idx in range(tt_dim):
        n, m = ns[idx], ms[idx]
        rank_prev = 1 if idx == 0 else tt_rank
        rank_next = 1 if idx == tt_dim - 1 else tt_rank
        core = torch.rand(rank_next, rank_prev, *(ms[:idx] + [m, n] + ns[idx + 1:]))
        core.d_in = n * rank_prev
        cola_init(core, zero_init and idx == tt_dim - 1)
        cores.append(core)

    A = BlockTeTrain(cores)
    return CoLALayer(A, bias=bias)


def build_perm_block_tt(d_in, d_out, tt_dim, tt_rank, bias=True, zero_init=False, **_):
    # tt_rank^2 should be much smaller than d_in and d_out
    ns = factorize(d_in, tt_dim)
    ms = factorize(d_out, tt_dim)
    print(f'TT shape: {ns} -> {ms}')
    cores = []
    for idx in range(tt_dim):
        n, m = ns[idx], ms[idx]
        rank_prev = 1 if idx == 0 else tt_rank
        rank_next = 1 if idx == tt_dim - 1 else tt_rank
        core = torch.rand(rank_next, rank_prev, *(ms[:idx] + [m, n] + ns[idx + 1:]))
        core.d_in = n * rank_prev
        cola_init(core, zero_init and idx == tt_dim - 1)
        cores.append(core)

    A = BlockTeTrain(cores)
    P_in = Permutation(torch.randperm(d_in), dtype=A.dtype)
    P_out = Permutation(torch.randperm(d_out), dtype=A.dtype)
    A = P_in @ A @ P_out
    return CoLALayer(A, bias=bias)


def count_btt_flops(cores):
    return sum(prod(core.shape) for core in cores)


def build_monarch(d_in, d_out, num_blocks=None, bias=True, zero_init=False, **kwargs):
    monarch_nblocks = kwargs['monarch_nblocks']

    tt_dim = 2
    if num_blocks is None:  # ~sqrt(d_in) blocks
        ns = factorize(d_in, tt_dim)
        ms = factorize(d_out, tt_dim)
    else:
        ns = [num_blocks, ceil(d_in // num_blocks)]
        ms = [num_blocks, ceil(d_out // num_blocks)]
    print(f'Monarch shape: {ns} -> {ms} ({num_blocks} blocks)')
    cores = []
    for idx in range(tt_dim):
        n, m = ns[idx], ms[idx]
        rank_prev = 1
        rank_next = 1
        core = torch.rand(rank_next, rank_prev, *(ms[:idx] + [m, n] + ns[idx + 1:]))
        core.d_in = n
        cola_init(core, zero_init and idx == tt_dim - 1)
        cores.append(core)

    return MonarchLinear(in_features=d_in, out_features=d_out, nblocks=monarch_nblocks, bias=bias)
    
    # A = BlockTeTrain(cores)
    # A = A[:d_in, :d_out]
    # return CoLALayer(A, bias=bias)

class KroneckerLinear(nn.Module):
    def __init__(self, U, V, bias=False):
        super(KroneckerLinear, self).__init__()
        self.matrix_params_U = nn.Parameter(U)
        self.matrix_params_V = nn.Parameter(V)
        self.bias = nn.Parameter(torch.randn(self.matrix_params_U.shape[1] * self.matrix_params_V.shape[1]), requires_grad=bias) if bias else None

    def forward(self, x):
        W = torch.kron(self.matrix_params_U, self.matrix_params_V)
        out = x @ W
        if self.bias is not None:
            out = out + self.bias
        return out

def integer_decompositions(n):
    decompositions = []
    for a in range(1, n + 1):
        if n % a == 0:
            b = n // a
            if a <= b:
                decomposition = (a, b)
                decompositions.append(decomposition)
    return decompositions

def kron_prime_factorization(d_in, d_out):
    d_in_kron_pairs = integer_decompositions(d_in)
    d_out_kron_pairs = integer_decompositions(d_out)
    return d_in_kron_pairs, d_out_kron_pairs

def build_kron(d_in, d_out, permute=False, bias=True, zero_init=False, **kwargs):
    kron_factorized_mode = kwargs['kron_factorized_mode']
    # n1, n2 = factorize(d_in, 2) # 768 -> 16 , 48 -> 32 , 24
    # m1, m2 = factorize(d_out, 2) # 768 -> 16, 48

    d_in_kron_pairs, d_out_kron_pairs = kron_prime_factorization(d_in, d_out)
    n1, n2 = d_in_kron_pairs[kron_factorized_mode]
    m1, m2 = d_out_kron_pairs[kron_factorized_mode]
    
    # kron
    # (d_in / n_blocks)**2 + n_blocks**2

    # monarch
    # (d_in / n_blocks)**2 * n_blocks + n_blocks**2 * d_in

    U = torch.randn(n1, m1)
    V = torch.randn(n2, m2)
    U.d_in = n1
    V.d_in = n2
    cola_init(U)
    cola_init(V, zero_init)
    A = kron(Dense(U), Dense(V))
    print(f"building KroneckerLinear with U.shape={U.shape} and V.shape={V.shape}")
    return KroneckerLinear(U,V,bias=bias)

    # if permute:
    #     P_in = Permutation(torch.randperm(d_in), dtype=A.dtype)
    #     P_out = Permutation(torch.randperm(d_out), dtype=A.dtype)
    #     A = P_in @ A @ P_out
    # return CoLALayer(A, bias=bias)


def build_blockdiag(d_in, d_out, bias=True, transpose=True, zero_init=False, **_):
    num_blocks = ceil(np.sqrt(d_out))
    block_in = ceil(d_in / num_blocks)
    block_out = ceil(d_out / num_blocks)
    M = torch.randn(num_blocks, block_in, block_out)
    M.d_in = block_in
    cola_init(M, zero_init)
    A = BlockDiagWithTranspose(M, transpose)
    A = A[:d_in, :d_out]
    return CoLALayer(A, bias=bias)


def colafy(model, struct, layers, cola_lr_mult=1., device='cuda', **kwargs):
    build_cola = build_fns[struct]
    layer_select_fn = layer_select_fns[layers]
    # Create a list of all linear layers and their names
    linear_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    num_layers = len(linear_layers)

    for layer_idx, (name, module) in enumerate(linear_layers):
        if build_cola and layer_select_fn(name, layer_idx, num_layers):
            # replace linear layer with CoLA layer
            d_in, d_out = module.in_features, module.out_features
            bias = module.bias is not None
            zero_init = hasattr(module.weight, 'zero_init') and module.weight.zero_init
            if zero_init:
                print(f'Zero init: {name}')
            
            # assert hasattr(module.weight, 'lr_mult'), 'Weights in linear layer must have lr_mult attribute' # TODO: yilun commented this out 
            # lr_mult = module.weight.lr_mult # TODO: yilun commented this out 
            
            cola_layer = build_cola(d_in, d_out, bias=bias, zero_init=zero_init, layer_idx=layer_idx, **kwargs)
            
            # breakpoint()

            # for p in cola_layer.matrix_params: # TODO: yilun commented this out 
            #     # rescale lr # TODO: yilun commented this out 
            #     p.lr_mult *= lr_mult # TODO: yilun commented this out 
            #     # apply global cola lr multiplier # TODO: yilun commented this out 
            #     p.lr_mult *= cola_lr_mult # TODO: yilun commented this out 
            
            # Split the name to get parent module and attribute name
            name_split = name.rsplit('.', 1)

            # breakpoint()

            # If it's a top-level module
            if len(name_split) == 1:
                setattr(model, name_split[0], cola_layer)
            else:
                parent_name, attr_name = name_split
                parent_module = reduce(getattr, parent_name.split('.'), model)
                setattr(parent_module, attr_name, cola_layer)
    
    model.to(device)
    return model


build_fns = {
    'low_rank': build_low_rank,
    'kron': build_kron,
    'tt': build_tt,
    'block_tt': build_block_tt,
    'perm_block_tt': build_perm_block_tt,
    'btt': build_opt_btt,
    'hbtt': build_opt_hbtt,
    'sbtt': build_sbtt,
    'bfly': build_bfly,
    'block_tt': build_block_tt,
    'monarch': build_monarch,
    'tridiag': build_tridiag,
    'blockdiag': lambda *args, **kwargs: build_blockdiag(*args, transpose=False, **kwargs),
    'blockdiagT': lambda *args, **kwargs: build_blockdiag(*args, transpose=True, **kwargs),
    'dense': build_dense,
    'dense_test': build_dense_test,
    'none': None,
}


def select_ffn_layers(name, layer_idx, num_layers):
    return 'fn.net' in name

def select_attn_layers(name, layer_idx, num_layers):
    return 'to_qkv' in name or 'to_out' in name

def select_attn_and_lm_head_layers(name, layer_idx, num_layers):
    return 'attn.c_attn' in name or 'attn.q_proj' in name or 'attn.k_proj' in name or 'attn.v_proj' in name or 'attn.c_proj' in name or 'lm_head' in name


layer_select_fns = {
    'all': lambda *_: True,
    'none': lambda *_: False,
    'all_but_last': lambda name, i, n: i < n - 1,
    'intermediate': lambda name, i, n: i > 0 and i < n - 1,
    'ffn': select_ffn_layers,
    'attn': select_attn_layers,
    'attn_and_lm_head': select_attn_and_lm_head_layers,
}


def cola_parameterize(model_builder, base_config, lr, input_shape, target_config=None, init_mult=1, struct='none',
                      layer_select_fn='all_but_last', zero_init_fn=lambda w, name: False, extra_lr_mult_fn=lambda p_name: 1,
                      device='cuda', cola_kwargs={}, optim_kwargs={}):
    """
    Create a model and its optimizer in an appropriate parameterization.
    Takes care of lr adjustment both for scaling up model size and for switching to CoLA layers.

    Usage:
        1. Regular μP: struct == 'none'
        2. Regular μP + dense layers initialized by CoLA: struct == 'dense' (e.g. specify custom zero inits)
        3. Regular μP + switching to arbitrary CoLA layers: struct == 'btt', 'tt', etc.

    Assumes:
    1. Classification head has been zero initialized.
    2. Every parameter tensor with more > 2 axes has been annotated with an attribute "fan_in_dims",
        a tuple of dimensions that are considered to be fan-in dimensions.
    3. If struct == 'none', init scale for the dense model is automatically in μP (often true with standard inits).
    4. layer_select_fn does not select the last layer.
    5. If struct != 'none', zero_init_fn selects the last linear layer in every residual block.

    Args:
        model_builder: function that builds the model
        base_config: kwargs for base model
        lr: learning rate for base model
        input_shape: shape of input model
        target_config: kwargs for scaled up model, same as base_config if not specified.
        init_mult: rescale initialization of all matrix-like parameters
        struct: structure of CoLA layers
        layer_select_fn: function that selects which layers to replace with CoLA layers
        zero_init_fn: function maps linear.weight and module name to whether to zero initialize
        extra_lr_mult_fn: function that maps parameter names to extra lr multipliers
        device: device to put model on
        cola_kwargs: kwargs for building CoLA layers
        optim_kwargs: kwargs for optimizer (AdamW)
    """
    base_model = deferred_init(model_builder, **base_config)
    base_param_shapes = [p.shape for p in base_model.parameters()]
    del base_model
    if target_config is None:
        target_config = base_config
    model = deferred_init(model_builder, **target_config)
    params, param_names, param_shapes = [], [], []
    for name, p in model.named_parameters():
        params.append(p)
        param_names.append(name)
        param_shapes.append(p.shape)
    # compute μP lr multipliers
    lr_mults = {}
    for base_shape, name, shape, p in zip(base_param_shapes, param_names, param_shapes, params):
        if base_shape == shape:
            mult = 1
        else:
            # special cases, e.g. word embeddings, positional embeddings
            if hasattr(p, 'fan_in_dims'):
                fan_in_dims = p.fan_in_dims
                d_in_base = base_shape[fan_in_dims]
                d_in = shape[fan_in_dims]
                if isinstance(d_in_base, tuple):
                    d_in_base = prod(d_in_base)
                    d_in = prod(d_in)
                mult = d_in_base / d_in
            # matrix like (d_out, d_in)
            elif len(base_shape) == 2:
                d_in_base = base_shape[1]
                d_in = shape[1]
                mult = d_in_base / d_in
            # vector like (d_out,), e.g. bias, layernorm gamma
            elif len(base_shape) == 1:
                mult = 1
            else:
                raise ValueError(
                    f'Non matrix or vector parameter {name} has shape {shape}, but does not have fan_in_dims attribute.')
        p.lr_mult = mult
        lr_mults[name] = mult
    # replace some linear layers with cola layers,
    # with custom initialization and updated lr multipliers
    if struct != 'none':
        build_cola = build_fns[struct]
        if isinstance(layer_select_fn, str):
            layer_select_fn = layer_select_fns[layer_select_fn]
        assert callable(layer_select_fn), f'layer_select_fn must be callable, got {layer_select_fn}'
        # Create a list of all linear layers and their names
        linear_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
        num_layers = len(linear_layers)
        for layer_idx, (name, module) in enumerate(linear_layers):
            if layer_select_fn(name, layer_idx, num_layers):
                d_in, d_out = module.in_features, module.out_features
                bias = module.bias is not None
                zero_init = zero_init_fn(module.weight, name)
                if zero_init:
                    print(f'Zero init: {name}')
                assert hasattr(module.weight, 'lr_mult'), 'Weights in linear layer must have lr_mult attribute'
                dense_lr_mult = module.weight.lr_mult
                cola_layer = build_cola(d_in, d_out, bias=bias, zero_init=zero_init,
                                        **cola_kwargs)  # cola specific lr mult are attached
                for p in cola_layer.matrix_params:
                    p.lr_mult = p.lr_mult * dense_lr_mult  # final cola mult = cola mult * dense mult
                # Split the name to get parent module and attribute name
                name_split = name.rsplit('.', 1)
                # If it's a top-level module
                if len(name_split) == 1:
                    setattr(model, name_split[0], cola_layer)
                else:
                    parent_name, attr_name = name_split
                    parent_module = reduce(getattr, parent_name.split('.'), model)
                    setattr(parent_module, attr_name, cola_layer)
    # materialize all other tensors untouched by cola
    materialize_module(model)
    model.to(device)
    # materialization gets rid of lr_mults of those tensors, so we need to add them back
    for name, p in model.named_parameters():
        if not hasattr(p, 'lr_mult'):
            p.lr_mult = lr_mults[name]
    # adjust lr and create optimizer
    param_groups = []
    lrs = []
    for name, param in model.named_parameters():
        assert hasattr(param, 'lr_mult'), f'lr_mult not found for {name}'
        mult = param.lr_mult
        extra_mult = extra_lr_mult_fn(name)
        if extra_mult != 1:
            print(f'{name}: {extra_mult} * {mult}')
            mult *= extra_mult
        else:
            print(f'{name}: {mult}')
        adjusted_lr = lr * mult
        lrs.append(adjusted_lr)
        param_groups.append({'params': param, 'lr': adjusted_lr})
    optimizer = torch.optim.AdamW(param_groups, **optim_kwargs)

    # globally rescale initialization for all params with >= 2 dimensions (skip bias, LN)
    for p in model.parameters():
        if p.dim() >= 2:
            p.data *= init_mult

    print('Model:')
    stats = summary(model, input_shape)
    cola_params = stats.trainable_params
    fake_input = torch.randn(*input_shape).to(device)
    cola_flops = FlopCountAnalysis(model, fake_input).set_op_handle(**custom_ops).total()
    print(f'Params: {cola_params / 1e6:.2f}M')
    print(f'FLOPs: {cola_flops / 1e6:.2f}M')
    print('=' * 90)

    info = {'cola_params': cola_params, 'cola_flops': cola_flops}
    return model, optimizer, info


def btt_flop_count(inputs, outputs):
    x, W1, W2 = inputs
    batch_size = get_shape(x)[0]
    params = get_numel(W1) + get_numel(W2)
    return batch_size * params


def get_shape(x):
    return x.type().sizes()


def get_numel(x):
    return prod(get_shape(x))


custom_ops = {'prim::PythonOp.BlockTeTr': btt_flop_count}