import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import diffeq_layers
from .squeeze import squeeze, unsqueeze
from ..matrix_utils import vjp, jvp

__all__ = ["ODEnet", "AutoencoderDiffEqNet", "ODEfunc", "AutoencoderODEfunc"]


def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


# def divergence_bf(f, y, **unused_kwargs):
#     jac = _get_minibatch_jacobian(f, y)
#     diagonal = jac.view(jac.shape[0], -1)[:, ::jac.shape[1]]
#     return torch.sum(diagonal, 1)


def _get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.

    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(y[:, j], x, torch.ones_like(y[:, j]), retain_graph=True,
                                      create_graph=True)[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx




def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)
    
def divergence_hutchplusplus(f, y, S=None, G=None, AS_implement='jvp_'):

    '''
        Implementation of Hutch++ with vector-queries.
        S.shape  = [bs, d]
        G.shape  = [bs, d]
    '''
    if G is None:
        G = S.detach().clone()  # [bs, d], sample_gaussian_like(e)

    if AS_implement == 'bf_':
        # brute_force
        # step 1: conmpute complete jacobian matrix 
        # step 2: Jacobian-Vector Product by torch.bmm()
        dzdx = _get_minibatch_jacobian(f, y)  
        dzdx_S = torch.bmm(dzdx, S.unsqueeze(-1))  # [bs, d, m/3=1]
    elif AS_implement == 'db_':
        # double backward
        v = torch.ones_like(f, requires_grad=True)
        vjp = torch.autograd.grad(f, y, grad_outputs=v, create_graph=True)[0]
        dzdx_S = torch.autograd.grad(vjp, v, grad_outputs=S, create_graph=True)[0]
        dzdx_S = dzdx_S.unsqueeze(-1)
    elif AS_implement == 'jvp_':
        dzdx_S = jvp(f, y, S, create_graph=True)
        dzdx_S = dzdx_S.unsqueeze(-1)
    else:
        raise NameError("The implement method must be in ['bf_', 'db_', 'jvp_']")

    [Q, _] = torch.linalg.qr(dzdx_S, mode='reduced')  # [bs, d, m/3=1]
    
    ## detach() ?
    # Q = Q.detach()
    
    # update G = G - Q @ (Q^T @ G)
    G = G.unsqueeze(-1) - torch.bmm(Q, torch.bmm(Q.permute(0, 2, 1), G.unsqueeze(-1)))  # [bs, d, m/3=1]
    G = G.squeeze(-1)  # [bs, d]
    Q = Q.squeeze(-1)  # [bs, d]

    # compute tr(Q^T @ A @ Q)
    Q_dzdx = torch.autograd.grad(f, y, Q, create_graph=True)[0]
    Q_dzdx_Q = Q_dzdx * Q
    trace_1 = Q_dzdx_Q.view(y.shape[0], -1).sum(dim=1)

    # compute tr(G^T @ A @ G)       
    # hutchinson's estimator == divergence_approx(f, y, e=G)
    G_dzdx = torch.autograd.grad(f, y, G, create_graph=True)[0]
    G_dzdx_G = G_dzdx * G
    trace_2 = G_dzdx_G.view(y.shape[0], -1).sum(dim=1)

    approx_tr_dzdx = trace_1 + trace_2


    return approx_tr_dzdx

def divergence_hutchplusplus_m(f, y, S=None, G=None, AS_implement='jvp_'):

    '''
        Implementation of Hutch++ with m queries
        S.shape  = [bs, d, m]
        G.shape  = [bs, d, m]
    '''
    bs = f.size(0)
    m = S.size(-1)
    if G is None:
        G = S.detach().clone()  # [bs, d], sample_gaussian_like(e)
    if AS_implement == 'bf_':
        dzdx = _get_minibatch_jacobian(f, y)  # batch jacobian matrix [bs, d, d]
        dzdx_S = torch.bmm(dzdx, S)  # [bs , d, m]
        [Q, _] = torch.linalg.qr(dzdx_S, mode='reduced')  # [bs , d, mim(d, m)]
        # Q = Q.detach()

        # update G = G - Q @ (Q^T @ G)
        G = G - torch.bmm(Q, torch.bmm(Q.permute(0, 2, 1), G))  # [bs, d, m]

        Q_dzdx_Q = torch.bmm(torch.bmm(Q.permute(0, 2, 1), dzdx), Q)
        index_Q = torch.arange(Q_dzdx_Q.size(-1))
        diagonal_1 = Q_dzdx_Q[:, index_Q, index_Q]
        # diagonal_1 = Q_dzdx_Q.view(Q_dzdx_Q.shape[0], -1)[:, ::Q_dzdx_Q.shape[1] + 1]
        trace_1 = torch.sum(diagonal_1, 1)

        G_dzdx_G = torch.bmm(torch.bmm(G.permute(0, 2, 1), dzdx), G)
        index_G = torch.arange(G_dzdx_G.size(-1))
        diagonal_2 = G_dzdx_G[:, index_G, index_G]
        # diagonal_2 = G_dzdx_G.view(G_dzdx_G.shape[0], -1)[:, ::G_dzdx_G.shape[1] + 1]
        trace_2 = torch.sum(diagonal_2, 1)

    elif AS_implement == 'jvp_':
        m = S.shape[-1]
        dzdx_S_list = []
        for i in range(m):
            dzdx_S = jvp(f, y, S[..., i], create_graph=True)
            dzdx_S_list.append(dzdx_S)
        dzdx_S_full = torch.stack(dzdx_S_list, dim=-1)
        [Q, _] = torch.linalg.qr(dzdx_S_full, mode='reduced')  # [bs, d, m/3=1]

        # update G = G - Q @ (Q^T @ G)
        G = G - torch.bmm(Q, torch.bmm(Q.permute(0, 2, 1), G))  # [bs, d, m/3=1]

        dzdx_Q_list, dzdx_G_list = [], []
        for i in range(Q.shape[-1]):
            dzdx_Q = jvp(f, y, Q[..., i], create_graph=True)
            dzdx_Q_list.append(dzdx_Q)
        for i in range(G.shape[-1]):
            dzdx_G = jvp(f, y, G[..., i], create_graph=True)
            dzdx_G_list.append(dzdx_G)
        dzdx_Q_full = torch.stack(dzdx_Q_list, dim=-1)
        dzdx_G_full = torch.stack(dzdx_G_list, dim=-1)  

        Q_dzdx_Q = torch.bmm(Q.permute(0, 2, 1), dzdx_Q_full)
        index_Q = torch.arange(Q_dzdx_Q.size(-1))
        diagonal_1 = Q_dzdx_Q[:, index_Q, index_Q]
        # diagonal_1 = Q_dzdx_Q.view(Q_dzdx_Q.shape[0], -1)[:, ::Q_dzdx_Q.shape[1] + 1]
        trace_1 = torch.sum(diagonal_1, 1)

        G_dzdx_G = torch.bmm(G.permute(0, 2, 1), dzdx_G_full)
        index_G = torch.arange(G_dzdx_G.size(-1))
        diagonal_2 = G_dzdx_G[:, index_G, index_G]
        # diagonal_2 = G_dzdx_G.view(G_dzdx_G.shape[0], -1)[:, ::G_dzdx_G.shape[1] + 1]
        trace_2 = torch.sum(diagonal_2, 1)
        

    approx_tr_dzdx = trace_1 + 1 / m * trace_2


    return approx_tr_dzdx

def divergence_approx_m(f, y, e=None, mode='loop'):
    bs = f.size(0)
    if mode == 'loop':
        approx_tr_dzdx = []
        for i in range(e.size(-1)):
            e_dzdx = torch.autograd.grad(f, y, e[..., i], create_graph=True)[0]
            e_dzdx_e = e_dzdx * e[..., i]
            approx_tr_dzdx.append(e_dzdx_e.view(y.shape[0], -1).sum(dim=1))
        tr = torch.stack(approx_tr_dzdx, dim=-1).mean(-1)
    elif mode == 'matrix':
        dzdx = _get_minibatch_jacobian(f, y)  # batch jacobian matrix [bs, d, d]
        m = e.size(-1)
        e_dzdx_e = torch.bmm(torch.bmm(e.permute(0, 2, 1), dzdx), e)  # [bs, m, m]
        index = torch.arange(m)
        # diagonal = e_dzdx_e.view(e_dzdx_e.shape[0], -1)[:, ::e_dzdx_e.shape[1]+1]
        diagonal = e_dzdx_e[:, index, index]
        tr = diagonal.mean(dim=-1)
    else:
        tr = None

    return tr


def divergence_na_hutchplusplus(f, y, S=None, R=None, G=None):

    '''
        Implementation of NA-Hutch++
        S.shape  = [bs, d]
        R.shape  = [bs, d]
        G.shape  = [bs, d]
    '''

    Z= jvp(f, y, R, create_graph=True)
    Z = Z.unsqueeze(-1)
    W = jvp(f, y, S, create_graph=True)
    W = W.unsqueeze(-1)

    pinv_SZ = torch.linalg.pinv(torch.bmm(S.unsqueeze(-1).permute(0, 2, 1), Z))
    pinv_SZ = pinv_SZ.detach()  # It will cost infinite time to backward the gradient without tensor.detach()
    
    WZ = torch.bmm(W.permute(0, 2, 1), Z)
    # WG = torch.bmm(W.permute(0, 2, 1), G.unsqueeze(-1))
    GZ = torch.bmm(G.unsqueeze(-1).permute(0, 2, 1), Z)
    trace_1 = (pinv_SZ * WZ).view(y.shape[0], -1).sum(dim=1)
    
    G_dzdx = torch.autograd.grad(f, y, G, create_graph=True)[0]
    G_dzdx_G = G_dzdx * G
    trace_2 = G_dzdx_G.view(y.shape[0], -1).sum(dim=1)
    
    trace_3 = (torch.bmm(torch.bmm(GZ * pinv_SZ, W.permute(0, 2, 1)), G.unsqueeze(-1))).view(y.shape[0], -1).sum(dim=1)

    approx_tr_dzdx = trace_1 + (trace_2 - trace_3) / 3.

    return approx_tr_dzdx

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}


class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(
        self, hidden_dims, input_shape, strides, conv, layer_type="concat", nonlinearity="softplus", num_squeeze=0
    ):
        super(ODEnet, self).__init__()
        self.num_squeeze = num_squeeze
        if conv:
            assert len(strides) == len(hidden_dims) + 1
            base_layer = {
                "ignore": diffeq_layers.IgnoreConv2d,
                "hyper": diffeq_layers.HyperConv2d,
                "squash": diffeq_layers.SquashConv2d,
                "concat": diffeq_layers.ConcatConv2d,
                "concat_v2": diffeq_layers.ConcatConv2d_v2,
                "concatsquash": diffeq_layers.ConcatSquashConv2d,
                "blend": diffeq_layers.BlendConv2d,
                "concatcoord": diffeq_layers.ConcatCoordConv2d,
            }[layer_type]
        else:
            strides = [None] * (len(hidden_dims) + 1)
            base_layer = {
                "ignore": diffeq_layers.IgnoreLinear,
                "hyper": diffeq_layers.HyperLinear,
                "squash": diffeq_layers.SquashLinear,
                "concat": diffeq_layers.ConcatLinear,
                "concat_v2": diffeq_layers.ConcatLinear_v2,
                "concatsquash": diffeq_layers.ConcatSquashLinear,
                "blend": diffeq_layers.BlendLinear,
                "concatcoord": diffeq_layers.ConcatLinear,
            }[layer_type]

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape

        for dim_out, stride in zip(hidden_dims + (input_shape[0],), strides):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {"ksize": 3, "stride": 1, "padding": 1, "transpose": False}
            elif stride == 2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": False}
            elif stride == -2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": True}
            else:
                raise ValueError('Unsupported stride: {}'.format(stride))

            layer = base_layer(hidden_shape[0], dim_out, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] // 2, hidden_shape[2] // 2
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] * 2, hidden_shape[2] * 2

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t, y):
        dx = y
        # squeeze
        for _ in range(self.num_squeeze):
            dx = squeeze(dx, 2)
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        # unsqueeze
        for _ in range(self.num_squeeze):
            dx = unsqueeze(dx, 2)
        return dx


class AutoencoderDiffEqNet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(self, hidden_dims, input_shape, strides, conv, layer_type="concat", nonlinearity="softplus"):
        super(AutoencoderDiffEqNet, self).__init__()
        assert layer_type in ("ignore", "hyper", "concat", "concatcoord", "blend")
        assert nonlinearity in ("tanh", "relu", "softplus", "elu")

        self.nonlinearity = {"tanh": F.tanh, "relu": F.relu, "softplus": F.softplus, "elu": F.elu}[nonlinearity]
        if conv:
            assert len(strides) == len(hidden_dims) + 1
            base_layer = {
                "ignore": diffeq_layers.IgnoreConv2d,
                "hyper": diffeq_layers.HyperConv2d,
                "squash": diffeq_layers.SquashConv2d,
                "concat": diffeq_layers.ConcatConv2d,
                "blend": diffeq_layers.BlendConv2d,
                "concatcoord": diffeq_layers.ConcatCoordConv2d,
            }[layer_type]
        else:
            strides = [None] * (len(hidden_dims) + 1)
            base_layer = {
                "ignore": diffeq_layers.IgnoreLinear,
                "hyper": diffeq_layers.HyperLinear,
                "squash": diffeq_layers.SquashLinear,
                "concat": diffeq_layers.ConcatLinear,
                "blend": diffeq_layers.BlendLinear,
                "concatcoord": diffeq_layers.ConcatLinear,
            }[layer_type]

        # build layers and add them
        encoder_layers = []
        decoder_layers = []
        hidden_shape = input_shape
        for i, (dim_out, stride) in enumerate(zip(hidden_dims + (input_shape[0],), strides)):
            if i <= len(hidden_dims) // 2:
                layers = encoder_layers
            else:
                layers = decoder_layers

            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {"ksize": 3, "stride": 1, "padding": 1, "transpose": False}
            elif stride == 2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": False}
            elif stride == -2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": True}
            else:
                raise ValueError('Unsupported stride: {}'.format(stride))

            layers.append(base_layer(hidden_shape[0], dim_out, **layer_kwargs))

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] // 2, hidden_shape[2] // 2
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] * 2, hidden_shape[2] * 2

        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.decoder_layers = nn.ModuleList(decoder_layers)

    def forward(self, t, y):
        h = y
        for layer in self.encoder_layers:
            h = self.nonlinearity(layer(t, h))

        dx = h
        for i, layer in enumerate(self.decoder_layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if i < len(self.decoder_layers) - 1:
                dx = self.nonlinearity(dx)
        return h, dx


class ODEfunc(nn.Module):

    def __init__(self, diffeq, divergence_fn="approximate", residual=False, rademacher=False):
        super(ODEfunc, self).__init__()
        assert divergence_fn in ("brute_force", "approximate", "approximate_m", "hutchplusplus", "hutchplusplus_m", "na_hutchplusplus")

        # self.diffeq = diffeq_layers.wrappers.diffeq_wrapper(diffeq)
        self.diffeq = diffeq
        self.residual = residual
        self.rademacher = rademacher
        self.div_fn_name = divergence_fn
        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx
        elif divergence_fn == "approximate_m":
            self.divergence_fn = divergence_approx_m
        elif divergence_fn == "hutchplusplus":
            self.divergence_fn = divergence_hutchplusplus
        elif divergence_fn == "hutchplusplus_m":
            self.divergence_fn = divergence_hutchplusplus_m
        elif divergence_fn == "na_hutchplusplus":
            self.divergence_fn = divergence_na_hutchplusplus

        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]

        # increment num evals
        self._num_evals += 1

        # if self._num_evals
        # convert to tensor
        t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = sample_rademacher_like(y)
                self._e_1 = sample_rademacher_like(y)
            else:
                if self.div_fn_name in ['hutchplusplus_m', "approximate_m"]:
                    m = 2
                    self._e = sample_gaussian_like(y.unsqueeze(-1).expand(-1, -1, m))
                    self._e_1 = sample_gaussian_like(y.unsqueeze(-1).expand(-1, -1, m))
                else:
                    self._e = sample_gaussian_like(y)
                    self._e_1 = sample_gaussian_like(y)
                    self._e_2 = sample_gaussian_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            # Hack for 2D data to use brute force divergence computation.
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf(dy, y, e=self._e).view(batchsize, 1)
            else:
                if self.div_fn_name in ['hutchplusplus', 'hutchplusplus_m']:

                    divergence = self.divergence_fn(dy, y, S=self._e, G = self._e_1).view(batchsize, 1)

                elif self.div_fn_name == 'na_hutchplusplus':
                    divergence = self.divergence_fn(dy, y, S=self._e, R=self._e_1, G = self._e_2).view(batchsize, 1)
                else:
                    # brute_force or hutchinson's estimator
                    divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)
                    
        if self.residual:
            dy = dy - y
            divergence -= torch.ones_like(divergence) * torch.tensor(np.prod(y.shape[1:]), dtype=torch.float32
                                                                     ).to(divergence)
        return tuple([dy, -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])


class AutoencoderODEfunc(nn.Module):

    def __init__(self, autoencoder_diffeq, divergence_fn="approximate", residual=False, rademacher=False):
        assert divergence_fn in ("approximate"), "Only approximate divergence supported at the moment. (TODO)"
        assert isinstance(autoencoder_diffeq, AutoencoderDiffEqNet)
        super(AutoencoderODEfunc, self).__init__()
        self.residual = residual
        self.autoencoder_diffeq = autoencoder_diffeq
        self.rademacher = rademacher

        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def forward(self, t, y_and_logpy):
        y, _ = y_and_logpy  # remove logpy

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            h, dy = self.autoencoder_diffeq(t, y)

            # Sample and fix the noise.
            if self._e is None:
                if self.rademacher:
                    self._e = sample_rademacher_like(h)
                else:
                    self._e = sample_gaussian_like(h)

            e_vjp_dhdy = torch.autograd.grad(h, y, self._e, create_graph=True)[0]
            e_vjp_dfdy = torch.autograd.grad(dy, h, e_vjp_dhdy, create_graph=True)[0]
            divergence = torch.sum((e_vjp_dfdy * self._e).view(batchsize, -1), 1, keepdim=True)

        if self.residual:
            dy = dy - y
            divergence -= torch.ones_like(divergence) * torch.tensor(np.prod(y.shape[1:]), dtype=torch.float32
                                                                     ).to(divergence)

        return dy, -divergence
