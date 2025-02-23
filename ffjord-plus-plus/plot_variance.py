import matplotlib.pylab as pl
import numpy as np
import os
import ot
import argparse
import torch

import lib.toy_data as toy_data
import lib.layers.odefunc as odefunc

from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular
from train_toy import get_transforms
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='swissroll'
)
parser.add_argument(
    "--layer_type", type=str, default="concatsquash",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--dims', type=str, default='64-64-64')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="hutchplusplus_m", choices=["brute_force", "approximate", "approximate_m", "hutchplusplus", "hutchplusplus_m", "na_hutchplusplus"])
parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES)

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--niters', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)

# Track quantities
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

parser.add_argument('--save', type=str, default='exp_checkerboard_t')
parser.add_argument('--viz_freq', type=int, default=100)
parser.add_argument('--val_freq', type=int, default=100)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args(args=[])
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
regularization_fns, regularization_coeffs = create_regularization_fns(args)
p_samples = toy_data.inf_train_gen(args.data, batch_size=100**2)

# save source data
source_data_path = f'./vis_eval/{args.data}/source_data_1w.npy'
np.save(source_data_path, p_samples)

# model = build_model_tabular(args, 2, regularization_fns).to(device)
# if args.spectral_norm: add_spectral_norm(model)
# set_cnf_options(args, model)
# ckpt = torch.load('exp_checkerboard_hpp/checkpt.pth')
# model.load_state_dict(ckpt['state_dict'])

# with torch.no_grad():
#     model.eval()
#     p_samples = toy_data.inf_train_gen(args.data, batch_size=100**2)

#     # save source data
#     source_data_path = './vis_eval/source_data.npy'
#     np.save(source_data_path, p_samples)

#     # sample 
#     sample_fn, density_fn = get_transforms(model)
#     z = torch.randn(800 * 800, 2).type(torch.float32).to(device)
#     zk = []
#     memory = 100
#     inds = torch.arange(0, z.shape[0]).to(torch.int64)
#     for ii in torch.split(inds, int(memory**2)):
#         zk.append(sample_fn(z[ii]))
#     zk = torch.cat(zk, 0).cpu().numpy()

# # save sampling data
# sampling_data_path = './vis_eval/sampling_hpp_data.npy'
# np.save(sampling_data_path, zk)

