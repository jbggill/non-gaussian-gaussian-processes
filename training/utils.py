import math
import os
from math import pi
import sys
sys.path.append('/Users/jessegill/Desktop/nggp/nggp_lib')
import matplotlib.pyplot as plt
import numpy as np
import six
import torch

import CNF_lib.layers as layers
import CNF_lib.layers.wrappers.cnf_regularization as reg_lib
import CNF_lib.spectral_norm as spectral_norm
from CNF_lib.layers.odefunc import divergence_bf, divergence_approx


def prepare_for_plots(pred, y_true, sample_fn, context, new_means):
    samples = []
    flow_samples = []
    true_y = []
    flow_y = []
    gauss_y = []
    device = torch.device("cpu")
    for k in range(pred.mean.shape[0]):
        sample = torch.normal(pred.mean[k], pred.stddev[k], size=(1, 10000)).reshape(10000, 1)
        samples.append(sample)
        true_y.append(y_true[k])
        gauss_y.append(pred.mean[k])
        if new_means is not None:
            if context is not None:
                flow_sample = sample_fn(sample.to(device), context[k].repeat(10000, 1))
            else:
                flow_sample = sample_fn(sample.to(device))
            flow_samples.append(flow_sample)
            flow_y.append(new_means[k])
    if new_means is not None:
        return samples, true_y, gauss_y, flow_samples, flow_y
    else:
        return samples, true_y, gauss_y, None, None


def plot_histograms(path, samples, true_y, gauss_y,  n_support, flow_samples=None, flow_y=None, name_nr=0):

    f, a = plt.subplots(5, 4, figsize=(30, 30))
    a = a.ravel()
    for idx, ax in enumerate(a):
        if idx < len(samples):
            single_guass = samples[idx].cpu().detach().numpy()
            ax.hist(single_guass, 50, density=True, facecolor='g', alpha=0.75)
            if flow_samples is not None:
                single_flow = flow_samples[idx].cpu().detach().numpy()
                ax.hist(single_flow, 50, density=True, facecolor='r', alpha=0.75)
                ax.axvline(flow_y[idx].cpu().detach().numpy(), 0, 1.6, color='r')
            ax.axvline(gauss_y[idx].cpu().detach().numpy(), 0, 1.6, color='g')
            ax.axvline(true_y[idx].cpu().detach().numpy(), 0, 1.6, color='b')
    plt.tight_layout()

    plt.grid(True)
    plt.savefig(os.path.join(path, "image_" + str(n_support) + "_{}.png".format(name_nr)))

def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)


def DBindex(cl_data_file):
    class_list = cl_data_file.keys()
    cl_num = len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append(np.mean(cl_data_file[cl], axis=0))
        stds.append(np.sqrt(np.mean(np.sum(np.square(cl_data_file[cl] - cl_means[-1]), axis=1))))

    mu_i = np.tile(np.expand_dims(np.array(cl_means), axis=0), (len(class_list), 1, 1))
    mu_j = np.transpose(mu_i, (1, 0, 2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis=2))

    for i in range(cl_num):
        DBs.append(np.max([(stds[i] + stds[j]) / mdists[i, j] for j in range(cl_num) if j != i]))
    return np.mean(DBs)


def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x != 0) for x in cl_data_file[cl]]))

    return np.mean(cl_sparsity)


def normal_logprob(z, mu, sigma):
    log_z = -0.5 * torch.log(2 * pi * sigma)
    z_diff = z - mu
    return log_z - 0.5 * z_diff.pow(2) / sigma.pow(2)


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def set_cnf_options(args, model):
    def _set(module):
        if isinstance(module, layers.CNF):
            # Set training settings
            module.solver = args.solver
            module.atol = args.atol
            module.rtol = args.rtol
            if args.step_size is not None:
                module.solver_options['step_size'] = args.step_size

            # If using fixed-grid adams, restrict order to not be too high.
            if args.solver in ['fixed_adams', 'explicit_adams']:
                module.solver_options['max_order'] = 4

            # Set the test settings
            module.test_solver = args.test_solver if args.test_solver else args.solver
            module.test_atol = args.test_atol if args.test_atol else args.atol
            module.test_rtol = args.test_rtol if args.test_rtol else args.rtol

        if isinstance(module, layers.ODEfunc):
            module.rademacher = args.rademacher
            module.residual = args.residual

    model.apply(_set)


def override_divergence_fn(model, divergence_fn):
    def _set(module):
        if isinstance(module, layers.ODEfunc):
            if divergence_fn == "brute_force":
                module.divergence_fn = divergence_bf
            elif divergence_fn == "approximate":
                module.divergence_fn = divergence_approx

    model.apply(_set)


def count_nfe(model):
    class AccNumEvals(object):

        def __init__(self):
            self.num_evals = 0

        def __call__(self, module):
            if isinstance(module, layers.ODEfunc):
                self.num_evals += module.num_evals()

    accumulator = AccNumEvals()
    model.apply(accumulator)
    return accumulator.num_evals


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_time(model):
    class Accumulator(object):

        def __init__(self):
            self.total_time = 0

        def __call__(self, module):
            if isinstance(module, layers.CNF):
                self.total_time = self.total_time + module.sqrt_end_time * module.sqrt_end_time

    accumulator = Accumulator()
    model.apply(accumulator)
    return accumulator.total_time


def add_spectral_norm(model, logger=None):
    """Applies spectral norm to all modules within the scope of a CNF."""

    def apply_spectral_norm(module):
        if 'weight' in module._parameters:
            if logger: logger.info("Adding spectral norm to {}".format(module))
            spectral_norm.inplace_spectral_norm(module, 'weight')

    def find_cnf(module):
        if isinstance(module, layers.CNF):
            module.apply(apply_spectral_norm)
        else:
            for child in module.children():
                find_cnf(child)

    find_cnf(model)


def spectral_norm_power_iteration(model, n_power_iterations=1):
    def recursive_power_iteration(module):
        if hasattr(module, spectral_norm.POWER_ITERATION_FN):
            getattr(module, spectral_norm.POWER_ITERATION_FN)(n_power_iterations)

    model.apply(recursive_power_iteration)


REGULARIZATION_FNS = {
    "l1int": reg_lib.l1_regularzation_fn,
    "l2int": reg_lib.l2_regularzation_fn,
    "dl2int": reg_lib.directional_l2_regularization_fn,
    "JFrobint": reg_lib.jacobian_frobenius_regularization_fn,
    "JdiagFrobint": reg_lib.jacobian_diag_frobenius_regularization_fn,
    "JoffdiagFrobint": reg_lib.jacobian_offdiag_frobenius_regularization_fn,
}

INV_REGULARIZATION_FNS = {v: k for k, v in six.iteritems(REGULARIZATION_FNS)}


def append_regularization_to_log(log_message, regularization_fns, reg_states):
    for i, reg_fn in enumerate(regularization_fns):
        log_message = log_message + " | " + INV_REGULARIZATION_FNS[reg_fn] + ": {:.8f}".format(reg_states[i].item())
    return log_message


def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if getattr(args, arg_key) is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(eval("args." + arg_key))

    regularization_fns = tuple(regularization_fns)
    regularization_coeffs = tuple(regularization_coeffs)
    return regularization_fns, regularization_coeffs


def get_regularization(model, regularization_coeffs):
    if len(regularization_coeffs) == 0:
        return None

    acc_reg_states = tuple([0.] * len(regularization_coeffs))
    for module in model.modules():
        if isinstance(module, layers.CNF):
            acc_reg_states = tuple(acc + reg for acc, reg in zip(acc_reg_states, module.get_regularization_states()))
    return acc_reg_states


def build_model_tabular(args, dims, regularization_fns=None):
    hidden_dims = tuple(map(int, args.dims.split("-")))

    def build_cnf():
        diffeq = layers.ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            strides=None,
            conv=False,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
        )
        odefunc = layers.ODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=args.residual,
            rademacher=args.rademacher,
        )
        cnf = layers.CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            regularization_fns=regularization_fns,
            solver=args.solver,
        )
        return cnf

    chain = [build_cnf() for _ in range(args.num_blocks)]
    if args.batch_norm:
        bn_layers = [layers.MovingBatchNorm1d(dims, bn_lag=args.bn_lag) for _ in range(args.num_blocks)]
        bn_chain = [layers.MovingBatchNorm1d(dims, bn_lag=args.bn_lag)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = layers.SequentialFlow(chain)

    set_cnf_options(args, model)

    return model


def build_conditional_cnf(args, dims, context_dim, regularization_fns=None):
    hidden_dims = tuple(map(int, args.dims.split("-")))

    def build_cnf():
        diffeq = layers.ODEnetC(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            context_dim=context_dim,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
        )
        odefunc = layers.ODEfuncC(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            conditional=True,
            residual=args.residual,
            rademacher=args.rademacher,
        )
        cnf = layers.CNFC(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            regularization_fns=regularization_fns,
            solver=args.solver,
        )
        return cnf

    chain = [build_cnf() for _ in range(args.num_blocks)]
    if args.batch_norm:
        bn_layers = [layers.MovingBatchNorm1d(dims, bn_lag=args.bn_lag) for _ in range(args.num_blocks)]
        bn_chain = [layers.MovingBatchNorm1d(dims, bn_lag=args.bn_lag)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = layers.SequentialFlowC(chain)

    set_cnf_options(args, model)

    return model
