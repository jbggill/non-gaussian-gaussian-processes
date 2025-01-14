import logging
import os

import numpy as np
import torch
from methods.regression_methods import NGGP 
from models import backbone
from reporting.loggers import ResultsLogger
from training.configs import Config
from training.io_utils import parse_args_regression
from training.utils import add_spectral_norm
from training.utils import build_model_tabular, build_conditional_cnf
from training.utils import create_regularization_fns
from training.utils import set_cnf_options


def main():
    params = parse_args_regression()
    setup_seed(params)
    config = Config(params)
    checkpoint_dir, save_path = setup_checkpoint_dir(params)

    results_logger = ResultsLogger(params)


    device = 'cpu'
    logging.info('Device: {}'.format(device))
    bb = setup_backbone(device, params)
    model = setup_model(bb, config, device, params)
    optimizer = setup_optimizer(model, params)
    if params.test:
        test(model, params, save_path, results_logger)
    else:
        train(model, optimizer, params, save_path, results_logger)

    results_logger.save()


def train(model, optimizer, params, save_path, results_logger):
    for epoch in range(params.stop_epoch):
        model.train_loop(epoch, optimizer, params, results_logger)
    model.save_checkpoint(save_path)
    return model


def test(model, params, save_path, results_logger):
    model.load_checkpoint(save_path, device=params.device)
    for epoch in range(params.n_test_epochs):
        print('going into' , type(model), 'test loop')
        res = model.test_loop(params.n_support, params, os.path.dirname(save_path))

        detached_res = []
        for r in res:
            detached = r.cpu().detach().numpy()
            detached_res.append(detached)
    
        names = ["mse_list", "nll_list", "mean", "lower", "upper", "x", "y"]
        for i, r in enumerate(detached_res):
            results_logger.log(names[i], r)

    print("-------------------")
    MSE = results_logger.get_array("mse_list")
    print("Average MSE: " + str(np.mean(MSE)) + " +- " + str(np.std(MSE)))
    print("-------------------")
    print("-------------------")
    NLL = results_logger.get_array("nll_list")
    print("Average NLL: " + str(np.mean(NLL)) + " +- " + str(np.std(NLL)))
    print("-------------------")

    results_logger.log("mse", np.mean(MSE))
    results_logger.log("mse_std", np.std(MSE))

    results_logger.log("nll", np.mean(NLL))
    results_logger.log("nll_std", np.std(NLL))


def setup_seed(params):
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_checkpoint_dir(params):
    checkpoint_dir = '%s/checkpoints/%s/' % (params.save_dir, params.dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = '%s/checkpoints/%s/%s_%s_model.th' % (params.save_dir, params.dataset, params.model, params.method)
    return checkpoint_dir, save_path


def setup_optimizer(model, params):
    method_lr = params.all_lr if params.all_lr is not None else params.method_lr
    feature_extractor_lr = params.all_lr if params.all_lr is not None else params.feature_extractor_lr
    cnf_lr = params.all_lr if params.all_lr is not None else params.cnf_lr

    params_groups = [{'params': model.model.parameters(), 'lr': method_lr},
                     {'params': model.feature_extractor.parameters(), 'lr': feature_extractor_lr}]
    if params.method == "NGGP":
        params_groups.append({'params': model.cnf.parameters(), 'lr': cnf_lr})

    optimizer = torch.optim.Adam(params_groups)
    return optimizer


def setup_model(bb, config, device, params):
    if params.method == "NGGP":
        if params.context_type == 'backbone' and params.dataset!="QMUL":
            context_dim = bb.output_dim
        elif params.context_type == 'backbone' and params.dataset=="QMUL":
            context_dim = config.nn_config["input_dim"]
        elif params.context_type == 'nn':
            context_dim = config.nn_config["output_dim"]
        else:
            raise ValueError("unknown context type")
        cnf = setup_flow(device, params, context_dim)
        model = NGGP(bb, device, num_tasks=params.num_tasks, config=config,
                         dataset=params.dataset, cnf=cnf, use_conditional=params.use_conditional,
                         add_noise=params.add_noise, context_type=params.context_type,
                         multi_type=params.multi_type)
    else:
        if params.method == 'DKT':
            model = NGGP(bb, device, num_tasks=params.num_tasks, config=config, dataset=params.dataset,
                             cnf=None, multi_type=params.multi_type)
        else:
            raise ValueError('Unrecognised method')
    return model


def setup_flow(device, params, context_dim):
    if params.use_conditional:
        cnf = build_conditional_cnf(params, params.num_tasks, context_dim).to(device)
    else:
        regularization_fns, regularization_coeffs = create_regularization_fns(params)
        cnf = build_model_tabular(params, params.num_tasks, regularization_fns).to(device)
    if params.spectral_norm:
        add_spectral_norm(cnf)
    set_cnf_options(params, cnf)
    return cnf


def setup_backbone(device, params):
    if params.model == "MLP2":
        bb = backbone.MLP2(input_dim=1, output_dim=params.output_dim).to(device)
    elif params.model == "Encoder":
        bb = backbone.Encoder().to(device)
    elif params.model == "Conv3":
        bb = backbone.Conv3().to(device)
    else:
        raise ValueError("Unknown backbone model {}".format(params.model))

    return bb


if __name__ == "__main__":
    main()
