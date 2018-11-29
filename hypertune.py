from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

import torch
from train import diffraction_train
from config import TrainConfig, Dropouts
import argparse
import os
from model import CRNN, ConvNet
import time

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Ugly way of defining fixed parameters
savedir = '/playpen/zytang/hypertune'
inputdir = '/playpen/zytang/Ambisonic_houses_features'
batch_size = 512
modelname = 'CRNN'
dropout = 0
epochs = 30


def black_box_function(learning_rate):
    results_dir = os.path.join(savedir,
                               "results" + '_{}'.format(modelname) + '_lr{}'.format(learning_rate) + '_bs{}'.format(
                                   batch_size) + '_drop{}'.format(dropout))
    print('writing results to {}'.format(results_dir))

    dropouts = Dropouts(dropout, dropout, dropout)
    if modelname == "CNN":
        model_choice = ConvNet(device, dropouts).to(device)
    elif modelname == "CRNN":
        model_choice = CRNN(device, dropouts).to(device)

    config = TrainConfig() \
        .set_data_folder(inputdir) \
        .set_learning_rate(learning_rate) \
        .set_batch_size(batch_size) \
        .set_num_epochs(epochs) \
        .set_test_to_all_ratio(0.1) \
        .set_results_dir(results_dir) \
        .set_model(model_choice)

    # negative sign for minimization
    return -diffraction_train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='hypertune',
                                     description="""Script to tune hyperparameters for deep learning""")
    parser.add_argument("--logdir", "-l", default=savedir, help="Directory to write logfiles", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logpath = os.path.join(args.logdir, 'logs_bs{}_{}.json'.format(batch_size, time.strftime("%Y_%m_%d_%H_%M_%S")))
    print('writing log file to {}'.format(logpath))

    # Bounded region of parameter space
    pbounds = {'learning_rate': (0, 1)}

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )

    logger = JSONLogger(path=logpath)
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=100,
        n_iter=3,
    )

    print(optimizer.max)
