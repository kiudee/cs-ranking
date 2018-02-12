import ast
import glob
import inspect
import os
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np
from skopt import Optimizer
from skopt import load, dump
from skopt.space import check_dimension
from skopt.utils import cook_estimator, normalize_dimensions

from csrank.util import print_dictionary
from experiments.util import CMPNET, BORDA_ZERO, BORDA, RANKNET, object_rankers

OPTIMIZER_SINGLE_FOLD = "optimizer_single_fold"


def get_seed(lines):
    for l in lines:
        if "Random seed for the optimizer:" in l:
            opt_seed = int(l.split("Random seed for the optimizer: ")[-1])
        if "Random seed for the GP surrogate: " in l:
            gp_seed = int(l.split("Random seed for the GP surrogate: ")[-1])
    return gp_seed, opt_seed


def create_opt(lines, ranker_name):
    gp_seed, opt_seed = get_seed(lines)
    _ranker_class = object_rankers[ranker_name]
    _ranker_class._use_early_stopping = True
    param_ranges = _ranker_class.set_tunable_parameter_ranges({})
    transformed = []
    for param in param_ranges:
        transformed.append(check_dimension(param))
    space = normalize_dimensions(transformed)
    base_estimator = cook_estimator("GP", space=space, random_state=gp_seed, noise="gaussian")
    optimizer = Optimizer(dimensions=param_ranges, random_state=opt_seed,
                          base_estimator=base_estimator)
    return optimizer


def renew_optimizer(log_file):
    file_name = os.path.split(log_file)[-1].split('.log')[0]
    opt_path = os.path.join(DIR_PATH, OPTIMIZER_SINGLE_FOLD, file_name)
    lines = [line.rstrip('\n') for line in open(log_file)]
    model_name = os.path.split(log_file)[-1].split('_')[-1].split(".")[0]
    if model_name == "zero": model_name = BORDA_ZERO
    try:
        optimizer = load(opt_path)
    except EOFError:
        optimizer = create_opt(lines, model_name)
    except FileNotFoundError:
        optimizer = create_opt(lines, model_name)
    print(optimizer.acq_func)

    n = 0
    for i, l in enumerate(lines):

        if "Next parameters:" in l:
            p = lines[i + 1]
            parameters = ast.literal_eval(p)
            n = n + 1
        if "Validation error for the parameters is " in l:
            loss = l.split("Validation error for the parameters is ")[-1]
            loss = float(loss)
            n = n + 1
        if "Time taken for the parameters is " in l:
            time = l.split("Time taken for the parameters is ")[-1]

            d = datetime.strptime(time.split('.')[0], '%H:%M:%S')

            time = timedelta(hours=d.hour, minutes=d.minute, seconds=d.second).total_seconds()
            n = n + 1
        if n == 3:
            if parameters not in optimizer.Xi:
                if "ps" in optimizer.acq_func:
                    optimizer.tell(parameters, [loss, time])
                else:
                    if model_name in [BORDA_ZERO, BORDA, RANKNET, CMPNET]:
                        parameters = parameters[0:-1]
                        parameters.extend([0.01, 1024, 300])
                    try:
                        optimizer.tell(parameters, loss)
                    except ValueError:
                        parameters[2] = 0.01
                        optimizer.tell(parameters, loss)
            n = 0
    if "ps" in optimizer.acq_func:
        best_i = np.argmin(np.array(optimizer.yi)[:, 0])
        best_loss = optimizer.yi[best_i]
        best_params = optimizer.Xi[best_i]
        print(
            "Best parameters so far with a loss of {:.4f} time of {:.4f}:\n {}".format(best_loss[0],
                                                                                       best_loss[1],
                                                                                       best_params))
    else:

        best_i = np.argmin(optimizer.yi)
        best_loss = optimizer.yi[best_i]
        best_params = optimizer.Xi[best_i]
        print(
            "Best parameters so far with a loss of {:.4f}:\n {}".format(best_loss, best_params))
    print("Main optimizer iterations done {} and saving the model".format(np.array(optimizer.yi).shape[0]))
    dump(optimizer, opt_path)


def check_optimizers(opt_path):
    try:
        optimizer = load(opt_path)
        if "ps" in optimizer.acq_func:
            best_i = np.argmin(np.array(optimizer.yi)[:, 0])
            best_loss = optimizer.yi[best_i]
            best_params = optimizer.Xi[best_i]
            print("Best parameters so far with a loss of {:.4f} time of {:.4f}:\n {}".format(best_loss[0], best_loss[1],
                                                                                             best_params))
        else:
            best_i = np.argmin(optimizer.yi)
            best_loss = optimizer.yi[best_i]
            best_params = optimizer.Xi[best_i]
            print(
                "Best parameters so far with a loss of {:.4f}:\n {}".format(best_loss, best_params))
        print("Main optimizer iterations done {} and saving the model".format(np.array(optimizer.yi).shape[0]))
        next_point = optimizer.ask()
        print('Next parameters:\n{}'.format(next_point))
        return np.array(optimizer.yi).shape[0]
    except EOFError:
        print("File corrupted")
        return 0


if __name__ == '__main__':
    DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # log = os.path.join(DIR_PATH, "logs_single_fold", "depth_old", "*.log")
    #
    # for log_path in glob.glob(log):
    #     fname = log_path.split('/')[-1]
    #     print(fname)
    #     renew_optimizer(log_path)
    iterations = dict()
    for path in glob.glob(os.path.join(DIR_PATH, OPTIMIZER_SINGLE_FOLD, "dataset_type_*")):
        print(path.split('/')[-1])
        iterations[path.split('/')[-1]] = check_optimizers(path)
    iterations = OrderedDict(sorted(iterations.items()))
    print(print_dictionary(iterations))
    l = 0
    for k, v in iterations.items():
        if v < 5:
            print(k)
            print(5 - v)
            l += 5 - v
    print("Total {}".format(l))
