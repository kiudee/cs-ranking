import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from csrank.tensorflow_util import get_tensor_value

__all__ = ['create_dir_recursively', 'duration_till_now', 'get_loss_for_array', 'get_mean_loss_for_dictionary',
           'print_dictionary', 'rename_file_if_exist', 'seconds_to_time', 'time_from_now', 'get_duration_seconds']


def print_dictionary(dictionary):
    output = "\n"
    for key, value in dictionary.items():
        output = output + str(key) + " => " + str(value) + "\n"
    return output


def create_dir_recursively(path, is_file_path=False):
    if is_file_path:
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def rename_file_if_exist(file_path):
    my_file = Path(file_path)
    try:
        extension = '.' + file_path.split('.')[1]
    except IndexError:
        extension = ''
    path = file_path.split('.')[0]
    i = 1
    while my_file.is_file():
        file_path = path + str(i) + extension
        my_file = Path(file_path)
        i += 1
    return file_path


def get_mean_loss_for_dictionary(metric, y_true, y_pred):
    losses = []
    total_instances = 0
    for n in y_pred.keys():
        loss = get_loss_for_array(metric, y_true[n], y_pred[n])
        if loss is not np.nan:
            loss = loss * y_pred[n].shape[0]
            total_instances += y_pred[n].shape[0]
            losses.append(loss)
    losses = np.array(losses)
    weighted_mean = np.sum(losses) / total_instances
    return weighted_mean


def get_loss_for_array(metric, y_true, y_pred):
    x = metric(y_true, y_pred)
    x = get_tensor_value(x)
    return np.nanmean(x)


def get_duration_seconds(duration):
    time = int(re.findall(r'\d+', duration)[0])
    d = duration.split(str(time))[1].upper()
    options = {"D": 24 * 60 * 60, "H": 60 * 60, "M": 60}
    return options[d] * time


def duration_till_now(start):
    return (datetime.now() - start).total_seconds()


def time_from_now(target_time_sec):
    base_datetime = datetime.now()
    delta = timedelta(seconds=target_time_sec)
    target_date = base_datetime + delta
    return target_date.strftime("%Y-%m-%d %H:%M:%S")


def seconds_to_time(target_time_sec):
    return str(timedelta(seconds=target_time_sec))


def convert_to_loss(loss_function):
    def loss(y_true, y_pred):
        return 1.0 - loss_function(y_true, y_pred)

    return loss