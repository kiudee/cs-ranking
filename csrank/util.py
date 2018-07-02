import inspect
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

__all__ = ['create_dir_recursively', 'duration_till_now', 'print_dictionary', 'rename_file_if_exist', 'seconds_to_time',
           'time_from_now', 'get_duration_seconds', 'setup_logger']


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


def setup_logger(log_path=None):
    """Function setup as many loggers as you want"""
    if log_path is None:
        dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        dirname = os.path.dirname(dirname)
        log_path = os.path.join(dirname, "experiments", "logs", "logs.log")
        create_dir_recursively(log_path, True)
    logging.basicConfig(filename=log_path, level=logging.DEBUG,
                        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
