from datetime import datetime
from datetime import timedelta
import inspect
import logging
import os
from pathlib import Path
import re
import sys

from csrank.metrics_np import f1_measure
from csrank.metrics_np import hamming
from csrank.metrics_np import instance_informedness
from csrank.metrics_np import precision
from csrank.metrics_np import recall
from csrank.metrics_np import subset_01_loss
from csrank.metrics_np import zero_one_accuracy_np

__all__ = [
    "create_dir_recursively",
    "duration_till_now",
    "print_dictionary",
    "rename_file_if_exist",
    "seconds_to_time",
    "time_from_now",
    "get_duration_seconds",
    "setup_logging",
    "progress_bar",
]


metrics_on_predictions = [
    f1_measure,
    precision,
    recall,
    subset_01_loss,
    hamming,
    instance_informedness,
    zero_one_accuracy_np,
]


class MissingExtraError(ImportError):
    """Indicates an ImportError that can be fixed by installing an extra"""

    def __init__(self, package, extra):
        super(MissingExtraError, self).__init__(
            f"Could not import the optional dependency {package}. "
            f'Please install it or specify the "{extra}" extra when installing this package.'
        )


def progress_bar(count, total, status=""):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    bar = "=" * filled_len + "-" * (bar_len - filled_len)

    sys.stdout.write("[%s] %s/%s ...%s\r" % (bar, count, total, status))
    sys.stdout.flush()


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
        extension = "." + file_path.split(".")[1]
    except IndexError:
        extension = ""
    path = file_path.split(".")[0]
    i = 1
    while my_file.is_file():
        file_path = path + str(i) + extension
        my_file = Path(file_path)
        i += 1
    return file_path


def get_duration_seconds(duration):
    time = int(re.findall(r"\d+", duration)[0])
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


def setup_logging(log_path=None, level=logging.DEBUG):
    """Function setup as many logging for the experiments"""
    if log_path is None:
        dirname = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
        dirname = os.path.dirname(dirname)
        log_path = os.path.join(dirname, "experiments", "logs", "logs.log")
        create_dir_recursively(log_path, True)
    logging.basicConfig(
        filename=log_path,
        level=level,
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("SetupLogger")
    logger.info("log file path: {}".format(log_path))
