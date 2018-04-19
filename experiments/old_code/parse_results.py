import glob
import inspect
import os

import pandas as pd

from csrank.constants import OBJECT_RANKING
from csrank.util import create_dir_recursively
from experiments.util import dataset_options_dict, rankers_dict, lp_metric_dict

ZERO_ONE_RANK_ACCURACY = "ZeroOneRankAccuracy"

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# TODO check how to combine resutls for different dataset parameters
if __name__ == '__main__':
    for dataset_name in dataset_options_dict[OBJECT_RANKING].keys():
        print("Dataset {}".format(dataset_name))
        data = []
        columns = ["Ranker"] + list(lp_metric_dict[OBJECT_RANKING].keys())
        columns.append(ZERO_ONE_RANK_ACCURACY)
        for ranker_name in rankers_dict[OBJECT_RANKING].keys():
            one_row = ["{}".format(ranker_name.upper())]
            df_path = os.path.join(DIR_PATH, "multiple_cv_results",
                ('{}_{}' + '.csv').format(dataset_name, ranker_name))
            if os.path.isfile(df_path):
                df = pd.read_csv(df_path)
                df[ZERO_ONE_RANK_ACCURACY] = 1.0 - df["ZeroOneRankLossTies"]
                std = df.std(axis=0).values
                mean = df.mean(axis=0).values
                one_row.extend(["{:.3f}+-{:.3f}".format(m, s) for m, s in zip(mean, std)])
                data.append(one_row)
        if (len(data) > 0):
            df_path = os.path.join(DIR_PATH, "multiple_cv_results",
                ('{}_{}' + '.csv').format(dataset_name, "dataset"))
            create_dir_recursively(df_path, is_file_path=True)
            dataFrame = pd.DataFrame(data, columns=columns)
            dataFrame.to_csv(df_path)
            del dataFrame["KendallsTau"]
            del dataFrame["ZeroOneRankLossTies"]
            del dataFrame["ZeroOneRankLoss"]
            print(dataFrame.to_latex())
            del dataFrame
    df_paths = os.path.join(DIR_PATH, "single_cv_results", '*.csv')
    for file_path in glob.glob(df_paths):
        print("Dataset {}".format(file_path.split('/')[-1]))
        dataFrame = pd.read_csv(file_path)
        dataFrame = dataFrame.round(3)
        del dataFrame["KendallsTau"]
        del dataFrame["ZeroOneRankLoss"]
        if "ZeroOneAccuracy" in dataFrame:
            del dataFrame["ZeroOneAccuracy"]
        dataFrame[ZERO_ONE_RANK_ACCURACY] = 1.0 - dataFrame["ZeroOneRankLossTies"]
        del dataFrame["ZeroOneRankLossTies"]
        print(dataFrame.to_latex())
