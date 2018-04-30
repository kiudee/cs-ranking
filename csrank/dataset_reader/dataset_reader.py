import inspect
import logging
import os
from abc import ABCMeta, abstractmethod

from csrank.constants import OBJECT_RANKING, DYAD_RANKING, EXCEPTION_OBJECT_ARRAY_SHAPE, \
    EXCEPTION_RANKINGS_FEATURES_INSTANCES, EXCEPTION_RANKINGS_FEATURES_NO_OF_OBJECTS, \
    EXCEPTION_UNWANTED_CONTEXT_FEATURES, LABEL_RANKING, DISCRETE_CHOICE, EXCEPTION_CONTEXT_ARRAY_SHAPE, \
    CHOICE_FUNCTIONS, EXCEPTION_SET_INCLUSION


class DatasetReader(metaclass=ABCMeta):
    def __init__(self, dataset_folder="", learning_problem=OBJECT_RANKING, **kwargs):
        self.dr_logger = logging.getLogger("DatasetReader")
        self.dr_logger.info("Learning Problem: {}".format(learning_problem))
        self.X = None
        self.rankings = None
        self.Xc = None
        self.choice_set = None
        self.learning_problem = learning_problem
        dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).replace("dataset_reader",
            "datasets")
        if dataset_folder is not None:
            self.dirname = os.path.join(dirname, dataset_folder)
            if not os.path.exists(self.dirname):
                self.dr_logger.warning("Path given for dataset does not exit {}".format(self.dirname))
        else:
            self.dirname = None

    @abstractmethod
    def __load_dataset__(self):
        raise NotImplementedError

    def __check_dataset_validity__(self):
        if self.X is not None or self.rankings is not None:
            if self.learning_problem == OBJECT_RANKING or self.learning_problem == DYAD_RANKING:
                assert len(self.X.shape) == 3, EXCEPTION_OBJECT_ARRAY_SHAPE.format(self.learning_problem, self.X.shape)
                n_instances, n_objects, n_features = self.X.shape
                assert (n_instances == self.rankings.shape[0]), EXCEPTION_RANKINGS_FEATURES_INSTANCES.format(
                    self.learning_problem, self.rankings.shape[0], n_instances)
                assert (n_objects == self.rankings.shape[1]), EXCEPTION_RANKINGS_FEATURES_NO_OF_OBJECTS.format(
                    self.rankings.shape[1], n_objects)
                if self.learning_problem == OBJECT_RANKING:
                    assert self.Xc is None, EXCEPTION_UNWANTED_CONTEXT_FEATURES.format(self.learning_problem)

            if self.learning_problem == LABEL_RANKING:
                assert len(self.X.shape) == 2, EXCEPTION_CONTEXT_ARRAY_SHAPE.format(self.learning_problem, self.X.shape)
                n_instances, n_features = self.X.shape
                assert (n_instances == self.rankings.shape[0]), EXCEPTION_RANKINGS_FEATURES_INSTANCES.format(
                    self.learning_problem, self.rankings.shape[0], n_instances)
                assert self.Xc is None, EXCEPTION_UNWANTED_CONTEXT_FEATURES.format(self.learning_problem)

            if self.learning_problem in [DISCRETE_CHOICE, CHOICE_FUNCTIONS]:
                assert len(self.X.shape) == 3, EXCEPTION_OBJECT_ARRAY_SHAPE.format(self.learning_problem, self.X.shape)
                n_instances, n_objects, n_features = self.X.shape
                assert (n_instances == self.choice_set.shape[0]), EXCEPTION_RANKINGS_FEATURES_INSTANCES.format(
                    self.learning_problem, self.choice_set.shape[0], n_instances)
                assert self.Xc is None, EXCEPTION_UNWANTED_CONTEXT_FEATURES.format(self.learning_problem)
                assert self.rankings is None, EXCEPTION_UNWANTED_CONTEXT_FEATURES.format(self.learning_problem)
                if self.learning_problem == CHOICE_FUNCTIONS:
                    assert (n_objects == self.choice_set.shape[1]), EXCEPTION_SET_INCLUSION

            if self.learning_problem == DYAD_RANKING:
                assert len(self.Xc.shape) == 2, EXCEPTION_CONTEXT_ARRAY_SHAPE.format(self.learning_problem,
                    self.Xc.shape)
                n_instances, n_features = self.Xc.shape
                assert (n_instances == self.rankings.shape[0]), EXCEPTION_RANKINGS_FEATURES_INSTANCES.format(
                    self.learning_problem, self.rankings.shape[0], n_instances)

    @abstractmethod
    def splitter(self, iter):
        raise NotImplementedError

    @abstractmethod
    def get_train_test_datasets(self, n_datasets):
        raise NotImplementedError

    @abstractmethod
    def get_complete_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def get_single_train_test_split(self):
        raise NotImplementedError
