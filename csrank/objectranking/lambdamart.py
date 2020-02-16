import logging, math
from collections import deque
from multiprocessing import Pool
from itertools import chain

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from csrank.learner import Learner
from csrank.objectranking.object_ranker import ObjectRanker


class LambdaMART(ObjectRanker, Learner):
    def __init__(
        self,
        n_objects=None,
        n_object_features=None,
        number_of_trees=5,
        learning_rate=1e-3,
        min_samples_split=2,
        max_depth=50,
        min_samples_leaf=1,
        max_leaf_nodes=None,
        num_process=None,
        criterion="mse",
        splitter="best",
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        min_impurity_decrease=0.0,
        min_impurity_split=1e-7,
        **kwargs
    ):
        """
        Create a LambdaMART based rank regression model. This model uses an 
        ensemble of trees that learn to predict the relevance scores of 
        the documents based on the features, which then can be turned into 
        rankings. The base learner used is the implementation of Decision 
        Tree from the sklearn tree package. The learner tries to indirectly 
        optimize the nDCG metric by learning the lambdas.

        Parameters
        ----------
        n_object_features : int
            Number of features of the object space
        n_objects : int
            Number of objects
        number_of_trees : int
            The maximum number of trees that are to be trained for the ensemble.
        learning_rate : float
            learning rate for the LambdaMART algorithm
        min_samples_split : int
            Number of samples required to split the internal node
        max_depth : int
            Maximum depth of the tree
        min_samples_leaf : int
            Minimum number of samples required to be at the leaf node

        References
        ----------
            [1] Burges, Chris J.C. (2010, June). "From RankNet to LambdaRank to LambdaMART: An Overview"
        """
        self.n_object_features = n_object_features
        self.n_objects = n_objects
        self.number_of_trees = number_of_trees
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.num_process = num_process
        self.ensemble = []
        self.random_state = random_state
        self.criterion = criterion
        self.splitter = splitter
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.logger = logging.getLogger(LambdaMART.__name__)

    def _prepare_train_data(self, X, Y, **kwargs):
        """
            Transform the data provided in the form of X_train of shape 
            (n_instances,n_objects,n_features) and y_train of shape 
            (n_instances,n_documents) into (n_instances*n_objects,n_features).
            The output format is similar to the oneprovided by the cusrom dataset reader.

            Parameters
            ---------
            X : numpy array
                (n_instances, n_objects, n_features)
                Feature vectors of the objects
            Y : numpy array
                (n_instances, n_objects)
                Rankings of the given objects
            Returns
            ------
            Returns an array of shape (n_instances*n_objects,n_features) with the 
            features and relevance scores derived from the ranking provided in y_train

        """
        # prepare array like features and imaginary qids
        xdim = X.shape[0]  # n_instances - qid
        ydim = X.shape[1]  # n_objects - documents
        zdim = X.shape[2]  # n_features

        features_as_list = deque()
        for i in range(0, xdim):
            for j in range(0, ydim):
                row_as_list = deque([i])
                features = deque()
                for k in range(0, zdim):
                    row_as_list.append(X[i, j, k])
                features_as_list.append(row_as_list)

        # Convert rankings to relevance scores
        scores_docsize = Y.shape[1]
        relscore_train = np.subtract(scores_docsize, Y)

        # prepare array like relevance score values
        xdim_scores = relscore_train.shape[0]
        ydim_scores = relscore_train.shape[1]

        scores_as_list = deque()
        for x in range(0, xdim_scores):
            for y in range(0, ydim_scores):
                scores_as_list.append(relscore_train[x, y])

        # Check if both the dimensions are the same
        assert len(features_as_list) == len(scores_as_list)

        # convert to numpy and resize the arrays
        features = np.asarray(features_as_list)
        scores_unflat = np.array(scores_as_list)
        scores = np.reshape(scores_unflat, (len(scores_unflat), 1))

        # Concatenate the reshaped arrays and return as trainin data
        train_data = np.concatenate((scores, features), axis=1)

        return train_data

    def _group_by_queries(self, data, queries):
        """
            Internal function which orders the data given as input based
            on the queries supplied.
        """
        result = []
        curr_query = None
        for s, q in zip(data, queries):
            if q != curr_query:
                result.append([])
                curr_query = q
            result[-1].append(s)
        result = list(map(np.array, result))
        return result

    def fit(self, X, y, **kwargs):
        """
           Fit a LambdaMART algorithm to the provided X and y arrays where X 
           contains the features and y being the relevance scores.

           Parameters
           ----------
           X : numpy array
               (n_instances, n_objects, n_features)
               Feature vectors of the objects
           Y : numpy array
               (n_instances, n_objects)
               Rankings of the given objects
           **kwargs
               Keyword arguments for the fit function
            
           Returns
           -------
           Returns the model which is in turn just a list of all the trees that
           make up the MART model

        """
        # check the case if the ensemble already has some trees then clear the trees
        # so that the trees from the previous iteration are not used.
        if len(self.ensemble) > 0:
            self.ensemble.clear()

        train_file = self._prepare_train_data(X, y)
        scores = train_file[:, 0]
        queries = train_file[:, 1]
        features = train_file[:, 2:]

        model_preds = np.zeros(len(features))

        for i in range(self.number_of_trees):
            true_data = self._group_by_queries(scores, queries)
            model_data = self._group_by_queries(model_preds, queries)

            with Pool(self.num_process) as pool:
                lambdas_draft = pool.map(
                    query_lambdas, list(zip(true_data, model_data))
                )
                lambdas = list(chain(*lambdas_draft))

            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter=self.splitter,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=None,
                random_state=self.random_state,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                min_impurity_split=self.min_impurity_split,
            )
            tree.fit(features, lambdas)

            self.ensemble.append(tree)

            prediction = tree.predict(features)
            model_preds += self.learning_rate * prediction
            # train_score = self._score(model_preds, scores, queries, 10)
            # print("iteration"+ i +" train score " + str(train_score)+" "+str(X.shape) + " and "+ str(y.shape))

    def _predict_scores_fixed(self, X, **kwargs):
        """
            Predict the scores for a given collection of sets of objects of same size.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_objects, n_features)


            Returns
            -------
            Y : array-like, shape (n_samples, n_objects)
                Returns the scores of each of the objects for each of the samples.
        """
        n_instances, n_objects, n_features = X.shape
        self.logger.info(
            "For Test instances {} objects {} features {}".format(*X.shape)
        )
        X1 = X.reshape(n_instances * n_objects, n_features)
        scores = np.zeros(n_instances * n_objects)
        for tree in self.ensemble:
            scores += tree.predict(X1)
        scores = scores.reshape(n_instances, n_objects)
        return scores

    def predict_scores(self, X, **kwargs):
        """
            Predict the utility scores for each object in the collection of set
            of objects called a query set.

            Parameters
            ----------
            X : numpy array of size (n_instances, n_objects, n_features)

            Returns
            -------
            Numpy array of size (n_instances, n_objects)
        """
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        """
         Predict rankings for the scores for a given collection of sets of objects
         (query sets). Wrapper that calls the function of the same name belonging
         to the ObjectRanker super class.
        """
        return ObjectRanker.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def _predict(self, pred_vector):
        """
            Predict the scores for the data supplied by iterating over the
            ensemble and returning the output.

            Parameters
            ----------
            pred_vector: this is a numpy array of shape (n_objects,n_features)

            Returns
            -------
            results: Predicted scores for each of the objects
            queries: queries corresponding to the predictions that are made
        """
        queries = pred_vector[:, 1]
        features = pred_vector[:, 2:]

        results = np.zeros(len(features))
        for tree in self.ensemble:
            results += tree.predict(features) * self.learning_rate
        return results, queries

    def _score(self, prediction, true_score, query, k=10):
        """
            Function that is used to score the performance of the model. 

            Parameters
            ----------
            prediction: Predictions of the model
            true_score: ground truth data of the predictions
            query: queries accompanying the prediction data used to calculate
            the ndcg value

            Returns
            -------
            Returns the average NDCG value calculated on the basis of the 
            queries supplied, for the predictions
        """
        true_data = self._group_by_queries(true_score, query)
        model_data = self._group_by_queries(prediction, query)

        total_ndcg = []

        for true_d, model_d in zip(true_data, model_data):
            data = true_d[np.argsort(model_d)[::-1]]
            total_ndcg.append(ndcg(data, k))

        return sum(total_ndcg) / len(total_ndcg)

    def set_tunable_parameters(
        self,
        min_samples_split=2,
        max_depth=50,
        min_samples_leaf=1,
        max_leaf_nodes=None,
        learning_rate=1e-3,
        number_of_trees=5,
        criterion="mse",
        splitter="best",
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        min_impurity_decrease=0.0,
        min_impurity_split=1e-7,
        **kwargs
    ):
        """
            Set the tunable hyperparameters of the DecisionTree model 
            used in LambdaMART

            Parameters
            ----------
            min_samples_split : int
                Number of samples required to split the internal node
            max_depth : int
                Maximum depth of the tree
            min_samples_leaf : int
                Minimum number of samples required to be at the leaf node
            max_leaf_nodes : int
                These are the maximum number of leaf nodes used to grow the tree
            number_of_trees : int
                The maximum number of trees that are to be trained for the ensemble.
            learning_rate : float
                learning rate for the LambdaMART algorithm
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.number_of_trees = number_of_trees
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.splitter = splitter
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


def query_lambdas(data, k=10):
    """
        This is used by the LambdaMART learner to compute the lambda values that 
        are to be used as the target variable for the learner.
        
        Parameters
        ----------
        data : This contains the training data and the predictions from the 
        previous iteration of the learning loop to calculate the lambda values

        Returns
        -------
        Returns the lambda values calculated for the current iteration
    """
    true_data, model_data = data
    worst_order = np.argsort(true_data)

    true_data = true_data[worst_order]
    model_data = model_data[worst_order]

    model_order = np.argsort(model_data)

    idcg = dcg(np.sort(true_data)[-10:][::-1])

    size = len(true_data)
    position_score = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            position_score[model_order[i], model_order[j]] = point_dcg(
                (model_order[j], true_data[model_order[i]])
            )

    lambdas = np.zeros(size)

    for i in range(size):
        for j in range(size):
            if true_data[i] > true_data[j]:

                delta_dcg = position_score[i][j] - position_score[i][i]
                delta_dcg += position_score[j][i] - position_score[j][j]

                delta_ndcg = abs(delta_dcg / idcg)

                rho = 1 / (1 + math.exp(model_data[i] - model_data[j]))

                lam = rho * delta_ndcg

                lambdas[j] -= lam
                lambdas[i] += lam
    return lambdas


def point_dcg(args):
    """
        Point DCG calculation function. Calculates the DCG for a given list. This
        list is assumed to be consisting of the rankings of documents belonging to
        the same query 
    """
    pos, label = args
    return (2 ** label - 1) / np.log2(pos + 2)


def dcg(preds):
    """
        List DCG calculation function. This function turns the list of rankings 
        into a form which is easier to be passed to the point DCG function
    """
    return sum(map(point_dcg, enumerate(preds)))


def ndcg(preds, k=10):
    """
        NDCG calculation function that calculates the NDCG values with the help 
        of the DCG calculation helper functions.
    """
    ideal_top = preds[:k]

    true_top = np.array([])
    if len(preds) > 10:
        true_top = np.partition(preds, -10)[-k:]
        true_top.sort()
    else:
        true_top = np.sort(preds)
    true_top = true_top[::-1]

    max_dcg = dcg(true_top)
    ideal_dcg = dcg(ideal_top)

    if max_dcg == 0:
        return 1

    return ideal_dcg / max_dcg
