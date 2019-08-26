import logging
from collections import deque
from multiprocessing import Pool

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from csrank.learner import Learner
from csrank.metrics import ndcg_at_k
from csrank.objectranking.object_ranker import ObjectRanker


class LambdaMART(ObjectRanker,Learner):
    def __init__(self, n_objects=None, n_object_features=None, number_of_trees=5, learning_rate=0.1,
                 min_samples_split=2, max_depth=50, min_samples_leaf=1, max_leaf_nodes=None):
        """
            Create a LambdaMART based rank regression model. This model uses an ensemble of trees that learn to predict
            the relevance scores of the documents based on the features, which then can be turned into rankings.
            The base learner used is the implementation of Decision Tree from the sklearn tree package. The learner
            tries to indirectly optimize the nDCG metric by learning the lambdas.

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
        """
        self.n_object_features = n_object_features
        self.n_objects = n_objects
        self.number_of_trees = number_of_trees
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.trees = []
        self.logger = logging.getLogger(LambdaMART.__name__)

    def _custom_letor_dataset_reader(self, filepath):
        """
            Custom dataset reader used for directly reading the separate files directory from a filepath.
            This will read through the provided files and extract the relevance score, query id and features from the text file provided

            Parameters
            ----------
            filepath : String
                Text path containing the path including the filename that is to be read

            Returns
            -------
            data : numpy array
                (n_instances,2 + n_objects + n_features)
                A numpy array that has all the data converted into rows containing relevance score, query_id and the features.
        """
        f = open(filepath, 'r')
        data = []
        for line in f:
            new_arr = []
            arr = line.split(' #')[0].split()
            score = arr[0]
            q_id = arr[1].split(':')[1]
            new_arr.append(int(score))
            new_arr.append(int(q_id))
            arr = arr[2:]
            for el in arr:
                new_arr.append(float(el.split(':')[1]))
            data.append(new_arr)
        f.close()
        return np.array(data)

    def _prepare_train_data(self, X, Y, **kwargs):
        """
            Transform the data provided in the form of X_train of shape (n_instances,n_objects,n_features) and y_train of shape (n_instances,n_documents) into (n_instances*n_objects,n_features). The output format is similar to the oneprovided by the cusrom dataset reader.

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
            Returns an array of shape (n_instances*n_objects,n_features) with the features and relevance scores derived from the ranking provided in y_train

        """
        #prepare array like features and imaginary qids
        xdim = X.shape[0]  # n_instances - qid
        ydim = X.shape[1]  # n_objects - documents
        zdim = X.shape[2]  # n_features

        features_as_list = deque()
        for i in range(0,xdim):
            for j in range(0,ydim):
                row_as_list=deque([i])
                features = deque()
                for k in range(0, zdim):
                    row_as_list.append(X[i, j, k])
                features_as_list.append(row_as_list)

        #Convert rankings to relevance scores     
        scores_docsize = Y.shape[1]
        relscore_train = np.subtract(scores_docsize, Y)

        #prepare array like relevance score values
        xdim_scores = relscore_train.shape[0]
        ydim_scores = relscore_train.shape[1]

        scores_as_list = deque()
        for x in range(0,xdim_scores):
            for y in range(0,ydim_scores):
                scores_as_list.append(relscore_train[x,y])
        
        #Check if both the dimensions are the same
        assert(len(features_as_list)==len(scores_as_list))
        
        #convert to numpy and resize the arrays 
        features = np.asarray(features_as_list)
        scores_unflat = np.array(scores_as_list)
        scores = np.reshape(scores_unflat,(len(scores_unflat),1))

        #Concatenate the reshaped arrays and return as trainin data
        train_data = np.concatenate((scores,features),axis=1)

        return train_data

    def _i_dcg(self,scores,k=None):
        """
            Returns the ideal DCG value for the given scores
        """
        scores = [score for score in sorted(scores)[::-1]]
        if k == None:
            return np.sum([(np.power(2, scores[i]) - 1) / np.log2(i + 2) for i in range(len(scores))])

        elif k > 0:
            return np.sum([(np.power(2, scores[i]) - 1) / np.log2(i + 2) for i in range(len(scores[:k]))])

    def _group_queries(self, training_data, qid_index):
        query_indexes = {}
        index = 0
        for record in training_data:
            query_indexes.setdefault(record[qid_index], [])
            query_indexes[record[qid_index]].append(index)
            index += 1
        return query_indexes

    def fit(self, X, y, **kwargs):
        """
            Fit a LambdaMART algorithm to the provided X and y arrays where X contains the features and y being the relevance scores.

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

        """

        training_data = self._prepare_train_data(X, y)
        predicted_scores = np.zeros(len(training_data))
        query_indexes = self._group_queries(training_data, 1)
        query_keys = query_indexes.keys()
        true_scores = [training_data[query_indexes[query], 0] for query in query_keys]

        sorted_query_pair = []
        for query_scores in true_scores:
            temp = sorted(query_scores, reverse=True)
            pairs = []
            for i in range(len(temp)):
                for j in range(len(temp)):
                    if temp[i] > temp[j]:
                        pairs.append((i,j))
            sorted_query_pair.append(pairs)

        idcg = [self._i_dcg(scores) for scores in true_scores]

        for k in range(self.number_of_trees):
            print("Tree %d" % (k))
            lambdas = np.zeros(len(predicted_scores))
            w = np.zeros(len(predicted_scores))
            pred_scores = [predicted_scores[query_indexes[query]] for query in query_keys]
            
            pool = Pool()
            for lambda_val, w_val, query_key in pool.map(compute_lambda_weights, zip(true_scores, pred_scores, sorted_query_pair, idcg, query_keys), chunksize=1):
                indexes = query_indexes[query_key]
                lambdas[indexes] = lambda_val
                w[indexes] = w_val
            pool.close()
            # filename = str("lmartDebug2_"+str(k)+"_lambdas.txt")
            # np.savetxt(filename, lambdas)

            #Hyperparameters to be considered are :"min_samples_split","max_depth","min_samples_leaf","max_leaf_nodes"
            tree = DecisionTreeRegressor(criterion="mse",
                                         splitter="best",
                                         max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split,
                                         min_samples_leaf=self.min_samples_leaf,
                                         min_weight_fraction_leaf=0.,
                                         max_features=None,
                                         random_state=9,
                                         max_leaf_nodes=self.max_leaf_nodes,
                                         min_impurity_decrease=0.,
                                         min_impurity_split=None)
            tree.fit(training_data[:,2:], lambdas)
            self.trees.append(tree)
            prediction = tree.predict(training_data[:,2:])
            predicted_scores += prediction * self.learning_rate
    
    def _predict_scores_fixed(self, X, **kwargs):
        n_instances, n_objects, n_features = X.shape
        self.logger.info("For Test instances {} objects {} features {}".format(*X.shape))
        X1 = X.reshape(n_instances * n_objects, n_features)
        scores = np.zeros(n_instances * n_objects)
        for tree in self.trees:
            scores += tree.predict(X1)
        scores = scores.reshape(n_instances, n_objects)
        return scores

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ObjectRanker.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)
    
    def _predict(self, data):
        """
            Predict the
        """        
        data = np.array(data)
        query_indexes = self._group_queries(data, 0)
        predicted_scores = np.zeros(len(data))
        for query in query_indexes:
            results = np.zeros(len(query_indexes[query]))
            for tree in self.trees:
                results += self.learning_rate * tree.predict(data[query_indexes[query], 1:])
            predicted_scores[query_indexes[query]] = results
        return predicted_scores

    def _validate(self, data, k=10):
        data = np.array(data)
        query_indexes = self._group_queries(data, 1)
        average_ndcg = []
        predicted_scores = np.zeros(len(data))
        for query in query_indexes:
            results = np.zeros(len(query_indexes[query]))
            for tree in self.trees:
                results += self.learning_rate * tree.predict(data[query_indexes[query], 2:])
            predicted_sorted_indexes = np.argsort(results)[::-1]
            t_results = data[query_indexes[query], 0]
            t_results = t_results[predicted_sorted_indexes]
            predicted_scores[query_indexes[query]] = results
            ndcg_val = ndcg_at_k(r=t_results, k=10)
            self.logger.info('Qid: {}, ndcg {} '.format(query, ndcg_val))
            average_ndcg.append(ndcg_val)
        average_ndcg = np.nanmean(average_ndcg)
        return average_ndcg, predicted_scores

    def set_tunable_parameters(self, min_samples_split, max_depth, min_samples_leaf, max_leaf_nodes, number_of_trees=5,
                               learning_rate=1e-4, **kwargs):
        """
            Set the tunable hyperparameters of the DecisionTree model used in LambdaMART

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


def compute_lambda_weights(args):
    """
        This is used by the LambdaMART learner. Computes the lambda targets that are considered to be the relationship between the different features and the ranking of a document. These lambdas are the learning target for the base learner.
        The parameters to this function are passed in the form of a zipped iterator.

        Parameters
        ----------

        true_scores : numpy array
            This contains the ground truth relevance scores of the document
        predicted_scores : numpy array
            This contains the scores predicted by the base learner in the current state of training
        sorted_pairs : numpy array
            This contains the indexes of the rankings inside the provided data where the documents with higher relevance are ranked higher.
        idcg : numpy array
            This contains the ideal dcg values, which are the best possible rankings for the set of documents provided
        query_keys : numpy array
            These are all the distinct query ids present in the provided data.
        Returns
        ------
        lambdas : numpy array
            The calculated lambda values for the documents provided
        weights : numpy array
            These are the weights calculated along with the lambdas for the documents provided
        query_keys : numpy array
            These are the distinct query ids in the provided data

    """
    true_scores, predicted_scores, sorted_pairs, idcg, query_key = args
    num_docs = len(true_scores)
    sorted_indexes = np.argsort(predicted_scores)[::-1]
    rev_indexes = np.argsort(sorted_indexes)
    true_scores = true_scores[sorted_indexes]
    predicted_scores = predicted_scores[sorted_indexes]

    lambdas = np.zeros(num_docs)
    w = np.zeros(num_docs)

    dcg_mat = {}
    for i, j in sorted_pairs:
        if (i, i) not in dcg_mat:
            dcg_mat[(i, i)] = _single_dcg(true_scores, i, i)
        dcg_mat[(i, j)] = _single_dcg(true_scores, i, j)
        if (j, j) not in dcg_mat:
            dcg_mat[(j, j)] = _single_dcg(true_scores, j, j)
        dcg_mat[(j, i)] = _single_dcg(true_scores, j, i)

    for i, j in sorted_pairs:
        z_ndcg = abs(dcg_mat[(i, j)] - dcg_mat[(i, i)] + dcg_mat[(j, i)] - dcg_mat[(j, j)]) / idcg
        rho = 1 / (1 + np.exp(predicted_scores[i] - predicted_scores[j]))
        rho_complement = 1.0 - rho
        lambda_val = z_ndcg * rho
        lambdas[i] += lambda_val
        lambdas[j] -= lambda_val

        w_val = rho * rho_complement * z_ndcg
        w[i] += w_val
        w[j] += w_val

    return lambdas[rev_indexes], w[rev_indexes], query_key


def _single_dcg(scores, i, j):
    "This is used by the LambdaMART learner. Computes a discounted cumilative gain DCG for given scores"
    return (np.power(2, scores[i]) - 1) / np.log2(j + 2)
