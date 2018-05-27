import itertools as iter
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def strongly_connected_components(graph):
    """ Find the strongly connected components in a graph using
        Tarjan's algorithm.
        # Taken from http://www.logarithmic.net/pfh-files/blog/01208083168/sort.py

        graph should be a dictionary mapping node names to
        lists of successor nodes.
        """

    result = []
    stack = []
    low = {}

    def visit(node):
        if node in low: return

        num = len(low)
        low[node] = num
        stack_pos = len(stack)
        stack.append(node)

        for successor in graph[node]:
            visit(successor)
            low[node] = min(low[node], low[successor])

        if num == low[node]:
            component = tuple(stack[stack_pos:])
            del stack[stack_pos:]
            result.append(component)
            for item in component:
                low[item] = len(graph)

    for node in graph:
        visit(node)

    return result


def create_graph_pairwise_matrix(matrix):
    n_objects = matrix.shape[0]
    graph = {key: [] for key in np.arange(n_objects)}
    for i, j in iter.combinations(np.arange(n_objects), 2):
        p_ij = matrix[i, j]
        p_ji = matrix[j, i]
        if (p_ij > p_ji):
            graph[j].append(i)
        if (p_ij < p_ji):
            graph[i].append(j)
        if (p_ij == p_ji):
            graph[j].append(i)
            graph[i].append(j)
    return graph


def create_pairwise_prob_matrix(n_objects):
    # Create a non-transitive pairwise probability matrix for n_objects*n_objects
    non_transitive = False
    while (not non_transitive):
        pairwise_prob = np.zeros([n_objects, n_objects])
        for i, j in iter.combinations(np.arange(n_objects), 2):
            pairwise_prob[i, j] = np.random.rand(1)[0]
            pairwise_prob[j, i] = 1.0 - pairwise_prob[i, j]
        for comp in strongly_connected_components(create_graph_pairwise_matrix(pairwise_prob)):
            if (len(comp) >= 3):
                non_transitive = True
                break
    return pairwise_prob


def quicksort(arr, matrix):
    # Apply the quick sort algorithm for the given set of objects and produces an ordering based on provided pairwise matrix
    if len(arr) < 2:
        return arr
    else:
        pivot = np.random.choice(arr, 1)[0]
        arr.remove(pivot)
        right = [i for i in arr if matrix[pivot, i] == 1]
        left = [i for i in arr if matrix[pivot, i] == 0]
        return quicksort(left, matrix) + [pivot] + quicksort(right, matrix)


def get_similarity_matrix(mypath):
    dataFrame = pd.read_csv(mypath)
    similarity_dictionary = dataFrame.set_index('col_major_index')['similarity'].to_dict()
    return similarity_dictionary


def get_key_for_indices(idx1, idx2):
    return str(tuple(sorted([idx1, idx2])))


def weighted_cosine_similarity(weights):
    def distance_function(x, y):
        denominator = np.sqrt(np.sum(weights * x * x)) * np.sqrt(
            np.sum(weights * y * y))
        sim = np.sum(weights * x * y) / denominator
        return 1 - sim

    return distance_function


def distance_metric_multilabel(X_labels, Y_labels, X, Y):
    similarity = f1_score(X_labels, Y_labels, average='macro')
    similarity = np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y)) + similarity
    return similarity


def print_no_newline(i, total):
    sys.stdout.write("Iterations: {}  out of {} \r".format(i, total))
    sys.stdout.flush()


