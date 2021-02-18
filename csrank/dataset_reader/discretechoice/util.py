import numpy as np
from sklearn.preprocessing import LabelBinarizer


def sub_sampling_discrete_choices_from_scores(Xt, Yt, n_objects=5):
    bucket_size = int(Xt.shape[1] / n_objects)
    X_train = []
    Y_train = []
    for i in range(bucket_size):
        X = np.copy(Xt)
        Y = np.copy(Yt)
        rs = np.random.RandomState(42 + i)
        idx = rs.randint(bucket_size, size=(len(X), n_objects))
        idx += np.arange(start=0, stop=X.shape[1], step=bucket_size)[:n_objects]
        X = X[np.arange(X.shape[0])[:, None], idx]
        Y = Y[np.arange(X.shape[0])[:, None], idx]
        Y = Y.argmax(axis=1)
        if len(X_train) == 0:
            X_train = X
            Y_train = Y
        else:
            Y_train = np.concatenate([Y_train, Y], axis=0)
            X_train = np.concatenate([X_train, X], axis=0)
    if len(Y_train) != 0:
        Y_train = convert_to_label_encoding(Y_train, n_objects)
    return X_train, Y_train


def sub_sampling_discrete_choices_from_relevance(Xt, Yt, n_objects=5):
    X_train = []
    Y_train = []
    bucket_size = int(Xt.shape[1] / n_objects)
    rs = np.random.RandomState(42 + n_objects)
    for x, y in zip(Xt, Yt):
        choices = np.append(np.where(y == 1)[0], np.where(y == 2)[0])
        zeros = np.delete(np.arange(0, Xt.shape[1]), choices)
        if len(zeros) != 0:
            if len(choices) > bucket_size:
                idx = choices
            else:
                idx = rs.choice(choices, size=bucket_size)
            if len(zeros) > n_objects - 1:
                idys = np.array(
                    [
                        rs.choice(zeros, size=(n_objects - 1), replace=False)
                        for i in range(len(idx))
                    ]
                )
            else:
                idys = rs.choice(zeros, size=(len(idx), (n_objects - 1)), replace=True)
            idx = np.append(idys, idx[:, None], axis=1)
            for i in idx:
                np.random.shuffle(i)
            if len(X_train) == 0:
                X_train = x[idx]
                Y_train = np.argmax(y[idx], axis=1)
            else:
                Y_train = np.concatenate([Y_train, np.argmax(y[idx], axis=1)])
                X_train = np.concatenate([X_train, x[idx]], axis=0)
    if len(Y_train) != 0:
        Y_train = convert_to_label_encoding(Y_train, n_objects)
    return X_train, Y_train


def convert_to_label_encoding(Y, n_objects):
    lb = LabelBinarizer().fit(np.arange(n_objects))
    return lb.transform(Y)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
