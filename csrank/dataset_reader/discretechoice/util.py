import numpy as np
from sklearn.preprocessing import LabelBinarizer


def sub_sampling_discrete_choices(Xt, St, n_objects=5):
    bucket_size = int(Xt.shape[1] / n_objects)
    X_train = []
    Y_train = []
    for i in range(bucket_size):
        X = np.copy(Xt)
        S = np.copy(St)
        rs = np.random.RandomState(42 + i)
        idx = rs.randint(bucket_size, size=(len(X), n_objects))
        idx += np.arange(start=0, stop=X.shape[1], step=bucket_size)[:n_objects]
        X = X[np.arange(X.shape[0])[:, None], idx]
        S = S[np.arange(X.shape[0])[:, None], idx]
        Y = S.argmax(axis=1)
        if len(X_train) == 0:
            X_train = X
            Y_train = Y
        else:
            Y_train = np.concatenate([Y_train, Y], axis=0)
            X_train = np.concatenate([X_train, X], axis=0)
    Y_train = convert_to_label_encoding(Y_train, n_objects)
    return X_train, Y_train


def convert_to_label_encoding(Y, n_objects):
    lb = LabelBinarizer().fit(np.arange(n_objects))
    return lb.transform(Y)
