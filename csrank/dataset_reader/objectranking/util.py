import numpy as np

__all__ = [
    "sub_sampling_rankings",
]


def sub_sampling_rankings(Xt, Yt, n_objects=5):
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
        tmp_sort = Y.argsort(axis=-1)
        Y = np.empty_like(Y)
        Y[np.arange(len(X))[:, None], tmp_sort] = np.arange(n_objects)
        if len(X_train) == 0:
            X_train = X
            Y_train = Y
        else:
            Y_train = np.concatenate([Y_train, Y], axis=0)
            X_train = np.concatenate([X_train, X], axis=0)
    return X_train, Y_train
