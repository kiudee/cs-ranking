import numpy as np
import logging

def sub_sampling_choices(name, Xt, St, n_objects=5):
    logger = logging.getLogger(name=name)
    bucket_size = int(Xt.shape[1] / n_objects)
    logger.info("###### X instances {} objects {} bucket_size {} ######".format(Xt.shape[0], Xt.shape[1], bucket_size))
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
    logger.info("Sampled instances {} objects {}".format(X_train.shape[0], X_train.shape[1]))
    return X_train, Y_train