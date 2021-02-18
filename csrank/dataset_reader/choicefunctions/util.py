import numpy as np


def sub_sampling_choices_from_relevance(Xt, Yt, n_objects=5, offset=2):
    X_train = []
    Y_train = []
    bucket_size = int(Xt.shape[1] / n_objects) + offset
    rs = np.random.RandomState(42 + n_objects)
    for x, y in zip(Xt, Yt):
        choices = np.append(np.where(y == 1)[0], np.where(y == 2)[0])
        zeros = np.delete(np.arange(0, Xt.shape[1]), choices)
        y[choices] = 1
        if len(zeros) != 0 and len(choices) >= 1:
            c_max = n_objects if n_objects < len(choices) else len(choices) + 1
            sizes = rs.choice(np.arange(1, c_max), size=bucket_size)
            if len(zeros) > n_objects - 1:
                idx = np.array(
                    [
                        np.append(
                            rs.choice(choices, size=size, replace=False),
                            rs.choice(zeros, size=(n_objects - size), replace=False),
                        )
                        for size in sizes
                    ]
                )
            else:
                idx = np.array(
                    [
                        np.append(
                            rs.choice(choices, size=size, replace=False),
                            rs.choice(zeros, size=(n_objects - size), replace=True),
                        )
                        for size in sizes
                    ]
                )
            for i in idx:
                np.random.shuffle(i)
            if len(X_train) == 0:
                X_train = x[idx]
                Y_train = y[idx]
            else:
                Y_train = np.concatenate([Y_train, y[idx]], axis=0)
                X_train = np.concatenate([X_train, x[idx]], axis=0)
    return X_train, Y_train
