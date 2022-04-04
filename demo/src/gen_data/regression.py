import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import random
from typing import Optional, Union

DataSplit = Union[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]


def generate_data(
    num_train: int,
    num_eval: int,
    num_features: int,
    seed: Optional[int] = None,
) -> DataSplit:
    num_informative = int(random.uniform(0, 1) * num_features)
    num_samples = num_train + num_eval

    noise = random.uniform(0, 5)
    bias = random.uniform(-10, 10)

    X, y = make_regression(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=num_informative,
        bias=bias,
        noise=noise,
        random_state=seed,
    )
    y = y[:, np.newaxis]
    if num_eval == 0:
        return X, y
    data = train_test_split(X, y, test_size=num_eval, random_state=seed)
    return data[0::2] + data[1::2]
