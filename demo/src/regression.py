import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import random
from typing import Optional

DataSplit = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def generate_data(
    n_train: int, n_eval: int, n_features: int, seed: Optional[int] = None
) -> DataSplit:
    n_informative = int(random.uniform(0, 1) * n_features)
    n_samples = n_train + n_eval

    noise = random.uniform(0, 5)
    bias = random.uniform(-10, 10)

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        bias=bias,
        noise=noise,
        random_state=seed,
    )
    y = y[:, np.newaxis]
    data_split = train_test_split(X, y, train_size=n_train, random_state=seed)
    return data_split
