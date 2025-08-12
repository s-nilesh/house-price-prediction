import random
import yaml
import os
import time


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"[Timing] Function '{func.__name__}' executed in {elapsed:.3f} seconds")
        return result

    return wrapper


class Config:
    def __init__(self, config_path="config/config.yml"):
        self.config_path = config_path
        self.config = None

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found at {self.config_path}"
            )
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
        return config


def train_test_split(X, y, test_size=0.25, shuffle=True, seed=None):
    if seed is not None:
        random.seed(seed)

    assert len(X) == len(y), "Error: length of X and y are not the same"

    n_samples = len(X)
    test_size = int(test_size * n_samples)
    indices = list(range(n_samples))

    if shuffle:
        random.shuffle(indices)

    train_indices = indices[test_size:]
    test_indices = indices[:test_size]
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test
