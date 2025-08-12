import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.algorithms import linear_regression

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/lr.pkl")
with open(MODEL_PATH, "rb") as f:
    coefs = pickle.load(f)

model = linear_regression.LinearRegression()
model.coefs = coefs


def predict(features: list[float]) -> float:
    X = np.array(features).reshape(1, -1)
    pred = model.predict(X)
    return float(pred[0])
