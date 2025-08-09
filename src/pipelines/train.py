import numpy as np
import src.algorithms.linear_regression import LinearRegression
from src.data.data_loader import load_data
from src.utils import utils

conf = utils.Config()
conf = conf.load_config()

data = load_data(conf.get("data").get("train_file"))

