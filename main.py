import pandas as pd
import numpy as np
import random
from house_price_prediction.src.algorithms.linear_regression import LinearRegression


def main():
    print("Hello from house-price-prediction!")
    data = pd.read_csv("data/Housing.csv")
    # print(data.head)
    # print(data.columns)

    X = np.array(data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',\
       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',\
       'parking', 'prefarea', 'furnishingstatus']])

    y = np.array(data["price"])

    print(len(X), len(y))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, 0.25, True, 42)
    # print(len)



if __name__ == "__main__":
    main()
