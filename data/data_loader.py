import pandas as pd
import os as os
from data_ready import data_github_monthly


def load_datasets(directory, goal):
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    for file in files:
        df = data_github_monthly(file, directory, goal)
        X, y = df.iloc[:,:-1], df.iloc[:,-1]
        yield X, y


if __name__ == "__main__":

    path = "data_selected"

    for X, y in load_datasets(path, 1):
        print(X.head)
        print(y.head)