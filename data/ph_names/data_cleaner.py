import pandas as pd
from datetime import datetime
import numpy as np
from scipy.io.arff import loadarff
import os


def data_remove_irrelevancy(repo_name, directory):
    df_raw = pd.read_csv(directory + repo_name, sep=',')
    for index, row in df_raw.iterrows():
        if int(row['dates'][0:4]) < 2015:
            df_raw.drop(index, inplace=True)
    for index, row in df_raw.iterrows():
        if int(row['number_of_open_PRs']) == 0 and int(row['number_of_open_issues']) == 0:
            df_raw.drop(index, inplace=True)
        else:
            break
    for index, row in df_raw.iterrows():
        if int(row['dates'][0:4]) == 2020 and int(row['dates'][6]) == 9:  # data before 2020-09-01
            df_raw.drop(index, inplace=True)
    file_output = r'../data/data_cleaned/{}'.format(repo_name)
    df_raw.to_csv(file_output, index=False, encoding='utf-8')
    return df_raw


if __name__ == '__main__':

    repo_pool = []
    path = r'../data/data_raw/'

    for filename in os.listdir(path):
        if not filename.startswith('.'):
            repo_pool.append(os.path.join(filename))

    for repo in repo_pool:
        data_remove_irrelevancy(repo, path)
        print(repo)

