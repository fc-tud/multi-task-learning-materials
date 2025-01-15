import os
import numpy as np
import config
from datetime import datetime


def check_dataset(path):
    try:
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    except NotADirectoryError:
        print(path, 'is not a dataset directory!')
        files = []

    return files


def random_missing(data, nan_rate):
    np.random.seed(seed=config.SEED)
    random_array = np.random.rand(data.shape[0], data.shape[1])
    random_array = np.where(random_array > nan_rate, np.nan, 1)
    data_rand = data*random_array
    return data_rand


def get_time_from_folder_name(folder_name):
    # Split the folder name and extract the timestamp part
    timestamp_str = '_'.join(folder_name.split('_')[-7:])
    # Convert the timestamp string to a datetime object
    return datetime.strptime(timestamp_str, '%Y_%m_%d__%H_%M_%S')
