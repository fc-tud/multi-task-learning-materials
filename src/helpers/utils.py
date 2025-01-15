#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import config


def create_work_dir(model_name):
    output_dir = os.path.join('workdir', model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def split_with_index(index_list, n_splits, train_size, seed=config.SEED):
    np.random.seed(seed)
    splits = []
    index_list = list(set(index_list))
    for n in range(n_splits):
        train_index = np.random.choice(index_list, size=int(len(index_list)*train_size), replace=False)
        val_index = list(set(index_list) - set(train_index))
        splits.append([train_index, val_index])
    return splits


def truncate_log_file(file_path):
    try:
        with open(file_path, 'w') as file:
            pass  # Opening the file in 'w' mode clears its contents
        print(f"Log file '{file_path}' has been truncated.")
    except Exception as e:
        print(f"An error occurred while truncating the log file: {e}")
