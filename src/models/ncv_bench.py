#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from src.helpers.results import regression_results
import config
from config import model_dict


def ncv_bench(path, name, model_key, sparse_dict):
    # Import the chosen Model
    mod = __import__(f"src.models."
                     f"{model_dict[model_key]['dir']}."
                     f"{model_dict[model_key]['script']}", fromlist=['object'])
    model = getattr(mod, model_dict[model_key]['class'])
    # print(name)
    model = model(path, name)
    # Read Data and Create Output Folder
    model.read_data()
    model.create_output_dir()
    print(f'Workdir: {model.output_dir}')
    # Model to GPU if possible
    if model_dict[model_key]['dir'] == 'pytorch':
        model.to(model.device)
        print('Device: ', model.device)
    model.save_exp_def()

    # Start "outer loop CV"
    rs = ShuffleSplit(n_splits=config.OUTER_SPLITS, train_size=model.train_size, random_state=config.SEED)
    rs.get_n_splits(model.X)
    scores = pd.DataFrame(columns=['r2', 'rmse', 'mae', 'mse', 'mape', 'Task', 'split', 'mode'])
    i = 1

    for train_index, test_index in rs.split(model.X_data):
        print('{:-^60}'.format(f'SPLIT {i}'))
        model.create_sub_dirs(i)

        # Set and save indexes for data splitting
        model.train_index, model.test_index = train_index, test_index
        np.savetxt(os.path.join(model.split_path, 'train_index.txt'), [model.train_index], fmt="%d", delimiter=",")
        np.savetxt(os.path.join(model.split_path, 'test_index.txt'), [model.test_index], fmt="%d", delimiter=",")

        # Model Tuning and retraining
        y_pred = model.evaluate(sparse_dict, i)
        if y_pred.empty:
            break

        # Evaluation and saving of results
        y_true = model.y_data.loc[test_index]
        for n in range(model.num_tasks):
            if model_dict[model_key]['dir'] == 'pytorch':
                print(f'\nTask: {model.y_label[n]}')
                y_pred = y_pred.set_axis(y_true.index, axis='index')
                task_result = regression_results(y_true.iloc[:, n].dropna(),
                                                 y_pred.iloc[:, n][y_true.iloc[:, n].notna()])
                task_result.extend([model.y_label[n], i, model.model_name])
                scores.loc[len(scores)] = task_result
            if model_dict[model_key]['dir'] == 'auto_ml':
                for mode in ['STL'] + config.MTL_LIST:
                    print(f'Task: {model.y_label[n]}, Mode: {mode}')
                    task_result = regression_results(y_true.iloc[:, n].dropna(),
                                                     y_pred.loc[:, f'{model.y_label[n]}_{mode}'][y_true.iloc[:, n].notna()])
                    task_result.extend([model.y_label[n], i, mode])
                    scores.loc[len(scores)] = task_result

        scores.to_csv(os.path.join(model.output_dir, 'regression_summary.csv'), index=None)

        i += 1
