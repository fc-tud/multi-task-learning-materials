#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from contextlib import redirect_stdout
from datetime import datetime
import config
from src.models.auto_ml.base_auto_ml import BaseModelAutoML
from smac.utils.constants import MAXINT
import sklearn.metrics as metrics


from autosklearn.metrics import r2
from autosklearn.metrics import mean_squared_error as mse
from autosklearn.metrics import mean_absolute_error as mae
from autosklearn.metrics import median_absolute_error as mabse
from autosklearn.regression import AutoSklearnRegressor as Regressor


class AutoSklearn(BaseModelAutoML):
    def __init__(self, path, name):
        super().__init__(path, name)
        self.model_name = 'AutoSklearn'

    def train(self, modus, task_key, X_data, y_data):
        self.create_task_dirs(modus, task_key)

        # --- FRAMEWORK SPECIFIC START --- #1 -------------------------------------

        auto_tmp_folder = os.path.join(self.split_path, modus, task_key, 'autosklearn_tmp')

        self.model[modus][task_key] = Regressor(time_left_for_this_task=config.MAX_TIME_MINUTES * 60,
                                                per_run_time_limit=config.MAX_TIME_MINUTES * 5,
                                                resampling_strategy='cv',
                                                resampling_strategy_arguments={'folds': config.INNER_SPLITS},
                                                n_jobs=config.NUM_CORES,
                                                memory_limit=None,
                                                seed=config.SEED,
                                                metric=mse,
                                                scoring_functions=[r2, mse, mae, mabse],
                                                tmp_folder=auto_tmp_folder,
                                                delete_tmp_folder_after_terminate=True)
        # --- FRAMEWORK SPECIFIC END ----- #1 -------------------------------------

        print(datetime.now())
        print(f'Training in {modus} of {task_key} over {config.MAX_TIME_MINUTES} min started')
        self.model[modus][task_key].fit(X_data, y_data)
        print(datetime.now())
        pd.DataFrame(self.model[modus][task_key].cv_results_).to_csv(os.path.join(
            self.task_path[modus][task_key], 'leaderboard.csv'))

        # --- FRAMEWORK SPECIFIC START --- #2 -------------------------------------
        print(self.model[modus][task_key].sprint_statistics())

        # plot train hist
        ra = np.arange(len(self.model[modus][task_key].cv_results_['status']))
        test_score = self.model[modus][task_key].cv_results_['mean_test_score']
        test_score[test_score < 0] = 0

        best = []
        for i in test_score:
            try:
                best.append(max(0, max(best), i))
            except ValueError:  # best is empty
                best.append(0)
        best = np.array(best)

        labels = []
        for i in self.model[modus][task_key].cv_results_['params']:
            labels.append(i['regressor:__choice__'])
        labels = np.array(labels)

        df = pd.DataFrame(dict(x=ra, y=test_score, label=labels))
        groups = df.groupby('label')

        fig, ax = plt.subplots(figsize=(15, 6))
        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name)
            ax.legend(frameon=True, framealpha=0.9)
            ax.plot(ra, best, color='gray')
            ax.set_title('Algorithm Performance')
            ax.set_ylabel('Score')
            ax.set_xlabel('BaseModelPytorch No.')
        plt.savefig(os.path.join(self.task_path[modus][task_key], 'train_history.svg'))

        with open(os.path.join(self.task_path[modus][task_key], 'stats.txt'), 'w') as f:
            with redirect_stdout(f):
                print(self.model[modus][task_key].sprint_statistics())

        with open(os.path.join(self.task_path[modus][task_key], 'metric_results.txt'), 'w') as f:
            with redirect_stdout(f):
                print(self.get_metric_result(modus, task_key).to_string(index=False))

        # --- FRAMEWORK SPECIFIC End--- #2 -------------------------------------

    # --- FRAMEWORK SPECIFIC Start--- #3 -------------------------------------
    def get_mae(self, modus, task_key):
        results = pd.DataFrame.from_dict(self.model[modus][task_key].cv_results_)
        mae_run = results['metric_mean_absolute_error'].loc[results['rank_test_scores'] == 1]
        mae_run = mae_run.mean()
        return mae_run

    def get_metric_result(self, modus, task_key):
        results = pd.DataFrame.from_dict(self.model[modus][task_key].cv_results_)
        cols = ['rank_test_scores', 'param_regressor:__choice__',
                'mean_test_score']
        cols.extend([key for key in self.model[modus][task_key].cv_results_.keys() if key.startswith('metric_')])
        return results[cols]
    # --- FRAMEWORK SPECIFIC End--- #3 -------------------------------------
