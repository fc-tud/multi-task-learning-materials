#!/usr/bin/env python
# coding: utf-8


import os
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import json
import shutil

from src.models.auto_ml.base_auto_ml import BaseModelAutoML
import config

from supervised import AutoML
from sklearn.neighbors import KernelDensity
import sklearn.metrics as metrics


class MLjar(BaseModelAutoML):
    def __init__(self, path, name):
        super().__init__(path, name)
        self.model_name = 'MLjar'

    def train(self, modus, task_key, X_data, y_data):
        self.create_task_dirs(modus, task_key)
        # --- FRAMEWORK SPECIFIC START --- #1 -------------------------------------

        self.model[modus][task_key] = AutoML(mode='Compete',
                                             ml_task='regression',
                                             total_time_limit=config.MAX_TIME_MINUTES*60,
                                             validation_strategy={'validation_type': 'kfold',
                                                                  'k_folds': config.INNER_SPLITS,
                                                                  "shuffle": True,
                                                                  'random_seed': config.SEED},
                                             n_jobs=config.NUM_CORES,
                                             random_state=config.SEED,
                                             eval_metric='mse',
                                             results_path=self.task_path[modus][task_key])

        # --- FRAMEWORK SPECIFIC END ----- #1 -------------------------------------
        print('{:-^45}'.format(f'Training in {modus} of {task_key} over {config.MAX_TIME_MINUTES} min started'))
        print(datetime.now())
        # print(f'Training in {modus} of {task_key} over {config.MAX_TIME_MINUTES} min started')
        self.model[modus][task_key].fit(X_data, y_data)
        print(datetime.now())
        # print(self.model[modus][task_key]._best_model)

        # --- FRAMEWORK SPECIFIC START ----- #2 -------------------------------------

    def get_kde(self, modus, task_key):
        path_to_val_csv = glob.glob(os.path.join(self.task_path[modus][task_key],
                                                 self.model[modus][task_key]._best_model.get_name(),
                                                 'pred*.csv'))
        val_preds = pd.read_csv(os.path.join(path_to_val_csv[0]))
        val_preds['diff'] = val_preds['target'] - val_preds['prediction']
        self.kde[task_key] = KernelDensity(kernel='gaussian', bandwidth=1).fit(val_preds[['diff']])

    def get_mae(self, modus, task_key):
        path_to_val_csv = glob.glob(os.path.join(self.task_path[modus][task_key],
                                                 self.model[modus][task_key]._best_model.get_name(),
                                                 'pred*.csv'))
        val_preds = pd.read_csv(os.path.join(path_to_val_csv[0]))

        mae = metrics.mean_absolute_error(val_preds['target'], val_preds['prediction'])
        return mae

    # specific save and load function for MLjar
    def save_model(self, modus, task_key):
        return

    def load_model(self, path, mode, task_key):
        print('Load Model from Dir: ', path)
        print(f'Mode of Model: {mode}, Task: {task_key}')
        self.model[mode][task_key] = AutoML(results_path=os.path.join(path, task_key))

    def clean_dir(self):
        for key, sub_dict in self.task_path.items():
            for run, path in sub_dict.items():
                print(run)
                print(path)
                folder_list = os.listdir(path)
                folder_list = [n for n in folder_list if (n.split('_')[0].isdigit())]
                print(folder_list)
                if not folder_list:
                    continue
                with open(os.path.join(path, 'params.json'), 'r') as file:
                    params = json.load(file)
                pred_model_list = params['load_on_predict']
                del_list = [item for item in folder_list if item not in pred_model_list]
                print(del_list)
                for folder in del_list:
                    shutil.rmtree(os.path.join(path, folder))
        # --- FRAMEWORK SPECIFIC END ----- #2 -------------------------------------
